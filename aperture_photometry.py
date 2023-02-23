# @Author: Andrés Gúrpide <agurpide>
# @Date:   19-10-2020
# @Email:  a.gurpide-lasheras@soton.ac.uk
# @Last modified by:   agurpide
# @Last modified time: 22-02-2023

import os
from regions import Regions
from photutils.aperture import aperture_photometry, ApertureStats, CircularAperture
from astropy.io import fits
import argparse
from astropy.stats import SigmaClip
from numpy import log10, sqrt
import astropy.units as u
from astropy import wcs
import matplotlib.pyplot as plt
from astropy.visualization import simple_norm
import logging
import hst_utils as hst_ut
import numpy as np
from extinction import ccm89, remove
import stsynphot as stsyn
from synphot import Observation
from astropy.time import Time
import warnings
from photutils.centroids import centroid_2dg
import subprocess

def stmag(flux, zeropint=-21.1):
    """Returns the ST magnitude as defined in https://hst-docs.stsci.edu/acsdhb/chapter-5-acs-data-analysis/5-1-photometry"""
    return -2.5 * np.log10(flux) + zeropint



Rv = 3.1

parser = argparse.ArgumentParser(description='Extracts fluxes from the given apertures.')
parser.add_argument("images", help="Image files where to extract fluxes", nargs='+', type=str)
parser.add_argument("-s", "--source", type=str, help='Source extraction region file to use for the aperture photometry', nargs=1)
parser.add_argument("-b", "--background", type=str, help='Background extraction region file to use for the aperture photometry (optional)', nargs="?")
parser.add_argument("-e", "--exclude", type=str, help='File with extraction regions to exclude from the source aperture photometry', nargs="?")
parser.add_argument("--av", type=float, help='Extinction correction to obtain derredened fluxes. Uses Cardelli+89 and Rv=3.1', nargs="?")
parser.add_argument("-a", "--aperture_correction", type=float,
                    help='Aperture correction (see https://stsci.edu/hst/instrumentation/wfc3/data-analysis/photometric-calibration/uvis-encircled-energy and https://stsci.edu/hst/instrumentation/acs/data-analysis/aperture-corrections)',
                    nargs="?", default=1)
parser.add_argument("--uncertainty", type=float,
                    help='Optional uncertainty on the aperture correction if any')
args = parser.parse_args()

source_reg = Regions.read(args.source[0], format="ds9")[0]

for image_file in args.images:
    if os.path.isfile(image_file):
        hst_hdul = fits.open(image_file)
        date = hst_hdul[0].header["DATE-OBS"]
        obsdate = Time(date).mjd
        #hst_filter = hst_ut.get_image_filter(hst_hdul[0].header)
        instrument = hst_hdul[0].header["INSTRUME"]
        obs_mode = hst_hdul[1].header["PHOTMODE"]
        if "WFC3" in obs_mode:
            keywords = obs_mode.split(" ")
            obsmode = "%s,%s,%s,mjd#%.2f" % (keywords[0], keywords[1], keywords[2], obsdate)
        elif "WFC2" in obs_mode:
            keywords = obs_mode.split(",")
            #['WFPC2', '1', 'A2D7', 'F656N', '', 'CAL']
            obsmode = "%s,%s,%s,%s,%s" % (keywords[0], keywords[1], keywords[2],  keywords[3], keywords[5])
        elif "WFC1" in obs_mode or "HRC" in obs_mode:
            keywords = obs_mode.split(" ")
            obsmode = "%s,%s,%s,%s" % (keywords[0], keywords[1], keywords[2],  keywords[3])
        hst_filter = keywords[2]
        # add aperture radius for correction
        obsmode += ",aper#%.2f" % source_reg.radius.to(u.arcsec).value
        bp = stsyn.band(obsmode)
        # https://stsynphot.readthedocs.io/en/latest/stsynphot/tutorials.html calculate vega zero point
        obs = Observation(stsyn.Vega, bp, binset=bp.binset)
        vega_zpt = obs.effstim(flux_unit=(u.erg/u.cm**2/u.AA/u.s), area=stsyn.conf.area)
        print(f'VEGAMAG zeropoint for {bp.obsmode} is {vega_zpt:.5E}')
        rect_width = bp.rectwidth()
        units = hst_hdul[1].header["BUNIT"]
        exp_time = float(hst_hdul[0].header["EXPTIME"])
        detector = hst_hdul[0].header["DETECTOR"] if "DETECTOR" in hst_hdul[0].header else ""
        pivot_wavelength = float(hst_hdul[1].header["PHOTPLAM"])
        filter_bandwidth = float(hst_hdul[1].header["PHOTBW"])
        #rect_width = filter_bandwidth * u.AA
        # if UV filter then https://www.stsci.edu/files/live/sites/www/files/home/hst/instrumentation/wfc3/documentation/instrument-science-reports-isrs/_documents/2017/WFC3-2017-14.pdf
        # use phftlam1 keyword for UV filters
        uv_filters = ["F200LP", "F300X", "F218W", "F225W", "F275W", "FQ232N", "FQ243N", "F280N"]
        if detector == "UVIS" and filter in uv_filters:
            photflam = float(hst_hdul[0].header["PHTFLAM1"])
        elif "PHOTFLAM" in hst_hdul[0].header:
            photflam = float(hst_hdul[0].header["PHOTFLAM"])
        elif "PHOTFLAM" in hst_hdul[1].header:
            photflam = float(hst_hdul[1].header["PHOTFLAM"])
        photflam = photflam * u.erg / u.AA / u.s / u.cm**2
        print("PHOTFLAM keyword value: %.2E %s" % (photflam.value, photflam.unit))
        zero_point = float(hst_hdul[1].header["PHOTZPT"])

        image_data = hst_hdul[1].data
        # divide by the exposure time if needed
        if "/S" not in units:
            print("Units: %s. Applying exposure time correction" % units)
            image_data /= exp_time
            units += "/S"
        hst_wcs = wcs.WCS(hst_hdul[1].header)
        source_aperture = hst_ut.region_to_aperture(source_reg, hst_wcs)
        src_mask = source_aperture.to_mask()
        cutout = src_mask.cutout(image_data)
        err_cutout = src_mask.cutout(np.sqrt(image_data * exp_time) / exp_time)
        mask_cutout = cutout <=0
        print("Refining source centroid...")
        x_cut, y_cut = centroid_2dg(cutout, error=err_cutout, mask=mask_cutout)
        # convert to image coordinates
        src_x, src_y = x_cut + source_aperture.bbox.ixmin, y_cut + source_aperture.bbox.iymin
        mask = image_data < 0
        # use the same radius but now the source is centered
        source_aperture = CircularAperture([src_x, src_y], source_aperture.r)
        print("Creating extraction region plot...")
        fig, ax = plt.subplots(1 , subplot_kw={'projection': hst_wcs})
        plt.imshow(image_data, norm=simple_norm(image_data, 'linear', percent=99),
                   interpolation='nearest', cmap="cividis")
        plt.scatter(src_x, src_y, marker="x", color="red", label="Centroid")
        plt.xlim(src_x - 50, src_x + 50)
        plt.ylim(src_y - 50, src_y + 50)
        source_aperture.plot(color='white', lw=1,
                                   label='r = %d' % source_aperture.r)
        plt.legend()
        plt.xlabel("Ra")
        plt.ylabel("Dec")
        plt.savefig("src_extraction_%s" % (args.source[0].replace(".reg",".png")), dpi=200)
        plt.close(fig)
        # new aperture with refined centroid
        phot_source = aperture_photometry(image_data, source_aperture,
                       error=np.sqrt(image_data * exp_time) / exp_time, wcs=hst_wcs, mask=mask)
        source_area = source_aperture.area
        aperture_keyword = "corrected_aperture_sum(%s)" % units

        if args.exclude is not None:
            for exclude_reg in Regions.read(args.exclude, format="ds9"):
                exclude_aperture = hst_ut.region_to_aperture(exclude_reg, hst_wcs)
                phot_exclude = aperture_photometry(image_data, exclude_aperture,
                                                   wcs=hst_wcs, error=np.sqrt(image_data * exp_time) / exp_time,
                                                   mask=mask)
                source_area  -= exclude_aperture.area
                phot_source["aperture_sum_err"] = np.sqrt(phot_exclude["aperture_sum_err"] ** 2 + phot_source["aperture_sum_err"] ** 2)
                phot_source["aperture_sum"] -= phot_exclude["aperture_sum"]

        # if a background region was given
        if args.background is not None:
            bkg_reg = Regions.read(args.background, format="ds9")[0]
            sigma = 4
            sigclip = SigmaClip(sigma=sigma, maxiters=10)
            bkg_aperture = hst_ut.region_to_aperture(bkg_reg, hst_wcs)
            phot_bkg = ApertureStats(image_data, bkg_aperture, sigma_clip=sigclip,
                                     error=np.sqrt(image_data * exp_time) / exp_time,
                                     mask=mask)
            # save sigma clipping algorithm
            fig = plt.figure()
            plt.hist(ApertureStats(image_data, bkg_aperture, sigma_clip=None,
                                     error=np.sqrt(image_data * exp_time) / exp_time,
                                     mask=mask).data_cutout.flatten())
            plt.axvline(phot_bkg.max, ls="--", label="%d$\sigma$" % sigma,
                        color="black")
            plt.xlabel("Background count rates (ct/s)")
            plt.legend()
            plt.savefig("bkg_clipping.png", dpi=100)
            plt.close(fig)

            #phot_bkg = aperture_photometry(image_data, bkg_aperture, wcs=hst_wcs, error=np.sqrt(image_data * exp_time) / exp_time, mask=mask)

            #phot_source[aperture_keyword] = (phot_source["aperture_sum"] - phot_bkg["aperture_sum"] / bkg_aperture.area * source_area) / args.aperture_correction
            phot_source[aperture_keyword]= phot_source["aperture_sum"] - phot_bkg.median * source_area # bkg subtracted counts
            # the uncertainty on the bkg is the standard error or RMSE
            bkg_unc = phot_bkg.std / np.sqrt(len(phot_bkg.data_cutout))
            phot_source["corrected_aperture_err"] = sqrt(phot_source["aperture_sum_err"] ** 2 + bkg_unc ** 2) / args.aperture_correction

        else:
            logging.warning("No background was given, no background correction will be performed.")
            phot_source[aperture_keyword] = phot_source["aperture_sum"] / args.aperture_correction
            phot_source["aperture_correction"] = args.aperture_correction
            if args.uncertainty is None:
                phot_source["corrected_aperture_err"] = phot_source["aperture_sum_err"] / args.aperture_correction
            else:
                phot_source["corrected_aperture_err"] = np.sqrt((phot_source["aperture_sum_err"] / args.aperture_correction)**2 + (phot_source[aperture_keyword] * args.uncertainty/ args.aperture_correction)**2)
                phot_source["aperture_correction_err"] = args.uncertainty
        phot_source_conf_pos = phot_source[aperture_keyword] + phot_source["corrected_aperture_err"]
        phot_source_conf_neg = phot_source[aperture_keyword] - phot_source["corrected_aperture_err"]


        flux_header = "flux(%s)" % photflam.unit
        flux_density = phot_source[aperture_keyword] * photflam
        phot_source[flux_header] = phot_source[aperture_keyword] * photflam
        phot_source["flux_err"] = phot_source_conf_pos * photflam - phot_source[flux_header]
        # "integrated" over the filter filter_bandwidth
        flux_bp = "int_flux(%s)" % (photflam.unit * rect_width.unit)
        phot_source[flux_bp] = flux_density * rect_width
        phot_source["int_flux_err"] = phot_source_conf_pos * photflam * rect_width - phot_source[flux_bp]
        # more appropiate for emission lines
        line_flux = "monochromatic_flux(%s)" % (photflam.unit)
        phot_source[line_flux] = phot_source[aperture_keyword] * bp.emflx(bp.area)
        phot_source["monochromatic_flux_err"] = phot_source_conf_pos * bp.emflx(bp.area) - phot_source[line_flux]
        #https://www.stsci.edu/hst/instrumentation/acs/data-analysis/zeropoints
        # be careful as the ZP is simply -21.1, the zeropoint for ACS needs to be computed
        ##phot_source["acsmag"] = -2.5 * log10(phot_source[aperture_keyword]) - zero_point # in e/s
        ##phot_source["acsmag_err_pos"] = -2.5 * log10(phot_source_conf_pos) - zero_point - phot_source["acsmag"]
        ##phot_source["acsmag_err_neg"] = phot_source["acsmag"] - (-2.5 * log10(phot_source_conf_neg) - zero_point)

        phot_source["vegamag"] = -2.5 * log10(flux_density.value /  vega_zpt.value)
        # this follows from error propagation
        phot_source["vegamag_err"] = phot_source["flux_err"] / (flux_density.value * np.log(10))

        phot_source["stmag"] = stmag(flux_density.value)
        phot_source["stmag_err"] = phot_source["flux_err"] / (flux_density.value * np.log(10))

        if args.av is not None:
            waves = np.array([pivot_wavelength])
            phot_source["der_flux"] = remove(ccm89(waves, args.av, Rv, unit="aa"), phot_source[flux_header])
            phot_source["der_flux_err"] = remove(ccm89(waves, args.av, Rv, unit="aa"), phot_source_conf_pos * photflam) - phot_source["der_flux"]
            phot_source['der_flux'].info.format = '%.2E'
            phot_source['der_flux_err'].info.format = '%.2E'
            phot_source["int_der_flux"] = phot_source["der_flux"]  * rect_width.value
            phot_source["int_der_flux_err"] = remove(ccm89(waves, args.av, Rv, unit="aa"), phot_source_conf_pos * photflam)  * rect_width.value - phot_source["int_der_flux"]
            phot_source["dervegamag"] = -2.5 * log10(phot_source["der_flux"].value  /  vega_zpt.value )
            # negative and positive errors become swapped
            phot_source["dervegamag_err"] = phot_source["der_flux_err"] / (phot_source["der_flux"] * np.log(10))
            phot_source["derstmag"] = stmag(phot_source["der_flux"].value)
            phot_source["derstmag_err"] = phot_source["der_flux_err"] / (phot_source["der_flux"] * np.log(10))
            phot_source["dervegamag"].info.format = "%.3f"
            phot_source["dervegamag_err"].info.format = "%.3f"
            phot_source["derstmag"].info.format = "%.3f"
            phot_source["derstmag_err"].info.format = "%.3f"

        # formatting
        phot_source["xcenter"].info.format = '%.2f'
        phot_source["ycenter"].info.format = '%.2f'
        phot_source["aperture_sum"].info.format = '%.3f'
        phot_source[aperture_keyword].info.format = '%.2f'
        phot_source["aperture_sum_err"].info.format = '%.2f'
        phot_source["corrected_aperture_err"].info.format = '%.2f'
        phot_source[flux_header].info.format = '%.3E'
        phot_source['flux_err'].info.format = '%.2E'
        phot_source[flux_bp].info.format = "%.3E"
        phot_source["int_flux_err"].info.format = "%.2E"
        phot_source[line_flux].info.format = "%.3E"
        # be careful as the ZP is simply -21.1, the zeropoint for ACS needs to be computed
        #phot_source["acsmag"].info.format = "%.2f"
        #phot_source["acsmag_err_neg"].info.format = "%.2f"
        #phot_source["acsmag_err_pos"].info.format = "%.3f"
        phot_source["vegamag"].info.format = "%.3f"
        phot_source["vegamag_err"].info.format = "%.3f"
        phot_source["stmag"].info.format = "%.2f"
        phot_source["stmag_err"].info.format = "%.3f"

        reg_basename = os.path.basename(args.source[0]).replace('.reg', '')
        out_data_file = "aperture_phot_%s_%s.csv" % (hst_filter, reg_basename)
        phot_source.write(out_data_file, overwrite=True,
                         format="ascii.commented_header", delimiter=",")
        out_info_file = image_file.replace(".fits", "%s_apt_info.txt" % reg_basename)
        f = open(out_info_file, "w+")
        f.write("Date:%s\nInstrument:%s\nFilter:%s\nDetector:%s\nExposure(s):%.2f\nPivot wavelength (A):%.1f\nRMS:%.1f\nPHOTFLAM:%s\nAperture correction:%.4f" % (date, instrument, hst_filter, detector, exp_time, pivot_wavelength, filter_bandwidth, photflam.value, args.aperture_correction))
        aperture_reg = type(source_aperture).__name__
        attributes = vars(source_aperture)
        att_list = ''.join("%s: %s" % item for item in attributes.items())
        f.write("\n###Aperture details###\n%s\n%s\n#XSPEC command\n" % (aperture_reg, att_list))
        xspec_outfile = "xspec_" + image_file.replace(".fits", ".txt")
        f.write("ftflx2xsp infile=%s xunit=angstrom yunit=ergs/cm^2/s/A nspec=1 phafile=hst_%s.fits rspfile=hst_%s.rsp"% (xspec_outfile, hst_filter, hst_filter))
        f.close()

        f = open("%s" % ("%s" % xspec_outfile), "w+")
        f.write("%.2f %.2f %.3e %.3e\n" % (pivot_wavelength - filter_bandwidth / 2, pivot_wavelength + filter_bandwidth / 2, phot_source[flux_header].value, phot_source['flux_err'].value))
        f.close()

        print("Use 'ftflx2xsp infile=%s xunit=angstrom yunit=ergs/cm^2/s/A nspec=1 phafile=hst_%s.fits rspfile=hst_%s.rsp' to convert to XSPEC format" % (xspec_outfile, hst_filter, hst_filter))
        print("Output stored to %s and %s" % (out_info_file, out_data_file))
        #
        # print some useful info
    else:
        warnings.warn("File %s not found, skipping it")

# @Author: Andrés Gúrpide <agurpide>
# @Date:   19-10-2020
# @Email:  agurpidelash@irap.omp.eu
# @Last modified by:   agurpide
# @Last modified time: 22-03-2022

import os
from regions import Regions
from photutils.aperture import aperture_photometry
from astropy.io import fits
import argparse
from numpy import log10, sqrt
import astropy.units as u
from astropy import wcs
import logging
import hst_utils as hst_ut
import numpy as np
from extinction import ccm89, remove
import stsynphot as stsyn
from astropy.time import Time

parser = argparse.ArgumentParser(description='Extracts fluxes from the given apertures.')
parser.add_argument("images", help="Image files where to extract fluxes", nargs='+', type=str)
parser.add_argument("-s", "--source", type=str, help='Source extraction region file to use for the aperture photometry', nargs=1)
parser.add_argument("-b", "--background", type=str, help='Background extraction region file to use for the aperture photometry (optional)', nargs="?")
parser.add_argument("-e", "--exclude", type=str, help='File with extraction regions to exclude from the source aperture photometry', nargs="?")
parser.add_argument("--av", type=float, help='Extinction correction to obtain derredened fluxes', nargs="?")
parser.add_argument("-a", "--aperture_correction", type=float, help='Aperture correction (see https://stsci.edu/hst/instrumentation/wfc3/data-analysis/photometric-calibration/uvis-encircled-energy and https://stsci.edu/hst/instrumentation/acs/data-analysis/aperture-corrections)', nargs="?", default=1)
args = parser.parse_args()

source_reg = Regions.read(args.source[0], format="ds9")[0]

for image_file in args.images:

    if os.path.isfile(image_file):
        hst_hdul = fits.open(image_file)
        date = hst_hdul[0].header["DATE-OBS"]
        obsdate = Time(date).mjd
        hst_filter = hst_ut.get_image_filter(hst_hdul[0].header)
        instrument = hst_hdul[0].header["INSTRUME"]
        obs_mode = hst_hdul[1].header["PHOTMODE"]
        if "WFC3" in obs_mode:
            keywords = obs_mode.split(" ")
            obsmode = "%s,%s,%s,mjd#%.2f" % (keywords[0], keywords[1], keywords[2], obsdate)
        elif "WFC2" in obs_mode:
            keywords = obs_mode.split(",")
            #['WFPC2', '1', 'A2D7', 'F656N', '', 'CAL']
            obsmode = "%s,%s,%s,%s,%s" % (keywords[0], keywords[1], keywords[2],  keywords[3], keywords[5])
        elif "WFC1" in obs_mode:
            obsmode = obs_mode.replace(" ", ",")
        bp = stsyn.band(obsmode)
        rect_width = bp.rectwidth()
        print(rect_width)
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
        hst_wcs = wcs.WCS(hst_hdul[1].header)
        source_aperture = hst_ut.region_to_aperture(source_reg, hst_wcs)
        mask = image_data < 0
        phot_source = aperture_photometry(image_data, source_aperture, error=np.sqrt(image_data * exp_time) / exp_time, wcs=hst_wcs, mask=mask)
        source_area  = source_aperture.area
        aperture_keyword = "corrected_aperture_sum(%s)" % units

        if args.exclude is not None:
            for exclude_reg in Regions.read(args.exclude, format="ds9"):
                exclude_aperture = hst_ut.region_to_aperture(exclude_reg, hst_wcs)
                phot_exclude = aperture_photometry(image_data, exclude_aperture, wcs=hst_wcs, error=np.sqrt(image_data * exp_time) / exp_time, mask=mask)
                source_area  -= exclude_aperture.area
                phot_source["aperture_sum_err"] = np.sqrt(phot_exclude["aperture_sum_err"] ** 2 + phot_source["aperture_sum_err"] ** 2)
                phot_source["aperture_sum"] -= phot_exclude["aperture_sum"]

        # if a background region was given
        if args.background is not None:
            bkg_reg = Regions.read(args.background, format="ds9")[0]
            bkg_aperture = hst_ut.region_to_aperture(bkg_reg, hst_wcs)
            phot_bkg = aperture_photometry(image_data, bkg_aperture, wcs=hst_wcs, error=np.sqrt(image_data * exp_time) / exp_time, mask=mask)
            phot_source[aperture_keyword] = (phot_source["aperture_sum"] - phot_bkg["aperture_sum"] / bkg_aperture.area * source_area) / args.aperture_correction
            phot_source["corrected_aperture_err"] = sqrt(phot_source["aperture_sum_err"] ** 2 + (phot_bkg["aperture_sum_err"] / bkg_aperture.area * source_area) ** 2) / args.aperture_correction

        else:
            logging.warning("No background was given, no background correction will be performed.")
            phot_source[aperture_keyword] = phot_source["aperture_sum"] / args.aperture_correction
            phot_source["corrected_aperture_err"] =phot_source["aperture_sum_err"] / args.aperture_correction

        phot_source_conf_pos = phot_source[aperture_keyword] + phot_source["corrected_aperture_err"]
        phot_source_conf_neg = phot_source[aperture_keyword] - phot_source["corrected_aperture_err"]

        # divide by the exposure time if needed
        if "/S" not in units:
            print("Units: %s. Applying exposure time correction" % units)
            counts_at_infinity /= exp_time
            counts_at_infinity_conf /= exp_time
            counts_at_infinity_neg /= exp_time

        flux_header = "flux(%s)" % photflam.unit
        phot_source[flux_header] = phot_source[aperture_keyword] * photflam
        phot_source["flux_err"] = phot_source_conf_pos * photflam - phot_source[flux_header]
        # "integrated" over the filter filter_bandwidth
        flux_bp = "int_flux(%s)" % (photflam.unit * rect_width.unit)
        phot_source[flux_bp] = phot_source[aperture_keyword] * photflam * rect_width
        phot_source["int_flux_err"] = phot_source_conf_pos * photflam * rect_width - phot_source[flux_bp]
        # more appropiate for emission lines
        line_flux = "monochromatic_flux(%s)" % (photflam.unit)
        phot_source[line_flux] = phot_source[aperture_keyword] * bp.emflx(bp.area)
        phot_source["monochromatic_flux_err"] = phot_source_conf_pos * bp.emflx(bp.area) - phot_source[line_flux]
        #https://www.stsci.edu/hst/instrumentation/acs/data-analysis/zeropoints
        phot_source["mag"] = -2.5 * log10(phot_source[aperture_keyword]) - zero_point
        phot_source["mag_err_neg"] = -2.5 * log10(phot_source_conf_pos) - zero_point - phot_source["mag"]
        phot_source["mag_err_pos"] = -2.5 * log10(phot_source_conf_neg) - zero_point - phot_source["mag"]

        if args.av is not None:
            waves = np.array([pivot_wavelength])
            phot_source["der_flux"] = remove(ccm89(waves, args.av, 3.1, unit="aa"), phot_source[flux_header])
            phot_source["der_flux_err"] = remove(ccm89(waves, args.av, 3.1, unit="aa"), phot_source_conf_pos* photflam ) - phot_source["der_flux"]

        # formatting
        phot_source["xcenter"].info.format = '%.2f'
        phot_source["ycenter"].info.format = '%.2f'
        phot_source["aperture_sum"].info.format = '%.3f'
        phot_source[aperture_keyword].info.format = '%.2f'
        phot_source["corrected_aperture_err"].info.format = '%.2f'
        phot_source[flux_header].info.format = '%.3E'
        phot_source['flux_err'].info.format = '%.2E'
        phot_source["mag"].info.format = "%.2f"
        phot_source["mag_err_neg"].info.format = "%.2f"
        phot_source["mag_err_pos"].info.format = "%.3f"
        reg_basename = os.path.basename(args.source[0]).replace('.reg', '')
        out_data_file = "aperture_phot_%s_%s.csv" % (hst_filter, reg_basename)
        phot_source.write(out_data_file, overwrite=True)
        out_info_file = image_file.replace(".fits", "%s_apt_info.txt" % reg_basename)
        f = open(out_info_file, "w+")
        f.write("Date:%s\nInstrument:%s\nFilter:%s\nDetector:%s\nExposure(s):%.2f\nPivot wavelength (A):%.1f\nRMS:%.1f\nPHOTFLAM:%s\nAperture correction:%.4f" % (date, instrument, hst_filter, detector, exp_time, pivot_wavelength, filter_bandwidth, photflam.value, args.aperture_correction))
        aperture_reg = type(source_aperture).__name__
        attributes = vars(source_aperture)
        att_list = ''.join("%s: %s" % item for item in attributes.items())
        f.write("\n###Aperture details###\n%s\n%s" % (aperture_reg, att_list))
        f.close()
        xspec_outfile = image_file.replace(".fits", "_to_xspec.ascii")
        f = open("%s" % ("%s" % xspec_outfile), "w+")
        f.write("%.2f %.2f %.3e %.3e" % (pivot_wavelength - filter_bandwidth / 2, pivot_wavelength + filter_bandwidth / 2, phot_source[flux_header].value, phot_source['flux_err'].value))
        f.close()
        print("Use 'ftflx2xsp infile=%s xunit=angstrom yunit=ergs/cm^2/s/A nspec=1 phafile=hst_%s.fits rspfile=hst_%s.rsp' to convert to XSPEC format" % (xspec_outfile, hst_filter, hst_filter))
        print("Warning: you may need to change the '-' sign in the ascii file for this to work")
        print("Output stored to %s and %s" % (out_info_file, out_data_file))
        #
        # print some useful info

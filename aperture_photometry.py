# @Author: Andrés Gúrpide <agurpide>
# @Date:   19-10-2020
# @Email:  agurpidelash@irap.omp.eu
# @Last modified by:   agurpide
# @Last modified time: 20-10-2020

import os
from regions import read_ds9
from photutils.aperture import aperture_photometry, CircularAperture, CircularAnnulus
from astropy.io import fits
import argparse
from numpy import log10, sqrt


def region_to_aperture(region):
    """Convert region object to aperture object."""
    source_center = (region.center.x, region.center.y)
    region_type = type(region).__name__
    if region_type == 'CirclePixelRegion':
        return CircularAperture(source_center, r=region.radius)
    elif region_type == "CircleAnnulusPixelRegion":
        return CircularAnnulus(source_center, r_in=region.inner_radius, r_out=region.outer_radius)


parser = argparse.ArgumentParser(description='Extracts fluxes from the given apertures.')
parser.add_argument("images", help="Image files where to look for sources", nargs='+', type=str)
parser.add_argument("-r", "--regions", type=str, help='Source (first) and background (second) extraction region file to use for the aperture photometry', nargs=1)
parser.add_argument("-a", "--aperture_correction", type=float, help='Aperture correction (see https://stsci.edu/hst/instrumentation/wfc3/data-analysis/photometric-calibration/uvis-encircled-energy)', nargs="?", default=1)
args = parser.parse_args()
regions = read_ds9(args.regions[0])
source_reg = regions[0]
bkg_reg = regions[1]
source_aperture = region_to_aperture(source_reg)
bkg_aperture = region_to_aperture(bkg_reg)
for image_file in args.images:

    if os.path.isfile(image_file):
        hst_hdul = fits.open(image_file)
        date = hst_hdul[0].header["DATE-OBS"]
        if "FILTER" in hst_hdul[0].header:
            filter = hst_hdul[0].header["FILTER"]
        else:
            filter = hst_hdul[0].header["FILTNAM1"]
        instrument = hst_hdul[0].header["INSTRUME"]
        units = hst_hdul[1].header["BUNIT"]
        exp_time = float(hst_hdul[0].header["EXPTIME"])
        detector = hst_hdul[0].header["DETECTOR"]
        pivot_wavelength = hst_hdul[0].header["PHOTPLAM"]
        # if UV filter then https://www.stsci.edu/files/live/sites/www/files/home/hst/instrumentation/wfc3/documentation/instrument-science-reports-isrs/_documents/2017/WFC3-2017-14.pdf
        # use phftlam1 keyword for UV filters
        if detector == "UVIS" and filter == "F225W" or "F275W" or "F336W":
            photflam = float(hst_hdul[0].header["PHTFLAM1"])
        else:
            photflam = float(hst_hdul[0].header["PHOTFLAM"])

        zero_point = float(hst_hdul[1].header["PHOTZPT"])
        image_data = hst_hdul[1].data
        phot_source = aperture_photometry(image_data, source_aperture)
        phot_bkg = aperture_photometry(image_data, bkg_aperture)
        # background correction
        phot_source["corrected_aperture"] = phot_source["aperture_sum"] - phot_bkg["aperture_sum"] / bkg_aperture.area * source_aperture.area
        # divide by the exposure time if needed
        phot_source["corrected_aperture_err"] = sqrt(phot_source["aperture_sum"] - (sqrt(phot_bkg["aperture_sum"]) / bkg_aperture.area * source_aperture.area) ** 2)
        if "/S" in units:
            print("Units: %s. Exposure time correction will not be applied" % units)
            phot_source["flux"] = phot_source["corrected_aperture"] / args.aperture_correction * photflam
            phot_source["flux_err"] = phot_source["corrected_aperture_err"] / args.aperture_correction * photflam
            phot_source["mag"] = -2.5 * log10(phot_source["corrected_aperture"] / args.aperture_correction) + zero_point
            phot_source["mag_err"] = -2.5 * log10(phot_source["corrected_aperture_err"] / args.aperture_correction) + zero_point
        else:
            print("Units: %s. Applying exposure time correction" % units)
            phot_source["flux"] = phot_source["corrected_aperture"] / args.aperture_correction * photflam / exp_time
            phot_source["flux_err"] = phot_source["corrected_aperture_err"] / args.aperture_correction * photflam / exp_time
            phot_source["mag"] = -2.5 * log10(phot_source["corrected_aperture"] / args.aperture_correction / exp_time) + zero_point
            phot_source["mag_err"] = -2.5 * log10(phot_source["corrected_aperture_err"] / args.aperture_correction / exp_time) + zero_point

        phot_source['flux'].info.format = '%.2E'

        phot_source.write("aperture_photometry.csv", overwrite=True)
        f = open("%s" % (image_file.replace(".fits", "log")), "w+")
        f.write("%s\n%s\n%s\n%.2f\t%s\n" % (date, filter, detector, exp_time, pivot_wavelength))
        f.close()
        # print some useful info

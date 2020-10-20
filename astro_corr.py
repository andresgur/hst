# @Author: Andrés Gúrpide <agurpide>
# @Date:   18-05-2020
# @Email:  agurpidelash@irap.omp.eu
# @Last modified by:   agurpide
# @Last modified time: 20-10-2020

# Script to adjust the coordinates of a cube from an input image( Preferabley HST)
# Created the 10/06/2019 by Andres Gurpide
# imports
import astropy.units as u
import glob

import os

from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
from astroquery.mast import Observations
import numpy as np
from ccdproc import ImageFileCollection

from drizzlepac import tweakreg
import argparse
import logging

# read arguments
ap = argparse.ArgumentParser(description='Adjust coordinates of HST images using the Gaia catalog sources from target or images')
ap.add_argument("-t", "--target", nargs='?', help="Target for HST images downloading or process images in the folder by default", type=str)
ap.add_argument("-r", "--radius", nargs='?', type=float, help="Radius search in arcseconds", default=250)
ap.add_argument("-m", "--mask", nargs='?', type=str, help="Config file with mask and input files", default="")
ap.add_argument("--update", help="Update header file with the new solution. Default false", action='store_true')

# parse args
args = ap.parse_args()
nsigma = 3
proper_motion_threshold = 20  # mas/yr
match_radius = 1.8  # arcsec

# logger
scriptname = os.path.basename(__file__)

logger = logging.getLogger(scriptname)
logger.setLevel(logging.DEBUG)
out_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')


target = args.target
search_radius = args.radius * u.arcsec

logger.info("Mask config file %s" % args.mask)


if target is not None:
    logger.info("Retrieving HST images for target %s" % target)
    # download observation
    obsTable = Observations.query_criteria(obs_collection=['HLA'], objectname=target,
                                           filters=['F656N', 'F814W', 'F606W', 'F606W', 'F555W'],
                                           dataRights='PUBLIC')
    products = Observations.get_product_list(obsTable)
    filtered = Observations.filter_products(products,
                                            mrp_only=False,
                                            productSubGroupDescription='DRZ')
    Observations.download_products(filtered, mrp_only=False)
    # move products into the current directory
    for flc in glob.glob('./mastDownload/HST/*/*drz.fits'):
        flc_name = os.path.split(flc)[-1]
        os.rename(flc, flc_name)
else:
    logger.info("Processing images in current folder")

collec = ImageFileCollection('./', glob_include="*drz.fits", ext='SCI',
                             keywords=["targname", "CRVAL1", "CRVAL2", "filter", "exptime", "postarg1", "postarg2"])

table = collec.summary
# get image center
ra_targ = table['CRVAL1'][0]
dec_targ = table['CRVAL2'][0]
logger.info("Looking Gaia sources around (%.3f, %.3f) with radius %.3f with less than %.1f mas/yr" % (ra_targ, dec_targ, search_radius.value, proper_motion_threshold))
coord = SkyCoord(ra=ra_targ, dec=dec_targ, unit=(u.deg, u.deg))
# query Gaia sources and write the result
ref_cat = 'gaia_hst.csv'
gaia_query = Gaia.query_object_async(coordinate=coord, radius=search_radius)

reduced_query = gaia_query['ra', 'dec', 'ra_error', 'dec_error', 'phot_g_mean_mag', 'ref_epoch', 'pmra', 'pmdec', 'pmra_error', 'pmdec_error', 'solution_id']

# np.abs was giving problems here so just filter twice with 10 and -10 proper motions are in mas
filter = ((np.abs(reduced_query['pmdec']) < proper_motion_threshold) & ((np.abs(reduced_query['pmra']) < proper_motion_threshold)) | (reduced_query['pmra'].mask == True))
reduced_query = reduced_query[filter]
reduced_query.write(ref_cat, format='ascii.commented_header', delimiter='\t', overwrite=True)

wcsname = 'gaia'

cw = 3.5   # The convolution kernel width in pixels. Recommended values (~2x the PSF FWHM): ACS/WFC & WFC3/UVIS ~3.5 pix and WFC3/IR ~2.5 pix.
# ACS/WFC 0.05 arcsec/pixel
# resolution 3.5 * 0.05 = 0.175 arcsec

tweakreg.TweakReg('*drz.fits',  # Pass input images
                  updatehdr=args.update,  # update header with new WCS solution
                  imagefindcfg={'threshold': 30., 'cw': cw},  # Detection parameters, threshold varies for different data
                  refcat=ref_cat,  # Use user supplied catalog (Gaia)
                  interactive=False,
                  use_sharp_round=True,
                  sharphi=1,
                  roundhi=0.9,
                  roundlo=-0.9,
                  sharplo=0.1,
                  minflux=0,
                  maxflux=1000000,
                  fluxunits='counts',
                  see2dplot=True,
                  verbose=False,
                  shiftfile=True,  # Save out shift file (so we can look at shifts later)
                  outshifts='gaia_shifts.txt',  # name of the shift file
                  wcsname=wcsname,  # Give our WCS a new name
                  reusename=True,
                  sigma=3,
                  nclip=4,
                  searchrad=match_radius,
                  searchunits='arcseconds',
                  minobj=5,
                  fitgeometry='general',
                  exclusions=args.mask)  # Use the 6 parameter fit

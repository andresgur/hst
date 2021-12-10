# @Author: Andrés Gúrpide <agurpide>
# @Date:   18-05-2020
# @Email:  agurpidelash@irap.omp.eu
# @Last modified by:   agurpide
# @Last modified time: 07-12-2021

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
ap = argparse.ArgumentParser(description='Adjust coordinates of HST images using the Gaia catalog sources or a reference image. By default uses all .drz images from the current folder.')
ap.add_argument("-s", "--search_radius", nargs='?', type=float, help="Search radius in arcseconds for the catalog search", default=250)
ap.add_argument("-m", "--mask", nargs='?', type=str, help="Config file with mask and input files (file\tmask(in image unit with exclude and include regions))", default="")
ap.add_argument("-r", "--match_radius", nargs='?', type=float, help="Match radius for the matching of HST sources to catalog sources in arcsec", default=1)
ap.add_argument("--update", help="Update header file with the new solution. Default false", action='store_true')
ap.add_argument("--gaia", help="Uses the gaia catalogue as reference for the astrometric alignment, otherwise uses bigger image. Default false", action='store_true')
# parse args
args = ap.parse_args()
nsigma = 3
proper_motion_threshold = 20  # mas/yr
match_radius = args.match_radius  # arcsec (adjust manually)

# logger
scriptname = os.path.basename(__file__)

logger = logging.getLogger(scriptname)
logger.setLevel(logging.DEBUG)
out_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

search_radius = args.search_radius * u.arcsec

logger.info("Mask config file %s" % args.mask)

logger.info("Processing images in current folder")

collec = ImageFileCollection('./', glob_include="*drz.fits", ext='SCI',
                             keywords=["targname", "CRVAL1", "CRVAL2", "exptime", "naxis1", "naxis2", "wcsname"])

# put largest image first
collec.sort(("naxis1", "naxis2"))
table = collec.summary

if args.gaia:
    # get image center
    ra_targ = table['CRVAL1'][0]
    print(table)
    dec_targ = table['CRVAL2'][0]
    logger.info("Looking Gaia sources around (%.3f, %.3f) with radius %.3f with less than %.1f mas/yr" % (ra_targ, dec_targ, search_radius.value, proper_motion_threshold))
    # improve this using a box like the image https://astroquery.readthedocs.io/en/latest/gaia/gaia.html (width and height)
    coord = SkyCoord(ra=ra_targ, dec=dec_targ, unit=(u.deg, u.deg))
    # query Gaia sources and write the result
    ref_cat = 'gaia_hst.csv'

    max_rows = 3000

    Gaia.ROW_LIMIT = max_rows
    gaia_query = Gaia.query_object_async(coordinate=coord, radius=search_radius)
    #https://gea.esac.esa.int/archive/documentation/GDR2/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html
    reduced_query = gaia_query['ra', 'dec', 'ra_error', 'dec_error', 'phot_g_mean_flux', 'ref_epoch', 'pmra', 'pmdec', 'pmra_error', 'pmdec_error', 'solution_id']

    ngaia = len(reduced_query)

    # np.abs was giving problems here so just filter twice with 10 and -10 proper motions are in mas
    filter = ((np.abs(reduced_query['pmdec']) < proper_motion_threshold) & ((np.abs(reduced_query['pmra']) < proper_motion_threshold)))
    reduced_query = reduced_query[filter]
    reduced_query.write(ref_cat, format='ascii.commented_header', delimiter='\t', overwrite=True)
    print('Found %d Gaia sources' % ngaia)
    wcsname = 'gaia'
    nbright_sources = ngaia * 2
    refimage = ""
    ref_bright_sources = None
else:
    wcsname = "hst"
    nbright_sources = None
    ref_bright_sources = None
    ref_cat = ""
    refimage = table["file"][0]
    wcsname = table["wcsname"][0]


cw = 3.5   # The convolution kernel width in pixels. Recommended values (~2x the PSF FWHM): ACS/WFC & WFC3/UVIS ~3.5 pix and WFC3/IR ~2.5 pix.
# ACS/WFC 0.05 arcsec/pixel
# resolution 3.5 * 0.05 = 0.175 arcsec
tweakreg.TweakReg('*drz.fits',  # Pass input images
                  updatehdr=args.update,  # update header with new WCS solution
                  imagefindcfg={'threshold': 75, 'cw': cw, "nbright": nbright_sources},  # Detection parameters, threshold varies for different data # The object detection threshold above the local background in units of sigma.
                  refimagefindcfg={'threshold': 75, 'cw': cw, "refnbright": ref_bright_sources},  # Detection parameters, threshold varies for different data # The object detection threshold above the local background in units of sigma.
                  refcat=ref_cat,  # Use user supplied catalog (Gaia)
                  refimage=refimage,
                  rfluxcol=5,
                  rfluxunits="flux",
                  rminflux=0,
                  interactive=False,
                  use_sharp_round=True,
                  expand_refcat=True,
                  sharphi=1,
                  roundhi=1,
                  roundlo=-1,
                  sharplo=0.2,
                  see2dplot=True,
                  verbose=False,
                  shiftfile=True,  # Save out shift file (so we can look at shifts later)
                  outshifts='%s_shifts.txt' % wcsname,  # name of the shift file
                  wcsname=wcsname,  # Give our WCS a new name
                  reusename=True,
                  sigma=3, # sigma clipping parameters
                  nclip=3,
                  searchrad=match_radius,
                  searchunits='arcseconds',
                  separation=0,
                  tolerance=1,
                  minobj=5,
                  fitgeometry='general',
                  exclusions=args.mask) # Use the 6 parameter fit
#nclip=4,
# wirter parameters used
f = open(os.path.basename(__file__).replace('.py', '.log'), 'w+')
f.write('#search_radius\tmatch_radius\n')
f.write('%.2f\t%.2f' % (args.search_radius, match_radius))
f.close()

print("Run '%s/ds9 -scale mode 99.5 -cmap heat *drz.fits -frame lock wcs -catalog import csv gaia_hst.csv -catalog symbol color green -catalog symbol shape circle -catalog symbol size %.2f -catalog symbol units arcsecs -catalog import csv hst_9796_01_acs_wfc_f555w_drz_sky_catalog.coo -catalog symbol shape point -catalog symbol color white"
     %(os.environ['HOME'], match_radius))

# examine the Results
# ds9 hst_13364_98_wfc3_uvis_total_drz.fits -catalog import csv gaia_hst.csv -catalog symbol shape cross point -catalog import csv hst_13364_98_wfc3_uvis_total_drz_sky_catalog.coo -catalog symbol shape cross point -catalog symbol color blue -catalog import csv hst_13364_98_wfc3_uvis_total_drz_xy_catalog.match -catalog symbol color green -catalog symbol shape circle

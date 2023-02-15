#!/usr/bin/env python
# coding: utf-8
# @Author: Andrés Gúrpide <agurpide>
# @Date:   14-04-2023
# @Email:  agurpidelash@a.gurpide-lasheras@soton.ac.uk
# @Last modified by:   agurpide
# @Last modified time: 14-04-2023

import argparse
from astroquery.mast import Observations
from astropy.time import Time
from coordinates import convert_ra_dec

ap = argparse.ArgumentParser(description='Script to fetch images from the mast portal (https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html)')
ap.add_argument("--ra", help="Right ascension. In degree or hh:mm:ss format", type=str, nargs="?", required=True)
ap.add_argument("--dec", help="Declination. In degree or dd:mm:ss format", type=str, nargs="?", required=True)
ap.add_argument("-o", "--outdir", nargs='?', help="Output dir name", type=str, default="data")
args = ap.parse_args()

ra = args.ra
dec = args.dec

ra, dec = convert_ra_dec(ra, dec)

ra_dec = "%.12f %+.12f" % (ra, dec)
print("(RA, Dec) = (%s)" % (ra_dec))

# this searches reduced and calibrated HST observations from 2000-2030 with a minimum exposure time of 500s
#and avoiding the "DETECTION" ones
# it also avoids duplicating WFC and PC cameras
# Restricted to W filters
filters_keywords = "F*W"
print("Filters to be retreived: %s" % filters_keywords)
obs_table = Observations.query_criteria(coordinates="%.5f, %+.5f" % (ra, dec), radius="0:0:30 degrees",
                                        obs_collection=["HLA"], t_exptime=[500, 50000], filters="F*W",
                                        t_min=[51544, 62502], instrument_name=["WFPC2/WFC", "ACS/WFC", "WFC3/UVIS", "ACS/HRC", "WFC3/IR", "NICMOS/NIC2", "NICMOS/NIC1"])

print("Found %d observations" % len(obs_table))

print(obs_table)

products = Observations.get_product_list(obs_table)

manifest = Observations.download_products(products, productType="SCIENCE", download_dir="")

latex = "Obs id & Date & Instrument & Filter & Exp\\\n & & & & ks \\ \n"

for obs in np.sort(obs_table, order="t_min"):
    mjds = Time(obs["t_min"], format='mjd')
    mjds.format = "iso"
    mjds.out_subfmt = "date"

    latex += "%s & %s & %s & %s & %.2f\\\\ \n" % (obs["obs_id"], mjds, obs["instrument_name"], obs["filters"], obs["t_exptime"] / 1000)
outfile = "latex_out.tex"

latex_out = open(outfile, "w+")
latex_out.write(outfile)

print("Table: %s \n stored to %s" % (latex, outfile))

# !/usr/bin/env python3.7
# Script to convert from counts to flux for HST images
# Author: Andres Gurpide Lasheras  andres.gurpide@gmail.com / PhD at the Institute de Recherche en Astrophysique et Planetologie
# 10-09-2019
# !/usr/bin/env python3
# coding=utf-8
# imports
import argparse
from math import log10
from astropy.io import fits
# read arguments
ap = argparse.ArgumentParser(description='Input hardness file to be processed')
ap.add_argument('input_fits', nargs=1, help="Input HST image")
ap.add_argument('-c', dest='counts', nargs=1, help="Counts you wish to convert to flux", type=float)
ap.add_argument('-b', dest='bkgcounts', nargs='?', help="Background counts", default=0, type=float)
args = ap.parse_args()

input_image = args.input_fits[0]
source_counts = args.counts[0]
background_counts = args.bkgcounts

hst_hdul = fits.open(input_image)
date = hst_hdul[0].header["DATE-OBS"]
if "FILTER" in hst_hdul[0].header:
    filter = hst_hdul[0].header["FILTER"]
else:
    filter = hst_hdul[0].header["FILTNAM1"]
exp_time = float(hst_hdul[0].header["EXPTIME"])
photflam = float(hst_hdul[1].header["PHOTFLAM"])
# http://www.stsci.edu/instruments/wfpc2/Wfpc2_dhb/intro_ch34.html
flux = (source_counts) * photflam
print("Corresponding flux in erg cm-2 s-1 Å-1:\n")
print(flux)
zero_point = float(hst_hdul[1].header["PHOTZPT"])
print("Zero point mag")
print(zero_point)
mag = -2.5 * log10(flux) + zero_point
print("Corresponding mag in the %s filter:\n" % filter)
print(mag)

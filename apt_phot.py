# @Author: Andrés Gúrpide <agurpide>
# @Date:   22-09-2021
# @Email:  agurpidelash@irap.omp.eu
# @Last modified by:   agurpide
# @Last modified time: 05-11-2021



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from regions import *
from photutils.aperture import aperture_photometry, CircularAperture, CircularAnnulus, EllipticalAperture, SkyCircularAperture, SkyCircularAnnulus, SkyEllipticalAperture
from astropy.io import fits
import argparse
from numpy import log10, sqrt
import pyautogui,time
import numpy as np
import pandas as pd
from astropy.wcs import WCS

from synphot import SourceSpectrum
import stsynphot as stsyn
from synphot import Observation

#you might want to put this in your .bashrc
os.environ['PYSYN_CDBS']='/home/mparra/PARRA/Observ/CCF/HST/synphot/grp/hst/cdbs'

'''
synphot needs CALSPEC files such as the Vega spectrum. You can download CALSPEC files using this tutorial :
https://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/astronomical-catalogs/calspec
'''

def calculate_values(detector, filt, mjd, aper):
    # parameters can be removed from obsmode as needed
    obsmode = 'wfc3,{},{},mjd#{},aper#{}'.format(detector, filt, mjd, aper)
    bp = stsyn.band(obsmode)

    # STMag
    photflam = bp.unit_response(stsyn.conf.area)  # inverse sensitivity in flam
    stmag = -21.1 -2.5 * log10(photflam.value)

    # Pivot Wavelength and bandwidth
    photplam = bp.pivot() # pivot wavelength in angstroms
    bandwidth = bp.photbw() # bandwidth in angstroms

    # ABMag
    abmag = stmag - 5 * log10(photplam.value) + 18.6921

    # Vegamag
    #for some reason stsyn.Vega doesn't load so we redefine it
    stsyn.Vega=SourceSpectrum.from_vega()
    obs = Observation(stsyn.Vega, bp, binset=bp.binset) # synthetic observation of vega in bandpass using vega spectrum
    vegamag = -obs.effstim(flux_unit='obmag', area=stsyn.conf.area)

    return obsmode, photplam.value, bandwidth.value, photflam.value, stmag, abmag, vegamag.value

def allreg_to_aperture(region):
    """Convert region object to aperture object."""

    region_type = type(region).__name__
    if "Pixel" in region_type:
        source_center = (region.center.x, region.center.y)
        if region_type == 'CirclePixelRegion':
            return CircularAperture(source_center, r=region.radius)
        elif region_type == "CircleAnnulusPixelRegion":
            return CircularAnnulus(source_center, r_in=region.inner_radius, r_out=region.outer_radius)
        elif region_type == "EllipsePixelRegion":
            # to be tested
            return EllipticalAperture(source_center, a=region.width, b=region.height, angle=region.angle)
    elif "Sky" in region_type:
        center = region.center.fk5
        if region_type == "CircleSkyRegion":
            return SkyCircularAperture(center, r=region.radius)
        elif region_type == "EllipseSkyRegion":
            return SkyEllipticalAperture(center, a=region.width / 2, b=region.height / 2, angle=region.angle)
        elif region_type == "CircleAnnulusSkyRegion":
            return SkyCircularAnnulus(center, r_in=region.inner_radius, r_out=region.outer_radius)
    else:
        print("Error region not implemented")
        return None

# def regions_to_aperture(region_src,region_bg,file):
#     """Convert region object to aperture object."""
#     reg_src=read_ds9(region_src)[0]
#     reg_bg=read_ds9(region_bg)[0]

#     region_src_type = type(region_src).__name__
#     region_bg_type = type(region_bg).__name__

#     #if the region is in sky coordinates, we convert it in physical and
#     #re-read it afterwards
#     if region_src_type == 'CircleSkyRegion' or 'CircleAnnulusSkyRegion':
#         reg_src_name='temp_phot_'+file[:file.find('.')]+'_src.reg'
#         command_src='ds9 '+file+' -region '+region_src+' \
#                 -region system physical \
#                 -regions select all \
#                 -regions save '+reg_src_name+' \
#                 -exit &'

#         os.system(command_src)

#         #since my ds9 takes forever to load and load faster when I alt-tab
#         #I loop an alt tab until the file exists (which means it's done)
#         while not os.path.isfile(reg_src_name):
#             pyautogui.keyDown('alt')
#             time.sleep(.2)
#             pyautogui.press('tab')
#             time.sleep(.2)
#             pyautogui.keyUp('alt')

#         reg_src=read_ds9(reg_src_name)[0]
#         region_src_type = type(reg_src).__name__

#     source_src_center = (reg_src.center.x, reg_src.center.y)

#     if region_src_type == 'CirclePixelRegion':
#         src_out=CircularAperture(source_src_center, r=reg_src.radius)
#     elif region_src_type == "CircleAnnulusPixelRegion":
#         src_out=CircularAnnulus(source_src_center, r_in=reg_src.inner_radius, r_out=reg_src.outer_radius)

#     #two loops with different variable names bc. it caused some conflicts
#     if region_bg_type == 'CircleSkyRegion' or 'CircleAnnulusSkyRegion':
#         reg_bg_name='temp_phot_'+file[:file.find('.')]+'_bg.reg'
#         command_bg='ds9 '+file+' -region '+region_bg+' \
#                 -region system physical \
#                 -regions select all \
#                 -regions save '+reg_bg_name+' \
#                 -exit &'

#         os.system(command_bg)

#         #since my ds9 takes forever to load AND loads faster when I alt-tab
#         #I loop an alt tab until the file gets created
#         while not os.path.isfile(reg_bg_name):
#             pyautogui.keyDown('alt')
#             time.sleep(.2)
#             pyautogui.press('tab')
#             time.sleep(.2)
#             pyautogui.keyUp('alt')

#         reg_bg=read_ds9(reg_bg_name)[0]
#         region_bg_type = type(reg_bg).__name__

#     source_bg_center = (reg_bg.center.x, reg_bg.center.y)

#     if region_bg_type == 'CirclePixelRegion':
#         bg_out=CircularAperture(source_bg_center, r=reg_bg.radius)
#     elif region_bg_type == "CircleAnnulusPixelRegion":
#         bg_out=CircularAnnulus(source_bg_center, r_in=reg_bg.inner_radius, r_out=reg_bg.outer_radius)

#     if os.path.exists(reg_src_name):
#         os.remove(reg_src_name)

#     if os.path.exists(reg_bg_name):
#         os.remove(reg_bg_name)

#     return [src_out,bg_out]

def region_to_aperture(region, wcs=None):
    """Convert region object to photutils.aperture.aperture_photometry object. The wcs object is needed only if the input regions are in sky coordinates.
    Parameters
    ----------
    region: regions.Region
        Output of read_ds9 method or str
    wcs: astropy.wcs.WCS
        A world coordinate system if the region in sky coordinates IS needed to convert it to pixels.
    """

    if type(region)==str:
        region=read_ds9(region)[0]
    print(region)
    region_type = type(region).__name__
    if "Pixel" in region_type:
        source_center = (region.center.x, region.center.y)
        if region_type == 'CirclePixelRegion':
            return CircularAperture(source_center, r=region.radius)
        elif region_type == "CircleAnnulusPixelRegion":
            return CircularAnnulus(source_center, r_in=region.inner_radius, r_out=region.outer_radius)
        elif region_type == "EllipsePixelRegion":
            # to be tested
            return EllipticalAperture(source_center, a=region.width, b=region.height, theta=region.angle)
    elif "Sky" in region_type:
        if wcs is None:
            print("Error, cannot obtain aperture without a wcs.")
            return None
        center = region.center.fk5
        if region_type == "CircleSkyRegion":
            return SkyCircularAperture(center, r=region.radius).to_pixel(wcs)
        elif region_type == "CircleAnnulusSkyRegion":
            print("Region %s not implemented")
        elif region_type == "EllipseSkyRegion":
            return SkyEllipticalAperture(center, a=region.width, b=region.height, theta=region.angle).to_pixel(wcs)
        elif region_type == "CircleAnnulusSkyRegion":
            return SkyCircularAnnulus(center, r_in=region.inner_radius, r_out=region.outer_radius).to_pixel(wcs)
    else:
        print("Error region not implemented")
        return None

def apt_phot(images,source_reg,background_reg=None,apt_corr=1):

    #if the image is unique we convert it to a list to keep the next loop
    #unchanged

    if type(images)==str:
        images=[images]

    for image_file in images:
        if os.path.isfile(image_file):

            hst_hdul = fits.open(image_file)
            date = hst_hdul[0].header["DATE-OBS"]
            if "FILTER" in hst_hdul[0].header:
                hst_filter = hst_hdul[0].header["FILTER"]
            elif "FILTER1" in hst_hdul[0].header:
                hst_filter = hst_hdul[0].header["FILTER1"]
            else:
                hst_filter = hst_hdul[0].header["FILTNAM1"]
            instrument = hst_hdul[0].header["INSTRUME"]


            if "BUNIT" in hst_hdul[0].header:
                if hst_hdul[0].header["BUNIT"]=='':
                    print('no unit-assuming ELECTRONS/S')
                    units = 'ELECTRONS/S'
                else :
                    units=hst_hdul[0].header["BUNIT"]
            elif "BUNIT" in hst_hdul[1].header:
                if hst_hdul[1].header["BUNIT"]=='':
                    print('no unit-assuming ELECTRONS/S')
                    units = 'ELECTRONS/S'
                else :
                    units=hst_hdul[1].header["BUNIT"]

            exp_time = float(hst_hdul[0].header["EXPTIME"])
            detector = hst_hdul[0].header["DETECTOR"] if "DETECTOR" in hst_hdul[0].header else ""
            if "PHOTPLAM" in hst_hdul[0].header :
                pivot_wavelength = float(hst_hdul[0].header["PHOTPLAM"])
            elif "PHOTPLAM" in hst_hdul[1].header :
                pivot_wavelength = float(hst_hdul[1].header["PHOTPLAM"])
            if "PHOTBW" in hst_hdul[0].header :
                filter_bandwidth = float(hst_hdul[0].header["PHOTBW"])
            elif "PHOTBW" in hst_hdul[1].header :
                filter_bandwidth = float(hst_hdul[1].header["PHOTBW"])
            # if UV filter then https://www.stsci.edu/files/live/sites/www/files/home/hst/instrumentation/wfc3/documentation/instrument-science-reports-isrs/_documents/2017/WFC3-2017-14.pdf
            # use phftlam1 keyword for UV filters
            uv_filters = ["F200LP", "F300X", "F218W", "F225W", "F275W", "FQ232N", "FQ243N", "F280N"]
            if detector == "UVIS" and filter in uv_filters:
                photflam = float(hst_hdul[0].header["PHTFLAM1"])
            elif "PHOTFLAM" in hst_hdul[0].header:
                photflam = float(hst_hdul[0].header["PHOTFLAM"])
            elif "PHOTFLAM" in hst_hdul[1].header:
                photflam = float(hst_hdul[1].header["PHOTFLAM"])
            print("PHOTFLAM keyword value: %.2E" % photflam)

            if "PHOTZPT" in hst_hdul[0].header :
                zero_point = float(hst_hdul[0].header["PHOTZPT"])
            elif "PHOTZPT" in hst_hdul[1].header :
                zero_point = float(hst_hdul[1].header["PHOTZPT"])

            if len(hst_hdul)==1:
                image_data=hst_hdul[0].data
                wcs = WCS(hst_hdul[0].header)
            else:
                image_data = hst_hdul[1].data
                wcs = WCS(hst_hdul[1].header)

            source_aperture=region_to_aperture(source_reg,wcs)
            print(source_aperture)
            bkg_aperture=region_to_aperture(background_reg,wcs)

            phot_source = aperture_photometry(image_data, source_aperture)

            if background_reg is not None:
                phot_bkg = aperture_photometry(image_data, bkg_aperture)
                # background correction
                phot_source["corrected_aperture"] = phot_source["aperture_sum"] - phot_bkg["aperture_sum"] / bkg_aperture.area * source_aperture.area
                phot_source["corrected_aperture_err"] = sqrt(phot_source["aperture_sum"] + (sqrt(phot_bkg["aperture_sum"]) / bkg_aperture.area * source_aperture.area) ** 2)
                phot_source_conf = phot_source["corrected_aperture"] + phot_source["corrected_aperture_err"]
                phot_source_conf_neg = phot_source["corrected_aperture"] - phot_source["corrected_aperture_err"]
            else:
                phot_source["corrected_aperture"] = phot_source["aperture_sum"]
                phot_source["corrected_aperture_err"] = 0
                phot_source_conf = phot_source["corrected_aperture"] + phot_source["corrected_aperture_err"]
                phot_source_conf_neg = phot_source["corrected_aperture"] - phot_source["corrected_aperture_err"]
            # divide by the exposure time if needed
            if "/S" in units:
                print("Units: %s. Exposure time correction will not be applied" % units)
                phot_source["flux"] = phot_source["corrected_aperture"] / apt_corr * photflam
                phot_source["flux_err"] = phot_source_conf / apt_corr * photflam - phot_source["flux"]

                if phot_source["corrected_aperture"]>0:
                    phot_source["mag"] = -2.5 * log10(phot_source["corrected_aperture"] / apt_corr) - zero_point
                    phot_source["mag_err_neg"] = -2.5 * log10(phot_source_conf / apt_corr) - zero_point - phot_source["mag"]
                else :
                    phot_source["mag"]='Nan.'
                    phot_source["mag_err_neg"]='Nan.'

                if phot_source_conf_neg>0:
                    phot_source["mag_err_pos"] = -2.5 * log10(phot_source_conf_neg / apt_corr) - zero_point - phot_source["mag"]
                else :
                    phot_source["mag_err_pos"]='Nan.'

            else:
                print("Units: %s. Applying exposure time correction" % units)
                phot_source["flux"] = phot_source["corrected_aperture"] / apt_corr * photflam / exp_time
                phot_source["flux_err"] = phot_source_conf / apt_corr * photflam / exp_time - phot_source["flux"]

                if phot_source["corrected_aperture"]>0:
                    phot_source["mag"] = -2.5 * log10(phot_source["corrected_aperture"] / apt_corr / exp_time) - zero_point
                    phot_source["mag_err_neg"] = -2.5 * log10(phot_source_conf / apt_corr / exp_time) - zero_point - phot_source["mag"]
                else :
                    phot_source["mag"]='Nan.'
                    phot_source["mag_err_neg"]='Nan.'

                if phot_source_conf_neg>0:
                    phot_source["mag_err_pos"] = -2.5 * log10(phot_source_conf_neg / apt_corr / exp_time) - zero_point - phot_source["mag"]
                else :
                    phot_source["mag_err_pos"]='Nan.'

            phot_source['flux'].info.format = '%.3E'
            phot_source['flux_err'].info.format = '%.3E'

            if phot_source["mag"]!='Nan.':
                phot_source["mag"].info.format = "%.3f"

            if phot_source["mag_err_neg"]!='Nan.':
                phot_source["mag_err_neg"].info.format = "%.3f"

            if phot_source["mag_err_pos"]!='Nan.':
                phot_source["mag_err_pos"].info.format = "%.3f"


            os.system('mkdir -p photometry')
            os.chdir('photometry')
            phot_source.write("aperture_photometry.csv", overwrite=True)
            output=np.array(pd.read_csv("aperture_photometry.csv",delimiter=','))[0]
            os.chdir('..')
            return(output)

            # f = open("%s" % (image_file.replace(".fits", "_info.txt")), "w+")
            # f.write("Date:%s\nFilter:%s\nDetector:%s\nExposure(s):%.2f\nPivot wavelength (A):%.1f\nRMS:%.1f\nAperture correction:%.4f\nRadius:%.3f physical units" % (date, hst_filter, detector, exp_time, pivot_wavelength, filter_bandwidth, apt_corr, source_aperture.r))
            # f.close()
            # xspec_outfile = image_file.replace(".fits", "_to_xspec.ascii")
            # f = open("%s" % ("%s" % xspec_outfile), "w+")
            # f.write("%.2f %.2f %.3e %.3e" % (pivot_wavelength - filter_bandwidth / 2, pivot_wavelength + filter_bandwidth / 2, phot_source['flux'], phot_source['flux_err']))
            # f.close()
            # print("\nUse 'ftflx2xsp infile=%s xunit=angstrom yunit=ergs/cm^2/s/A nspec=1 phafile=hst_%s.fits rspfile=hst_%s.rsp' to convert to XSPEC format" % (xspec_outfile, hst_filter, hst_filter))
            # print("\nWarning: you may need to change the '-' sign in the ascii file for this to work \n")
            #
            # print some useful info

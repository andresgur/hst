# @Author: Andrés Gúrpide <agurpide>
# @Date: 24-02-2023
# @Email:  a.gurpide-lasheras@soton.ac.uk
# @Last modified by:   agurpide
# @Last modified time: 24-02-2023
import os
from astropy.io import fits
import argparse
import astropy.units as u
from astropy import wcs
import logging
import numpy as np
import stsynphot as stsyn
from synphot import Observation
from astropy.time import Time
import glob
from synphot.models import PowerLawFlux1D
from synphot import SourceSpectrum, units
from synphot import Observation
import stsynphot as stsyn
from lmfit.models import PowerLawModel
import matplotlib.pyplot as plt


working_dir = "%s/scripts/pythonscripts/hst/mastDownload/HLA/" % os.getenv("HOME")
os.chdir(working_dir)
dirs = glob.glob("%s/hst_9774_*f*w" % working_dir)
print(dirs)

bps = []
sp = SourceSpectrum(PowerLawFlux1D, alpha=0, amplitude=1, x_0=1)
fluxes = []
err = []


for directory in dirs:
    os.chdir(directory)
    image_file = glob.glob("hst_*f*w_drz.fits")[0]
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
    detector = hst_hdul[0].header["DETECTOR"] if "DETECTOR" in hst_hdul[0].header else ""
    uv_filters = ["F200LP", "F300X", "F218W", "F225W", "F275W", "FQ232N", "FQ243N", "F280N"]
    if detector == "UVIS" and filter in uv_filters:
        photflam = float(hst_hdul[0].header["PHTFLAM1"])
    elif "PHOTFLAM" in hst_hdul[0].header:
        photflam = float(hst_hdul[0].header["PHOTFLAM"])
    elif "PHOTFLAM" in hst_hdul[1].header:
        photflam = float(hst_hdul[1].header["PHOTFLAM"])
    photflam = photflam * u.erg / u.AA / u.s / u.cm**2
    file = glob.glob("aperture_phot_*.csv")[0]
    data = np.genfromtxt("%s" % (file), names=True, delimiter=",")
    fluxes.append(data["der_flux"])
    err.append(data["der_flux_err"])
    os.chdir(working_dir)
    bps.append(stsyn.band(obsmode))


pivots = [bp.pivot().value for bp in bps]
plt.errorbar(pivots, fluxes, yerr=err, ls="None")
plt.xlabel("Wavelength ($\AA$)")
sorting = np.argsort(pivots)
y = np.array(fluxes)[sorting]
x = np.array(pivots)[sorting]
yerr = np.array(err)[sorting]

powerlaw = PowerLawModel()
params= powerlaw.guess(y, x=x)
result = powerlaw.fit(y, params, x=x, weights=1/yerr)
sp = SourceSpectrum(PowerLawFlux1D, alpha=-result.best_values["exponent"],
                    amplitude=result.best_values["amplitude"], x_0=1)
#power_model.set_param_hint("amplitude", min=fluxes[-1], max=fluxes[0], value=-1)
#power_model.set_param_hint("exponent", min=0, max=5,value=1)
first_exponent = result.best_values["exponent"]
exponent = first_exponent
new_exponent = 0

i = 1
plt.figure()

while (np.abs((exponent - new_exponent) / exponent) > 0.01) and i<10:
    obs = [Observation(sp, bp) for bp in bps]
    new_fluxes = [ob.effstim(area=stsyn.conf.area).value for ob in obs]
    ynew = np.array(new_fluxes)[sorting]
    result = powerlaw.fit(ynew, params, x=x, weights=1/yerr)
    new_exponent = result.best_values["exponent"]
    sp = SourceSpectrum(PowerLawFlux1D, alpha=-result.best_values["exponent"],
                    amplitude=result.best_values["amplitude"], x_0=1)
    print("Iteration %d" % i)
    i+=1
plot_x = np.arange(min(x), max(x), 100)
plt.plot(plot_x, result.eval(x=plot_x), label='Last best-fit ($\\alpha$ = %.2f)' % new_exponent, ls="--")
print("Converged in %d iterations" % i)
print("First exponent")
print(first_exponent)
print("Newly found exponent")
print(new_exponent)
#plt.yscale("log")
#plt.xscale("log")
plt.ylabel("F$\lambda$ (erg/s/cm$^2$/$\AA$)")
plt.xlabel("$\AA$")
plt.errorbar(pivots, fluxes, yerr=yerr, fmt="o", label="Old")
plt.errorbar(x, ynew, yerr=yerr, fmt="o", label="New")
plt.legend()
plt.savefig("iteration_%d.png" %i, dpi=100)
print("Pivot wavelengths (\AA)")
print(x)
stmag = [-2.5 * np.log10(flux) - 21.1 for flux in y]
print("Original ST magnitudes:")
print(stmag)
print("Recalculated ST magntiudes:")
stmag = [-2.5 * np.log10(flux) - 21.1 for flux in ynew]
print(stmag)

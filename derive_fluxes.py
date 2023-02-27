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

parser = argparse.ArgumentParser(description='Refines fluxes using a powerlaw for the intrinsic fluxes.')
parser.add_argument("directories", nargs="+", help="Directories with the HST images and flux information", type=str)
parser.add_argument("-o", "--outdir", nargs="?", help="Output directory. Default is pow_fluxes_", type=str, default="")
args = parser.parse_args()

working_dir = os.getcwd()
print(working_dir)
os.chdir(working_dir)
dirs = args.directories


outdir = "pow_fluxes_" + args.outdir

if not os.path.isdir(outdir):
    os.mkdir(outdir)

bps = []
sp = SourceSpectrum(PowerLawFlux1D, alpha=0, amplitude=1, x_0=1)
fluxes = []
err = []
stmags = []
errmags = []

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
    stmags.append(data["derstmag"])
    errmags.append(data["derstmag_err"])
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

exponent_file = open("%s/exponents.dat" % outdir, "w+")
exponent_file.write("#iteration\texponent\n")
exponent_file.write("%d\t%.3f\n"%(0, exponent))
while (np.abs((exponent - new_exponent) / exponent) > 0.01) and i<10:
    obs = [Observation(sp, bp) for bp in bps]
    new_fluxes = [ob.effstim(area=stsyn.conf.area).value for ob in obs]
    ynew = np.array(new_fluxes)[sorting]
    result = powerlaw.fit(ynew, params, x=x, weights=1/yerr)
    new_exponent = result.best_values["exponent"]
    sp = SourceSpectrum(PowerLawFlux1D, alpha=-result.best_values["exponent"],
                    amplitude=result.best_values["amplitude"], x_0=1)
    print("Iteration %d" % i)
    exponent_file.write("%d\t%.3f\n"%(i, new_exponent))
    i+=1

exponent_file.close()
plot_x = np.arange(min(x), max(x), 100)
plt.plot(plot_x, result.eval(x=plot_x), label='Last best-fit ($\\alpha$ = %.2f)' % new_exponent, ls="--")
print("Converged in %d iterations" % i)
#plt.yscale("log")
#plt.xscale("log")
plt.ylabel("F$\lambda$ (erg/s/cm$^2$/$\AA$)")
plt.xlabel("$\AA$")
plt.errorbar(pivots, fluxes, yerr=yerr, fmt="o", label="Old")
plt.errorbar(x, ynew, yerr=yerr, fmt="o", label="New")
plt.legend()
plt.savefig("%s/iteration_%d.png" % (outdir,i), dpi=100)


print("Pivot wavelengths (\AA)")
print(x)

print("Original ST magnitudes:")
print(stmags)
print("Original fluxes")
print(fluxes)
print("Recalculated ST magntiudes:")
stmag = [-2.5 * np.log10(flux) - 21.1 for flux in fluxes]
print(stmag)
print("Recalculated fluxes")
print(new_fluxes)
scale = 10**-18
old_stmag = ["%.2f$\pm$%.2f" %(stmag, err) for stmag, err in zip(stmags, errmags)]
old_fluxes = ["%.2f$\pm$%.2f" %(flux/ scale, err/ scale) for flux, err in zip(fluxes, err)]
new_stmag = [-2.5 * np.log10(flux) - 21.1 for flux in new_fluxes]
new_stmag = ["%.2f$\pm$%.2f" %(stmag, err) for stmag, err in zip(new_stmag, errmags)]
new_fluxes_str = ["%.2f$\pm$%.2f" %(flux/ scale, err/ scale) for flux, err in zip(new_fluxes, err)]
outputs = np.vstack((dirs, pivots, old_stmag, old_fluxes, new_stmag, new_fluxes_str))
np.savetxt("%s/results.dat" % outdir, outputs.T, header="#directory\tpivots\tstmag\tfluxes(%.1e)\tnewstmag\tnewfluxes" % scale, delimiter="\t", fmt="%s")#, header="#directory\tpivots\tfluxes\tstmag\tnewfluxes\tnewstmag", fmt="%s\t%.2f\t%s\t%s\t%s\t%s")

xspec_file = open("%s/to_xspec.txt" % outdir, "w+")

for pivot_wavelength, flux, flux_error, bp in zip(pivots, new_fluxes, err, bps):
    filter_bandwidth = bp.rectwidth().value # http://svo2.cab.inta-csic.es/theory/fps/index.php?id=HST/ACS_HRC.F555W&&mode=browse&gname=HST&gname2=ACS_HRC#filter
    xspec_file.write("%.1f %.2f %.3e %.3e\n" % (pivot_wavelength - filter_bandwidth / 2, pivot_wavelength + filter_bandwidth / 2, flux, flux_error))
xspec_file.write("#ftflx2xsp infile=%s xunit=angstrom yunit=ergs/cm^2/s/A nspec=1 phafile=hst_%s.fits rspfile=hst_%s.rsp" % ("to_xspec.txt", args.outdir, args.outdir))
xspec_file.close()


print("Outputs stored to %s" % outdir)

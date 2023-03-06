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
from synphot import SourceSpectrum,ReddeningLaw, Observation
from extinction import ccm89, calzetti00, remove
from lmfit.models import PowerLawModel
import stsynphot as stsyn
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Refines fluxes using a powerlaw for the intrinsic fluxes.')
parser.add_argument("directories", nargs="+", help="Directories with the HST images and flux information", type=str)
parser.add_argument("-o", "--outdir", nargs="?", help="Output directory. Default is pow_fluxes_", type=str, default="")
parser.add_argument("-E", "--EBV", nargs=1, type=float, help="E(B-V) value for the extinction correction. Default assumes no extinction",
                    default=0)
args = parser.parse_args()

working_dir = os.getcwd()
print(working_dir)
os.chdir(working_dir)
dirs = args.directories
EBV = args.EBV[0]
outdir = "pow_fluxes_" + args.outdir

if not os.path.isdir(outdir):
    os.mkdir(outdir)

bps = []
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
    fluxes.append(data["fluxerg__Angstrom_cm2_s"]) # observed fluxes
    err.append(data["flux_err"])
    stmags.append(data["derstmag"])
    errmags.append(data["derstmag_err"])
    os.chdir(working_dir)
    bps.append(stsyn.band(obsmode))


pivots = [bp.pivot().value for bp in bps]
plt.xlabel("Wavelength ($\AA$)")
sorting = np.argsort(pivots)
fluxes = np.array(fluxes)[sorting]
pivots = np.array(pivots)[sorting]
yerr = np.array(err)[sorting]
bps = np.array(bps)[sorting]
dirs = np.array(dirs)[sorting]
plt.errorbar(pivots, fluxes, yerr=yerr, ls="None")

# fit powerlaw to the OBSERVED fluxes
powerlaw = PowerLawModel()

params= powerlaw.guess(fluxes, x=pivots)
powerlaw.set_param_hint("amplitude", max=np.inf, value=1.0017809528994497e-14) # setting a minimum value braks everything
powerlaw.set_param_hint("exponent", max=0, value=-2, min=-10)
params = powerlaw.make_params()
print(params)
result = powerlaw.fit(fluxes, params, x=pivots, weights=1/yerr)

observed_sp = SourceSpectrum(PowerLawFlux1D, alpha=-result.best_values["exponent"],
                    amplitude=result.best_values["amplitude"], x_0=1)
#power_model.set_param_hint("amplitude", min=fluxes[-1], max=fluxes[0], value=-1)
#power_model.set_param_hint("exponent", min=0, max=5,value=1)
exponent = -result.best_values["exponent"]
amplitude = result.best_values["amplitude"]

i = 1
plt.figure()

Rv = 3.1
av = Rv * EBV

#mwavg_file == Cardelli+1989 milkiway diffuse Rv=3.1
#Calzetti starbust diffuse xgal_file
deredden = ReddeningLaw.from_extinction_model('mwavg').extinction_curve(-EBV)
ext = ReddeningLaw.from_extinction_model('mwavg').extinction_curve(EBV)


# derreden the powerlaw
intrinsic_sp = observed_sp * deredden
# fit the intrinsic powerlaw
result = powerlaw.fit(intrinsic_sp(pivots).value, params, x=pivots, weights=1/yerr)

new_exponent = -result.best_values["exponent"]
exponent_file = open("%s/exponents.dat" % outdir, "w+")
exponent_file.write("#iteration\texponent\n")
exponent_file.write("%d\t%.3f\n"%(0, new_exponent))

exponent = 0.1

while (np.abs((exponent - new_exponent) / exponent) > 0.01) and i<100:
    exponent = new_exponent

    intrinsic_sp = SourceSpectrum(PowerLawFlux1D, alpha=new_exponent,
                        amplitude=result.best_values["amplitude"], x_0=1)
    obs = [Observation(intrinsic_sp * ext, bp) for bp in bps] # create observations with the extiguished spectrum
    new_fluxes = [ob.effstim(area=stsyn.conf.area).value for ob in obs]
    result = powerlaw.fit(new_fluxes, params, x=pivots, weights=1/yerr)
    observed_sp = SourceSpectrum(PowerLawFlux1D, alpha=-result.best_values["exponent"],
                        amplitude=result.best_values["amplitude"], x_0=1)
    intrinsic_sp = observed_sp * deredden
    result = powerlaw.fit(intrinsic_sp(pivots).value, params, x=pivots, weights=1/yerr)
    new_exponent = -result.best_values["exponent"]
    print("Iteration %d" % i)
    exponent_file.write("%d\t%.3f\n"%(i, new_exponent))
    i+=1

##print(result.best_values["amplitude"])
new_intrinsic_fluxes = intrinsic_sp(pivots).value

exponent_file.close()
plot_x = np.arange(min(pivots), max(pivots), 10)
plt.plot(plot_x, result.eval(x=plot_x), label='Last best-fit ($\\alpha$ = %.2f)' % new_exponent, ls="--")
print("Converged in %d iterations" % i)
#plt.yscale("log")
#plt.xscale("log")
plt.ylabel("F$\lambda$ (erg/s/cm$^2$/$\AA$)")
plt.xlabel("$\AA$")
plt.errorbar(pivots, fluxes, yerr=yerr, fmt="o", label="Old")
plt.errorbar(pivots, new_intrinsic_fluxes, yerr=yerr, fmt="o", label="New")
plt.legend()
plt.savefig("%s/iteration_%d.png" % (outdir,i), dpi=100)


print("Pivot wavelengths (\AA)")
print(pivots)

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
new_stmag = [-2.5 * np.log10(flux) - 21.1 for flux in new_intrinsic_fluxes]
new_stmag = ["%.2f$\pm$%.2f" %(stmag, err) for stmag, err in zip(new_stmag, errmags)]
new_fluxes_str = ["%.2f$\pm$%.2f" %(flux/ scale, err/ scale) for flux, err in zip(new_intrinsic_fluxes, err)]
outputs = np.vstack((dirs, pivots, old_stmag, old_fluxes, new_stmag, new_fluxes_str))
np.savetxt("%s/results.dat" % outdir, outputs.T, header="#directory\tpivots\tstmag\tfluxes(%.1e)\tnewstmag\tnewfluxes" % scale, delimiter="\t", fmt="%s")#, header="#directory\tpivots\tfluxes\tstmag\tnewfluxes\tnewstmag", fmt="%s\t%.2f\t%s\t%s\t%s\t%s")

xspec_file = open("%s/to_xspec.txt" % outdir, "w+")

for pivot_wavelength, flux, flux_error, bp in zip(pivots, new_fluxes, err, bps):
    filter_bandwidth = bp.rectwidth().value # http://svo2.cab.inta-csic.es/theory/fps/index.php?id=HST/ACS_HRC.F555W&&mode=browse&gname=HST&gname2=ACS_HRC#filter
    xspec_file.write("%.1f %.2f %.3e %.3e\n" % (pivot_wavelength - filter_bandwidth / 2, pivot_wavelength + filter_bandwidth / 2, flux, flux_error))
xspec_file.write("#ftflx2xsp infile=%s xunit=angstrom yunit=ergs/cm^2/s/A nspec=1 phafile=hst_%s.fits rspfile=hst_%s.rsp" % ("to_xspec.txt", args.outdir, args.outdir))
xspec_file.close()


print("Outputs stored to %s" % outdir)

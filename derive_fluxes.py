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
from synphot.models import PowerLawFlux1D, Empirical1D
from synphot import SourceSpectrum,ReddeningLaw, Observation
from extinction import ccm89, calzetti00, remove
from lmfit.models import PowerLawModel
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def minimize_powerlaw(params):
    index = params[0]
    amplitude = params[1]
    sp = SourceSpectrum(PowerLawFlux1D, alpha=-index,
                        amplitude=amplitude * photflam_units, x_0=1) # powerlaw
    obs = [Observation(sp * ext, bp) for bp in bps] # absorbed powerlaw
    model_rates = [ob.countrate(area=stsyn.conf.area).value for ob in obs]
    return countrates_residuals(rates, uncertainties, model_rates)

def minimize_index(params):
    index = params[0]
    sp = SourceSpectrum(PowerLawFlux1D, alpha=-index,
                        amplitude=1 * photflam_units,
                        x_0=1) # powerlaw
    sp = sp * ext # extinguish it
    sp = sp.normalize(rates[index_norm] * u.ct, bps[index_norm], area=stsyn.conf.area) # renormalize to match count rates
    obs = [Observation(sp, bp) for bp in bps] # absorbed powerlaw
    model_rates = [ob.countrate(area=stsyn.conf.area).value for ob in obs]
    return countrates_residuals(rates, uncertainties, model_rates)

def countrates_residuals(rates, uncertainties, model_rates):
    return np.sum(((rates - model_rates)/uncertainties)**2)

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
photflam_units = u.erg / u.AA / u.s / u.cm**2
#mwavg == Cardelli+1989 milkiway diffuse Rv=3.1
#Calzetti starbust diffuse xgal

fluxes = []
err = []
stmags = []
errmags = []
rates = []
uncertainties = []

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
    fluxes.append(data["fluxerg__Angstrom_cm2_s"]) # der_flux, fluxerg__Angstrom_cm2_s
    err.append(data["flux_err"])
    stmags.append(data["derstmag"])
    errmags.append(data["derstmag_err"])
    os.chdir(working_dir)
    bps.append(stsyn.band(obsmode))
    rates.append(data["corrected_aperture_sumELECTRONSS"])
    uncertainties.append(data["corrected_aperture_err"])

# sort everything in ascending wavelength
pivots = [bp.pivot().value for bp in bps]
plt.xlabel("Wavelength ($\AA$)")
sorting = np.argsort(pivots)
fluxes = np.array(fluxes)[sorting]
pivots = np.array(pivots)[sorting]
yerr = np.array(err)[sorting]
bps = np.array(bps)[sorting]
dirs = np.array(dirs)[sorting]
stmags = np.array(stmags)[sorting]
rates = np.array(rates)[sorting]
uncertainties = np.array(uncertainties)[sorting]
plt.errorbar(pivots, fluxes, yerr=yerr, ls="None")

Rv = 3.1
av = Rv * EBV
mag_ext = ccm89(pivots, av, Rv, unit="aa")

intrinsic_fluxes = remove(mag_ext, fluxes) # derreden the fluxes

ext = ReddeningLaw.from_extinction_model('mwavg').extinction_curve(EBV)
derreden = ReddeningLaw.from_extinction_model('mwavg').extinction_curve(-EBV)


powerlaw = PowerLawModel()


if len(fluxes) > 2:
    print("Minimizing powerlaw index and normalization...")

    bnds = ((-np.inf, 0), (1e-20, np.inf))
    params = powerlaw.guess(intrinsic_fluxes, x=pivots)
    result = powerlaw.fit(intrinsic_fluxes, params, x=pivots, weights=1/yerr)
    #exponent = first_exponent

    result = minimize(minimize_powerlaw, (result.best_values["exponent"],
                      result.best_values["amplitude"]),
                      bounds=bnds, method="L-BFGS-B", tol=1e-6)

    new_exponent = result.x[0]
    new_amplitude = result.x[1]
# if only one datapoint minimize the index only
else:
    print("Minimizing powerlaw index...")
    bnds = ((-np.inf, 0),)
    index_norm = np.argmin(uncertainties) # fix the normalization to the value with the lowest uncertainty
    result = minimize(minimize_index, np.array([-1.4]),
                          bounds=bnds, method="L-BFGS-B", tol=1e-6)
    new_exponent = result.x[0]
    sp = SourceSpectrum(PowerLawFlux1D, alpha=-new_exponent,
                        amplitude=1 * photflam_units,
                        x_0=1) # powerlaw
    sp = sp * ext # extinguish it
    # adjust normalization and derreden it
    sp = sp.normalize(rates[index_norm] * u.ct, bps[index_norm], area=stsyn.conf.area) * derreden
    new_amplitude = sp.model.factor_2

print("Done\n")
print(result)

best_params = powerlaw.make_params(exponent=new_exponent, amplitude=new_amplitude)

sp = SourceSpectrum(PowerLawFlux1D, alpha=-new_exponent,
                    amplitude=new_amplitude * photflam_units, x_0=1) # best-fit powerlaw

# residual plot
obs = [Observation(sp * ext, bp) for bp in bps]
model_counts = [ob.countrate(area=stsyn.conf.area).value for ob in obs]
plt.figure()
plt.errorbar(pivots, model_counts - rates, yerr=uncertainties, color="black")
chi_square = np.sum(((model_counts - rates) / uncertainties)**2)
print("\nChi-square/dof = %.2f/%d\n" % (chi_square, len(rates) - 2))
plt.xlabel("$\AA$")
plt.axhline(y=0, ls="solid", color="black", zorder=-10, lw=1)
plt.ylabel("Observed - model (ct/s)")
plt.savefig("%s/residuals.png" % outdir)

sp = SourceSpectrum(PowerLawFlux1D, alpha=-new_exponent,
                    amplitude=new_amplitude, x_0=1) # best-fit powerlaw
# with the best fit powerlaw, get now the INTRINSIC FLUXES
intrinsic_fluxes = sp(pivots).value
#plt.yscale("log")
#plt.xscale("log")
plt.figure()
plot_x = np.arange(min(pivots), max(pivots), 100)
plt.plot(plot_x, powerlaw.eval(params=best_params, x=plot_x), label='Last best-fit ($\\alpha$ = %.2f)' % new_exponent,
         ls="--")
plt.ylabel("F$_\lambda$ (erg/s/cm$^2$/$\AA$)")
plt.xlabel("$\AA$")
plt.errorbar(pivots, fluxes, yerr=yerr, fmt="o", label="Old extinction-corrected")
plt.errorbar(pivots, intrinsic_fluxes, yerr=yerr, fmt="o", label="Extinction-corrected fluxes")
plt.legend()
plt.savefig("%s/minimized.png" % (outdir), dpi=100)

print("Pivot wavelengths (\AA)")
print(pivots)

print("Original ST magnitudes:")
print(stmags)
print("Original fluxes")
print(fluxes)
print("Recalculated ST magntiudes:")
stmag = [-2.5 * np.log10(flux) - 21.1 for flux in intrinsic_fluxes]
print(stmag)
print("Recalculated fluxes")
print(intrinsic_fluxes)
scale = 10**-18
old_stmag = ["%.2f$\pm$%.2f" %(stmag, err) for stmag, err in zip(stmags, errmags)]
old_fluxes = ["%.2f$\pm$%.2f" %(flux/ scale, err/ scale) for flux, err in zip(fluxes, err)]
new_stmag = [-2.5 * np.log10(flux) - 21.1 for flux in intrinsic_fluxes]
new_stmag = ["%.2f$\pm$%.2f" %(stmag, err) for stmag, err in zip(new_stmag, errmags)]
intrinsic_fluxes_str = ["%.2f$\pm$%.2f" %(flux/ scale, err/ scale) for flux, err in zip(intrinsic_fluxes, err)]
outputs = np.vstack((dirs, pivots, old_stmag, old_fluxes, new_stmag, intrinsic_fluxes_str))
np.savetxt("%s/results.dat" % outdir, outputs.T, header="directory\tpivots\tstmag\tfluxes(%.1e)\tnewstmag\tnewfluxes" % scale,
            delimiter=" & ", fmt="%s")#, header="#directory\tpivots\tfluxes\tstmag\tnewfluxes\tnewstmag", fmt="%s\t%.2f\t%s\t%s\t%s\t%s")

xspec_file = open("%s/to_xspec.txt" % outdir, "w+")

for pivot_wavelength, flux, flux_error, bp in zip(pivots, intrinsic_fluxes, err, bps):
    filter_bandwidth = bp.rectwidth().value # http://svo2.cab.inta-csic.es/theory/fps/index.php?id=HST/ACS_HRC.F555W&&mode=browse&gname=HST&gname2=ACS_HRC#filter
    xspec_file.write("%.1f %.2f %.3e %.3e\n" % (pivot_wavelength - filter_bandwidth / 2, pivot_wavelength + filter_bandwidth / 2, flux, flux_error))
xspec_file.write("#ftflx2xsp infile=%s xunit=angstrom yunit=ergs/cm^2/s/A nspec=1 phafile=hst_%s.fits rspfile=hst_%s.rsp clobber=yes" % ("to_xspec.txt", args.outdir, args.outdir))
xspec_file.close()


print("Outputs stored to %s" % outdir)

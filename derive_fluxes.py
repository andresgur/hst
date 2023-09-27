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
from synphot.blackbody import BlackBody1D
from synphot import SourceSpectrum,ReddeningLaw, Observation
from extinction import ccm89, calzetti00, remove
from lmfit.models import PowerLawModel
import matplotlib.pyplot as plt
from scipy.optimize import minimize, least_squares


def minimize_model(params):
    # dictionary of variable parameters
    param_dict = {name:value for name,value in zip(varparamnames, params)}
    # add fixed parameters
    for paramkey, param in zip(model.fixed.keys(), model.parameters):
        if model.fixed[paramkey]:
            param_dict[paramkey] = param

    if "amplitude" in param_dict:
        param_dict["amplitude"] = param_dict["amplitude"] * photflam_units # add units
    elif "temperature" in param_dict:
        param_dict["temperature"] = param_dict["temperature"] * u.K
    elif "bolometric_flux" in param_dict:
        param_dict["bolometric_flux"] = param_dict["bolometric_flux"] * u.Hz

    sp = SourceSpectrum(model.__class__, **param_dict, z=z) # model
    obs = [Observation(sp * ext, bp) for bp in bps] # absorbed model
    model_rates = [ob.countrate(area=stsyn.conf.area).value for ob in obs] # rates from absorbed model
    return countrates_residuals(rates, uncertainties, model_rates)


def minimize_index(params):
    index = params[0]
    sp = SourceSpectrum(PowerLawFlux1D, alpha=-index,
                        amplitude=1 * photflam_units,
                        x_0=1, z=z) # powerlaw
    sp = sp * ext # extinguish it
    sp = sp.normalize(rates[index_norm] * u.ct, bps[index_norm], area=stsyn.conf.area) # renormalize to match count rates
    obs = [Observation(sp, bp) for bp in bps] # absorbed powerlaw
    model_rates = [ob.countrate(area=stsyn.conf.area).value for ob in obs]
    return countrates_residuals(rates, uncertainties, model_rates)

def countrates_residuals(rates, uncertainties, model_rates):
    return (rates - model_rates)/uncertainties

parser = argparse.ArgumentParser(description='Refines fluxes using a powerlaw for the intrinsic fluxes.')
parser.add_argument("directories", nargs="+", help="Directories with the HST images and flux information", type=str)
parser.add_argument("-o", "--outdir", nargs="?", help="Output directory. Default is pow_fluxes_", type=str, default="")
parser.add_argument("-E", "--EBV", nargs=1, type=float, help="E(B-V) value for the extinction correction. Default assumes no extinction",
                    default=0)
parser.add_argument("-n", "--nsims", metavar="N", nargs="?", type=int, help="Number of powerlaw draws for the error calculation. At least 500. Default 600",
                    default=600)
parser.add_argument("-m", "--model", nargs="?", metavar="model", type=str, choices=["blackbody", "powerlaw"], help="Model for the derivation of fluxes. Default powerlaw",
                    default="powerlaw")
parser.add_argument("-z", "--redshift", nargs="?", metavar="z", type=float, help="Redshift of the spectrum",
                    default=0)
args = parser.parse_args()

working_dir = os.getcwd()
print(working_dir)
os.chdir(working_dir)
dirs = args.directories
print("Analysing the following directories")
print(dirs)
EBV = args.EBV[0]
z = args.redshift
#mwavg == Cardelli+1989 milkiway diffuse Rv=3.1
#Calzetti starbust diffuse xgal
#Rv = 3.1
extinction_curve = "Cardelli"

outdir = "%s_fluxes_%s_EBV_%.2f_%s_z%.2e" %(args.model, args.outdir, EBV, extinction_curve, z)

if not os.path.isdir(outdir):
    os.mkdir(outdir)

bps = []
photflam_units = u.erg / u.AA / u.s / u.cm**2

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

if extinction_curve == "Cardelli":
    Rv=3.1
    keyword = "mwavg"
    av = Rv * EBV
    mag_ext = ccm89(pivots, av, Rv, unit="aa")

elif extinction_curve=="Calzetti":
    Rv=4.05
    keyword = "xgalsb"
    av = Rv * EBV
    mag_ext = calzetti00(pivots, av, Rv, unit="aa")

print("Using extinction curve from %s with Rv = %.2f" % (extinction_curve, Rv))

intrinsic_fluxes = remove(mag_ext, fluxes) # derreden the fluxes
ext = ReddeningLaw.from_extinction_model(keyword).extinction_curve(EBV)
derreden = ReddeningLaw.from_extinction_model(keyword).extinction_curve(-EBV)


if args.model=="powerlaw":
    # set the norm to the value with the lowest normalization as a starting point
    index_norm = np.argmin(uncertainties)
    # bracket alpha between two sensitive values
    model = PowerLawFlux1D(alpha=2, amplitude=intrinsic_fluxes[index_norm], x_0=pivots[index_norm],
                bounds={"alpha":[0, 10], "amplitude":[1e-26, np.inf]}, fixed={"x_0": True})
elif args.model == "blackbody":
    distance = 4 * u.Mpc
    bolflux = 10**37 / (4 * np.pi * distance.to(u.cm).value**2) * u.Hz# u.erg/u.s /
    bolbounds = [10**20/ (4 * np.pi * distance.to(u.cm).value**2), 10**39 / (4 * np.pi * distance.to(u.cm).value**2)]
    model = BlackBody1D(temperature=40000 * u.K, bolometric_flux=bolflux,
                        bounds={"temperature":[1000, 100000], "bolometric_flux":bolbounds})
    raise Exception("Blackbody model was never implemented due to unit problems with astropy and synphot")

print(model)


nsims = args.nsims

if len(fluxes)<=2:
    # if there's only 2 fluxes, fix the normalization and only fit for the index of the powerlaw
    if "amplitude" in model.param_names:
        model.amplitude.fixed = True
    elif "bolometric_flux" in model.param_names:
        model.bolometric_flux.fixed = True
    print("Less than two flux data points. Amplitude will be fixed during the fit")

varparams = {parname:param for parname, param in zip(model.param_names, model.parameters) if not model.fixed[parname]}
varparamnames = [*varparams.keys()]
# create bounds list for least_square method
bounds_list = ([],[])
for i in range(2):
    for param_name in varparamnames:
        bounds_list[i].append(model.bounds[param_name][i])

# pass the parameters
result = least_squares(minimize_model, x0=[*varparams.values()], bounds=bounds_list)

best_params = result.x
print("Best-fit parameters minimizing count rate residuals:")
print(best_params)
# draw powerlaws from covariance
J = result.jac
cov = np.linalg.inv(J.T.dot(J))
samples = np.random.multivariate_normal(result.x, cov, size=nsims).T

plt.figure()
try:
    plt.scatter(samples[0], samples[1])
    plt.ylabel("$\\alpha$")
    plt.xlabel("$N$")
except IndexError:
    plt.hist(samples.T)
    plt.ylabel("Instances")
    plt.xlabel("Parameter")
plt.savefig("%s/parsamples_%d.png" % (outdir, nsims))

sampled_fluxes = [ ]
# the code below can be improved
for sample_params in samples.T:
    param_dict = {name:value for name,value in zip(varparams.keys(), sample_params)}
    if "amplitude" in param_dict:
        if (param_dict["amplitude"] < 0) or (param_dict["alpha"] <0):
            continue
    elif "bolometric_flux" in param_dict:
        if param_dict["bolometric_flux"] < 0:
            continue

    # add fixed parameters back
    for paramkey, param in zip(model.fixed.keys(), model.parameters):
        if model.fixed[paramkey]:
            param_dict[paramkey] = param
    sampled_fluxes.append(SourceSpectrum(model.__class__, **param_dict, z=z)(pivots).value)

     # fix the normalization to the value with the lowest uncertainty
    #result = least_squares(minimize_index, -2.72,
                          #bounds=bnds)
    #new_exponent = result.x[0]
    #sp = SourceSpectrum(PowerLawFlux1D, alpha=-new_exponent,
                        #amplitude=1 * photflam_units,
                        #x_0=1) # powerlaw
    #sp = sp * ext # extinguish it
    # adjust normalization of the absorbed spectrum and derreden it
    #sp = sp.normalize(rates[index_norm] * u.ct, bps[index_norm], area=stsyn.conf.area) * derreden
    #new_amplitude = sp.model.factor_2.value

    # draw powerlaws from covariance
    #J = result.jac
    #cov = np.linalg.inv(J.T.dot(J))
    #if len(pivots)==1:
    #    std = 0
    #else:
#        std = np.sqrt(np.diagonal(cov))
#    exponents = np.random.normal(new_exponent, std, size=nsims).T
#    sampled_fluxes = []
#    for exp in exponents:
#        sp_exp_ext = SourceSpectrum(PowerLawFlux1D, alpha=-exp, amplitude=1 * photflam_units, x_0=1) * ext
#        sp_exp_derr = sp_exp_ext.normalize(rates[index_norm] * u.ct, bps[index_norm], area=stsyn.conf.area) * derreden
#        exp_amplitude = sp_exp_derr.model.factor_2.value
#        new_flux_exp = SourceSpectrum(PowerLawFlux1D, alpha=-exp, amplitude=exp_amplitude, x_0=1)(pivots).value
#        sampled_fluxes.append(new_flux_exp)
print("Done\n")

print(result)

std = np.sqrt(np.diagonal(cov))
plt.figure()

for wavelength, filter_fluxes in zip(pivots, np.array(sampled_fluxes).T):
    plt.figure()
    plt.hist(filter_fluxes)
    plt.axvline(np.mean(filter_fluxes), ls="solid", label="%.2e+-%.2e" % (np.mean(filter_fluxes), np.std(filter_fluxes)))
    plt.axvline(np.mean(filter_fluxes) + np.std(filter_fluxes), ls="--")

    plt.xlabel("Intrinsic fluxes ($\lambda$ = %.1f)" % wavelength)
    plt.legend()
    plt.savefig("%s/sampled_fluxes_%.2f_%.1f.png" % (outdir, std[-1], wavelength))

best_params_dict = {parname:param for parname, param in zip(varparamnames, best_params)}
# add fixed parameters back
for paramkey, param in zip(model.fixed.keys(), model.parameters):
    if model.fixed[paramkey]:
        best_params_dict[paramkey] = param

if "alpha" in model.param_names:
    print("Best fit powerlaw F ~ x^{%.2f$\pm$%.2f}" % (best_params_dict["alpha"], std[-1]))
elif "temperature" in model.param_names:
    print("Best fit blackbody T ~ %.2f$\pm$%.2f" % (best_params_dict["temperature"], std[0]))

if "amplitude" in best_params_dict:
    best_params_dict["amplitude"] = best_params_dict["amplitude"] * photflam_units

sp = SourceSpectrum(model.__class__, **best_params_dict, z=z) # best-fit powerlaw

# residual plot
obs = [Observation(sp * ext, bp) for bp in bps]
model_counts = [ob.countrate(area=stsyn.conf.area).value for ob in obs]
plt.figure()
plt.errorbar(pivots, model_counts - rates, yerr=uncertainties, color="black")
chi_square = np.sum(((model_counts - rates) / uncertainties)**2)
nvarparams = len(varparamnames)
dof =  len(rates) - nvarparams
print("\nChi-square/dof = %.2f/%d\n" % (chi_square, dof))
plt.xlabel("$\AA$")
plt.axhline(y=0, ls="solid", color="black", zorder=-10, lw=1)
plt.ylabel("Observed - model (ct/s)")
plt.text(0.5, 0.2, '$\chi^2$/dof = %.2f/%d' % (chi_square, dof), horizontalalignment='center',
     verticalalignment='center', transform=plt.gca().transAxes, fontsize=22)
plt.savefig("%s/residuals.png" % outdir)

# WE NOW REMOVE THE PHOTFLAM UNITS to get the fluxes in the right units!!!!
if "amplitude" in best_params_dict:
    best_params_dict["amplitude"] = best_params_dict["amplitude"].value

sp = SourceSpectrum(model.__class__, **best_params_dict, z=z) # best-fit powerlaw
# with the best fit powerlaw, get now the INTRINSIC FLUXES
intrinsic_fluxes = sp(pivots).value
# add the newfly found uncertainty with the previous one in quadrature (which takes into account the error on the count rate)
# I don't think we need to take into account the previous uncertainty as it is already embedded in the powerlaw uncertainties
# no the yerr are already calculated in the index of the powerlaw which we then convolve onto the uncertainties of the final sampled fluxes
err_intrinsic_fluxes = np.std(sampled_fluxes, axis=0)
# for few datapoints include the original uncertainty on the final estimation as we do not randomized the amplitude
if len(fluxes)>2:
    err_intrinsic_fluxes = np.std(sampled_fluxes, axis=0) #
else:
    err_intrinsic_fluxes = np.sqrt(err_intrinsic_fluxes**2 + yerr**2) #
print("Pivot wavelengths (\AA)")
print(pivots)

for pivot, flux, err_intrin in zip(pivots, intrinsic_fluxes, err_intrinsic_fluxes):
    print("Pivot:%.1f --> Derredened flux (1e-18 erg/cm2/AA/s): %.2f$pm$%.2f" %(pivot, flux * 1e18, err_intrin * 1e18))

#plt.yscale("log")
#plt.xscale("log")
plt.figure()

plot_x = np.arange(min(pivots), max(pivots), 100)

for par in best_params_dict.keys():
    setattr(model, par, best_params_dict[par])

if "alpha" in model.param_names:
    label = "Last best-fit ($\\alpha$ = %.2f+-%.3f)" %(best_params_dict["alpha"], std[-1])
elif "temperature" in model.param_names:
    label = "Last best-fit ($T$ = %.2f+-%.2f)" %(best_params_dict["temperature"], std[0])
else:
    label = None

plt.plot(plot_x, model(plot_x), label=label, ls="--")
plt.ylabel("F$_\lambda$ (erg/s/cm$^2$/$\AA$)")
plt.xlabel("$\AA$")
plt.errorbar(pivots, fluxes, yerr=yerr, fmt="o", label="Observed fluxes")
plt.errorbar(pivots, intrinsic_fluxes, yerr=err_intrinsic_fluxes, fmt="o", label="Extinction-corrected (E(B-V) = %.2f) fluxes" % (EBV))
plt.legend()
plt.savefig("%s/minimized.png" % (outdir), dpi=100)


print("Original ST magnitudes:")
print(stmags)
print("Original fluxes")
print(fluxes)
new_stmag = [-2.5 * np.log10(flux) - 21.1 for flux in intrinsic_fluxes]
# add the new uncertainty and the old one from the count rates in cuadrature
new_stmag_err = [2.5 * err_flux / (flux * np.log(10)) for flux, err_flux in zip(intrinsic_fluxes, err_intrinsic_fluxes)]
print("Recalculated ST magntiudes:")
print(new_stmag)
print("Recalculated fluxes")
print(intrinsic_fluxes)

scale = 10**-18
old_stmag = ["%.2f$\pm$%.2f" %(stmag, err) for stmag, err in zip(stmags, errmags)]
old_fluxes = ["%.2f$\pm$%.2f" %(flux/ scale, err/ scale) for flux, err in zip(fluxes, err)]
# TODO: we should calculate new errors on new mags based on the new fluxes
new_stmag = ["%.2f$\pm$%.2f" %(stmag, err) for stmag, err in zip(new_stmag, new_stmag_err)]
intrinsic_fluxes_str = ["%.2f$\pm$%.3f" %(flux/ scale, err/ scale) for flux, err in zip(intrinsic_fluxes, err_intrinsic_fluxes)]
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

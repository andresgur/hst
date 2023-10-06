# @Author: Andrés Gúrpide <agurpide>
# @Date: 24-02-2023
# @Email:  a.gurpide-lasheras@soton.ac.uk
# @Last modified by:   agurpide
# @Last modified time: 24-02-2023
import os
from astropy.io import fits
import argparse
import astropy.units as u
import logging
import numpy as np
import stsynphot as stsyn
import emcee
import corner
from multiprocessing import Pool
from astropy.time import Time
import glob
from synphot.models import PowerLawFlux1D, ConstFlux1D
from synphot.blackbody import BlackBody1D
from synphot import SourceSpectrum,ReddeningLaw, Observation
from extinction import ccm89, calzetti00, remove
import matplotlib.pyplot as plt
from scipy.optimize import minimize, least_squares
import warnings

def construct_model(model_pars, sp):
    #param_dict = {name:value for name, value in zip(varparamnames, model_pars)}
    [setattr(sp.model, parname + "_1", parvalue) for parname, parvalue in zip(modelvarparnames, model_pars)]
    #for par in param_dict.keys():
    #    param = param_dict[par]
    #    setattr(sp.model, par + "_1", param)
    return sp


def lnpriors(params):
    modelparams, EBV = params[:-1], params[-1]
    # prior on the extincion (always the last parameter)
    return np.log(1.0 / (np.sqrt(2*np.pi) * EBV_calzetti_err))-0.5*(EBV - EBV_calzetti)**2/EBV_calzetti_err**2


def lnlikelihood(params):
    if EBV_calzetti_err > 0:
        model_pars, EBV = params[:-1], params[-1]
    else:
        model_pars = params
        EBV = EBV_calzetti # fixed EBV

    sp_params = construct_model(model_pars, sp_cardelli_ext)
    sp_calzetti_ext = sp_params * calzetti_law.extinction_curve(EBV)# add extra extinction # add extra extinction (if EBV=0 this will be 0)

    if "amplitude" in model.fixed.keys():
        if model.fixed["amplitude"]:
            # if the amplitude is fixed, renormalize based on the new index
            sp_calzetti_ext = sp_calzetti_ext.normalize(rates[index_norm] * u.ct, bps[index_norm], area=stsyn.conf.area) # renormalize to match count rates

    model_rates = [Observation(sp_calzetti_ext, bp).countrate(area=stsyn.conf.area).value for bp in bps] # absorbed model

    return -0.5 * np.sum((((rates - model_rates)/uncertainties)**2))


def lnposterior(params):
    if not np.all(np.logical_and(bounds[:, 0] <= params, params <= bounds[:, 1])):
        return -np.inf
    return lnpriors(params) + lnlikelihood(params)


parser = argparse.ArgumentParser(description='Calculates intrinisc fluxes using convolving a model wih the filters. A Gaussian prior for the extinction can be included.')
parser.add_argument("directories", nargs="+", help="Directories with the HST images and flux information", type=str)
parser.add_argument("-o", "--outdir", nargs="?", help="Output directory. Default is pow_fluxes_", type=str, default="")
parser.add_argument("-E", "--EBV", nargs=2, type=float, help="Galactic (Rv=3.1 Cardelli) + extragalactic E(B-V) (Rv=4.05, Calzetti) value(s) for the extinction correction. Default assumes no extinction",
                    default=[0.0, 0.0])
parser.add_argument("--EBV_err", nargs="?", type=float, help="Error on the extragalactic E(B-V). Default assumes no error",
                    default=0)
parser.add_argument("-n", "--nsims", metavar="N", nargs="?", type=int, help="Number of powerlaw draws for the error calculation. At least 500. Default 600",
                    default=2000)
parser.add_argument("-m", "--model", nargs="?", metavar="model", type=str, choices=["blackbody", "powerlaw", "constant"], help="Model for the derivation of fluxes. Default powerlaw",
                    default="powerlaw")
parser.add_argument("-z", "--redshift", nargs="?", metavar="z", type=float, help="Redshift of the spectrum. Default 0",
                    default=0)
args = parser.parse_args()

working_dir = os.getcwd()
print(working_dir)
os.chdir(working_dir)
dirs = args.directories
print("Analysing the following directories")
print(dirs)
EBV_cardelli = args.EBV[0]
EBV_calzetti = args.EBV[1]
EBV_calzetti_err =args.EBV_err
z = args.redshift
#mwavg == Cardelli+1989 milkiway diffuse Rv=3.1
#Calzetti starbust diffuse xgal
#Rv = 3.1
##extinction_curve = args.curve

outdir = "%s_fluxes_%s_cardelli_EBV%.2f_calzetti_EBV%.2f_%.2f_z%.2e" %(args.model, args.outdir, EBV_cardelli, EBV_calzetti, EBV_calzetti_err, z)

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
#http://stsdas.stsci.edu/astropy_synphot/synphot/reddening.html
Rv=3.1
keyword = "mwavg"
av = Rv * EBV_cardelli
mag_ext_cardelli = ccm89(pivots, av, Rv, unit="aa")
ext_cardelli = ReddeningLaw.from_extinction_model(keyword).extinction_curve(EBV_cardelli)
Rv=4.05
keyword = "xgalsb"
av = Rv * EBV_calzetti
mag_ext_calzetti = calzetti00(pivots, av, Rv, unit="aa")
calzetti_law = ReddeningLaw.from_extinction_model(keyword)
##print("Using extinction curve from %s with Rv = %.2f" % (extinction_curve, Rv))
intrinsic_fluxes = remove(mag_ext_calzetti, remove(mag_ext_cardelli, fluxes)) # derreden the fluxes
##derreden = ReddeningLaw.from_extinction_model(keyword).extinction_curve(-EBV)
total_av = EBV_cardelli * 3.1 + EBV_calzetti * 4.05
print("Total Av: %.2f" % total_av)

# set the norm to the value with the lowest normalization as a starting point
index_norm = np.argmin(uncertainties)
if args.model=="powerlaw":
    # bracket alpha between two sensitive values
    model = PowerLawFlux1D(alpha=2, amplitude=intrinsic_fluxes[index_norm], x_0=pivots[index_norm],
                bounds={"alpha":[0, 8], "amplitude":[1e-26, 1e10]}, fixed={"x_0": True})
elif args.model == "blackbody":
    distance = 4 * u.Mpc
    bolflux = 10**37 / (4 * np.pi * distance.to(u.cm).value**2) * u.Hz# u.erg/u.s /
    bolbounds = [10**20/ (4 * np.pi * distance.to(u.cm).value**2), 10**39 / (4 * np.pi * distance.to(u.cm).value**2)]
    model = BlackBody1D(temperature=40000 * u.K, bolometric_flux=bolflux,
                        bounds={"temperature":[1000, 100000], "bolometric_flux":bolbounds})
    raise Exception("Blackbody model was never implemented due to unit problems with astropy and synphot")

elif args.model == "constant":
    model = ConstFlux1D(amplitude=intrinsic_fluxes[index_norm], bounds={"amplitude":[1e-26, np.inf]})

print(model)

modelparam_dict = {parname:param for parname, param in zip(model.param_names, model.parameters)}

if "amplitude" in modelparam_dict:
    modelparam_dict["amplitude"] *= photflam_units

sp_intrinsic = SourceSpectrum(model.__class__, **modelparam_dict, z=z)
sp_cardelli_ext = sp_intrinsic * ext_cardelli

nsims = args.nsims

if len(fluxes)<2 and args.model=="powerlaw":
    # if there's only 2 fluxes, fix the normalization and only fit for the index of the powerlaw
    if "amplitude" in model.param_names:
        model.amplitude.fixed = True
    elif "bolometric_flux" in model.param_names:
        model.bolometric_flux.fixed = True
    print("Less than two flux data points. Amplitude will be fixed during the fit")

varparams = {parname:param for parname, param in zip(model.param_names, model.parameters) if not model.fixed[parname]}
modelvarparnames = [*varparams.keys()]

ndim = len(varparams)
init_values = np.array([*varparams.values()])
# create bounds list for least_square method
bounds = []
for param_name in modelvarparnames:
    bounds.append((model.bounds[param_name][0], model.bounds[param_name][1]))

    # EBV
if EBV_calzetti_err > 0:
    bounds.append((0, 0.25))
    ndim+= 1 # for EBV
    init_values = np.append(init_values, EBV_calzetti)
    # priors case
    cost_function = lnposterior
else:
    cost_function = lnlikelihood

bounds = np.array(bounds)
print(bounds)
print("Parameter bounds")
# pass the parameters
max_n = 5000
#max_n = 250000
#max_n = 3000
# We'll track how the average autocorrelation time estimate changes
index = 0
autocorr = np.empty(max_n)
every_samples = 400
# This will be useful to testing convergence
old_tau = np.inf
cores = 16
nwalkers = 2 * cores
tau_factor = 65 # https://emcee.readthedocs.io/en/stable/tutorials/autocorr/
print("Running chain for a maximum of %d samples with %d walkers until the chain has a length %dxtau using %d cores" % (max_n, nwalkers, tau_factor, cores))
print("Initial samples\n----------")
initial_samples = np.random.normal(init_values, init_values*0.05, size=(nwalkers, ndim))
print(initial_samples)
with Pool(cores) as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, cost_function, pool=pool)

    # Now we'll sample for up to max_n steps
    for sample in sampler.sample(initial_samples, iterations=max_n, progress=True):
        # Only check convergence every 100 steps
        if sampler.iteration % every_samples:
            continue

        # Compute the autocorrelation time so far
        # Using tol=0 means that we'll always get an estimate even
        # if it isn't trustworthy
        tau = sampler.get_autocorr_time(tol=0)
        autocorr[index] = np.mean(tau)
        index += 1

        # Check convergence
        converged = np.all(tau * tau_factor < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        if converged:
            print("Convergence reached after %d samples!" % sampler.iteration)
            break
        old_tau = tau

acceptance_ratio = sampler.acceptance_fraction

print("Acceptance ratio: (%)")
print(acceptance_ratio)
print("Correlation parameters:")
print(tau)
mean_tau = np.mean(tau)
print("Mean correlation time:")
print(mean_tau)
if not converged:
    warnings.warn("The chains did not converge!")
    # tau will be very large here, so let's reduce the numbers
    thin = int(mean_tau / 4)
    discard = int(mean_tau) * 10 # avoid blowing up if discard is larger than the number of samples, this happens if the fit has not converged

else:
    discard = int(mean_tau * 40)
    if discard > max_n:
        discard = int(mean_tau * 10)
    thin = int(mean_tau / 2)

fig = plt.figure()
autocorr_index = autocorr[:index]
n = every_samples * np.arange(1, index + 1)
plt.plot(n, autocorr_index, "-o")
plt.ylabel("Mean $\\tau$")
plt.xlabel("Number of steps")
plt.savefig("%s/autocorr.png" % outdir, dpi=100)
plt.close(fig)

# plot the entire chain
chain = sampler.get_chain(flat=True)
median_values = np.median(chain, axis=0)
chain_fig, axes = plt.subplots(ndim, sharex=True, gridspec_kw={'hspace': 0.05, 'wspace': 0})
if len(np.atleast_1d(axes))==1:
    axes = [axes]
if EBV_calzetti_err > 0:
    varparamnames = np.append(modelvarparnames, "E(B-V)")
else:
    varparamnames = modelvarparnames

for param, parname, ax, median in zip(chain.T, varparamnames, axes, median_values):
    ax.plot(param, linestyle="None", marker="+", color="black")
    ax.set_ylabel(parname.replace("kernel:", "").replace("log_", ""))
    ax.axhline(y=median)

print("Discarding the first %d samples" % discard)
for ax in axes:
    ax.axvline(discard * nwalkers, ls="--", color="red")

# calculate R stat
samples = sampler.get_chain(discard=discard)

whithin_chain_variances = np.var(samples, axis=0) # this has nwalkers, ndim (one per chain and param)

samples = sampler.get_chain(flat=True, discard=discard)
between_chain_variances = np.var(samples, axis=0)

print("R-stat (values close to 1 indicate convergence)")
print(whithin_chain_variances / between_chain_variances[np.newaxis, :]) # https://stackoverflow.com/questions/7140738/numpy-divide-along-axis

final_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
loglikes = sampler.get_log_prob(discard=discard, thin=thin, flat=True)
axes[-1].set_xlabel("Step Number")
chain_fig.savefig("%s/chain_samples.png" % outdir, bbox_inches="tight", dpi=100)
plt.close(chain_fig)
# get median
median_parameters = np.median(final_samples, axis=0)
distances = np.linalg.norm(final_samples - median_parameters, axis=1)
closest_index = np.argmin(distances)
median_log_likelihood = loglikes[closest_index]

# get best
best_loglikehood = np.argmax(loglikes)
best_params = final_samples[best_loglikehood]
# save contours for each param indicating the maximum
header = ""
outstring = ''
for i, parname in enumerate(varparamnames):
    plt.figure()
    par_vals = final_samples.T[i]
    plt.scatter(par_vals, loglikes)
    plt.scatter(par_vals[best_loglikehood], loglikes[best_loglikehood],
                label="%.2f, L = %.2f" % (par_vals[best_loglikehood], loglikes[best_loglikehood]))
    plt.legend()
    plt.xlabel("%s" % parname)
    plt.ylabel("$L$")
    plt.savefig("%s/%s.png" % (outdir, parname), dpi=100)
    plt.close()
    q_16, q_50, q_84 = corner.quantile(final_samples.T[i], [0.16, 0.5, 0.84]) # your x is q_50
    dx_down, dx_up = q_50-q_16, q_84-q_50
    if parname=="amplitude":
        header += "%s_1e18\t" % parname
        q_50, dx_down, dx_up = q_50 *1e18, dx_down *1e18, dx_up*1e18
    else:
        header += "%s\t" % parname
    outstring += '%.2f$_{-%.2f}^{+%.2f}$ & ' % (q_50, dx_down, dx_up)
# store parameter medians
header += "loglikehood"
outstring += "%.3f" % median_log_likelihood
out_file = open("%s/parameter_medians.dat" % (outdir), "w+")
out_file.write("%s\n%s" % (header, outstring))
out_file.close()

# plot corner plot
corner_fig = corner.corner(final_samples, labels=varparamnames, title_fmt='.1f', #range=ranges,
                           quantiles=[0.16, 0.5, 0.84], show_titles=True,
                           title_kwargs={"fontsize": 18}, max_n_ticks=3, labelpad=0.08,
                           levels=(1 - np.exp(-0.5), 1 - np.exp(-0.5 * 2 ** 2)))
corner_fig.savefig("%s/corner_fig.png" % outdir, dpi=200)
plt.close(corner_fig)

# store samples
outputs = np.vstack((final_samples.T, loglikes))
header = "\t".join(varparamnames) + "\tloglikehood"
np.savetxt("%s/samples.dat" % outdir, outputs.T, delimiter="\t", fmt="%.5e",
          header=header)
# sample fluxes
if "amplitude" in modelparam_dict:
    modelparam_dict["amplitude"] = modelparam_dict["amplitude"].value

sp_intrinsic_nophotflam = SourceSpectrum(model.__class__, **modelparam_dict, z=z)
# set bestift params, skipping EBV
[setattr(sp_intrinsic_nophotflam.model, parname + "_1", par) for par, parname in zip(best_params[:-1], modelvarparnames)]

N = nsims if len(final_samples) > 100 else len(final_samples) - 1
print("Drawing %d samples from the posteriors" %N)

sampled_fluxes = [ ]
# the code below can be improved

fig = plt.figure()
if EBV_calzetti_err > 0:
    # remove EBV from params
    modelpars = final_samples.T[:-1].T
else:
    modelpars = final_samples

for sample_params in modelpars[:N]:
    # add new params, skipping EBV
    sp_params = construct_model(sample_params, sp_intrinsic_nophotflam)
    sampl_fluxes = sp_params(pivots).value
    plt.plot(pivots, sampl_fluxes, alpha=0.4, color="C1")
    sampled_fluxes.append(sampl_fluxes)

plt.xlabel("$\AA$")
plt.ylabel("F$_\lambda$ (erg/s/cm$^2$/$\AA$)")
plt.plot(pivots, sp_intrinsic_nophotflam(pivots), ls="solid", color="C2")
plt.xscale("log")
plt.yscale("log")
plt.savefig("%s/sampled_models.png" % outdir, dpi=100)
plt.close(fig)

plt.figure()
for wavelength, filter_fluxes in zip(pivots, np.array(sampled_fluxes).T):
    plt.figure()
    plt.hist(filter_fluxes)
    plt.axvline(np.mean(filter_fluxes), ls="solid", zorder=10, color="black", label="%.2e+-%.2e" % (np.mean(filter_fluxes), np.std(filter_fluxes)))
    plt.axvline(np.mean(filter_fluxes) + np.std(filter_fluxes), zorder=10, color="black", ls="--")
    plt.xlabel("Intrinsic fluxes ($\lambda$ = %.1f)" % wavelength)
    plt.ylabel("Instances")
    plt.legend()
    plt.savefig("%s/sampled_fluxes_%.1f.png" % (outdir,  wavelength))

intrinsic_fluxes = np.median(sampled_fluxes, axis=0)
err_intrinsic_fluxes = np.std(sampled_fluxes, axis=0)

for pivot, flux, err_intrin in zip(pivots, intrinsic_fluxes, err_intrinsic_fluxes):
    print("Pivot:%.1f --> Derredened flux (1e-18 erg/cm2/AA/s): %.2f$pm$%.3f" %(pivot, flux * 1e18, err_intrin * 1e18))

best_params_dict = {parname:param for parname, param in zip(varparamnames, best_params)}

if "alpha" in model.param_names:
    label = "Best-fit ($\\alpha$ = %.2f)" %(best_params_dict["alpha"])
elif "temperature" in model.param_names:
    label = "Best-fit ($T$ = %.2f+-%.2f)" %(best_params_dict["temperature"])
else:
    label = None

plot_x = np.linspace(min(pivots) - 10, max(pivots) + 10, 100)
plt.figure()
plt.plot(plot_x, sp_intrinsic_nophotflam(plot_x), label=label, ls="--")
plt.ylabel("F$_\lambda$ (erg/s/cm$^2$/$\AA$)")
plt.xlabel("$\AA$")
plt.errorbar(pivots, fluxes, yerr=yerr, fmt="o", label="Observed fluxes")
plt.errorbar(pivots, intrinsic_fluxes, yerr=err_intrinsic_fluxes, fmt="o", label="Extinction-corrected (Av = %.2f) fluxes" % (total_av))
if EBV_calzetti_err > 0:
    EBV_best = best_params[-1]
else:
    EBV_best = EBV_calzetti
ext_calzetti = calzetti_law.extinction_curve(EBV_best)
plt.plot(plot_x, (sp_intrinsic_nophotflam * ext_cardelli * ext_calzetti)(plot_x), label='Av=%.2f' %total_av)
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.savefig("%s/best_fit_model.png" % (outdir), dpi=100)

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
intrinsic_fluxes_str = ["%.2f$\pm$%.2f" %(flux/ scale, err/ scale) for flux, err in zip(intrinsic_fluxes, err_intrinsic_fluxes)]
outputs = np.vstack((dirs, pivots, old_stmag, old_fluxes, new_stmag, intrinsic_fluxes_str))
np.savetxt("%s/results_bayesian.dat" % outdir, outputs.T, header="directory\tpivots\tstmag\tfluxes(%.1e)\tnewstmag\tnewfluxes" % scale,
            delimiter=" & ", fmt="%s")

xspec_file = open("%s/to_xspec.txt" % outdir, "w+")

for pivot_wavelength, flux, flux_error, bp in zip(pivots, intrinsic_fluxes, err_intrinsic_fluxes, bps):
    filter_bandwidth = bp.rectwidth().value # http://svo2.cab.inta-csic.es/theory/fps/index.php?id=HST/ACS_HRC.F555W&&mode=browse&gname=HST&gname2=ACS_HRC#filter
    xspec_file.write("%.1f %.2f %.3e %.3e\n" % (pivot_wavelength - filter_bandwidth / 2, pivot_wavelength + filter_bandwidth / 2, flux, flux_error))
xspec_file.write("#ftflx2xsp infile=%s xunit=angstrom yunit=ergs/cm^2/s/A nspec=1 phafile=hst_%s.fits rspfile=hst_%s.rsp clobber=yes" % ("to_xspec.txt", args.outdir, args.outdir))
xspec_file.close()
print("Outputs stored to %s" % outdir)

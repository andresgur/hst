You are welcome to use this scripts. They allow things such as correcting the astrometry of HST images, performing aperture correction as well as SED fitting with very basic functions (e.g. a constant, a powerlaw). All scripts can be run from the terminal with the -h option to see the list of input parameters and their description.
```
astro_corr.py --> will correct the WCS of the HST image 
```
```
aperture_photometry.py --> To retrieve fluxes and magntiudes from point like sources using aperture photometry
```
```
mast.py --> Downloads observations and creates a latex table with them
```
```
aperture_correction.py --> Allows to determine the PSF aperture correction to infinity from stars in the field
```
```
derive_fluxes.py --> Minimizes the difference between an absorbed powerlaw and the observed count rates
```
```
derive_fluxes_bayesian.py --> Does a similar thing but allows to incorporate priors on the extinction and uses MCMC instead. 
```
```
derive_fluxes_Yang2011.py --> Does a similar thing but following an approach similar to Yang et al. 2011
```

If these scripts are useful to you we would appreciate a citation for the papers these were developed.

## Acknowledging

```
@article{10.1093/mnras/stae1329,
    author = {Gúrpide, A and Castro Segura, N},
    title = {Quasi-isotropic UV emission in the ULX NGC 1313 X–1},
    journal = {Monthly Notices of the Royal Astronomical Society},
    volume = {532},
    number = {2},
    pages = {1459-1485},
    year = {2024},
    month = {06},
    abstract = {A major prediction of most super-Eddington accretion theories is the presence of anisotropic emission from supercritical discs, but the degree of anisotropy and its dependence on energy remain poorly constrained observationally. A key breakthrough allowing to test such predictions was the discovery of high-excitation photoionized nebulae around ultraluminous X-ray sources (ULXs). We present efforts to tackle the degree of anisotropy of the ultraviolet/extreme ultraviolet (UV/EUV) emission in super-Eddington accretion flows by studying the emission-line nebula around the archetypical ULX NGC 1313 X–1. We first take advantage of the extensive wealth of optical/near-UV and X-ray data from Hubble Space Telescope, XMM–Newton, Swift X-ray telescope, and NuSTAR observatories to perform multiband, state-resolved spectroscopy of the source to constrain the spectral energy distribution (SED) along the line of sight. We then compare spatially resolved cloudy predictions using the observed line-of-sight SED with the nebular line ratios to assess whether the nebula ‘sees’ the same SED as observed along the line of sight. We show that to reproduce the line ratios in the surrounding nebula, the photoionizing SED must be a factor of ≈4 dimmer in UV emission than along the line of sight. Such nearly iosotropic UV emission may be attributed to the quasi-spherical emission from the wind photosphere. We also discuss the apparent dichotomy in the observational properties of emission-line nebulae around soft and hard ULXs, and suggest that only differences in mass-transfer rates can account for the EUV/X-ray spectral differences, as opposed to inclination effects. Finally, our multiband spectroscopy suggests that the optical/near-UV emission is not dominated by the companion star.},
    issn = {0035-8711},
    doi = {10.1093/mnras/stae1329},
    url = {https://doi.org/10.1093/mnras/stae1329},
    eprint = {https://academic.oup.com/mnras/article-pdf/532/2/1459/58472123/stae1329.pdf},
}


@article{10.1093/mnras/stae1336,
    author = {Gúrpide, A and Castro Segura, N and Soria, R and Middleton, M},
    title = {Absence of nebular He ii λ4686 constrains the UV emission from the ultraluminous X-ray pulsar NGC 1313 X-2},
    journal = {Monthly Notices of the Royal Astronomical Society},
    volume = {531},
    number = {3},
    pages = {3118-3135},
    year = {2024},
    month = {05},
    abstract = {While much has been learned in recent decades about the X-ray emission of the extragalactic ultraluminous X-ray sources (ULXs), their radiative output in the ultraviolet (UV) band remains poorly constrained. Understanding of the full ULX spectral energy distribution (SED) is imperative to constrain the accretion flow geometry powering them, as well as their radiative power. Here we present constraints on the UV emission of the pulsating ULX (PULX) NGC 1313 X-2 based on the absence of nebular He ii λ4686 emission in its immediate environment. To this end, we first perform multiband spectroscopy of the ULX to derive three realistic extrapolations of the SED into the inaccessible UV, each predicting varying levels of UV luminosity. We then perform photoionization modelling of the bubble nebula and predict the He ii λ4686 fluxes that should have been observed based on each of the derived SEDs. We then compare these predictions with the derived upper limit on He ii λ4686 from the Multi-Unit Spectroscopic Explorer data, which allows us to infer a UV luminosity LUV ≲ 1 × 1039 erg s−1 in the PULX NGC 1313 X-2. Comparing the UV luminosity inferred with other ULXs, our work suggests there may be an intrinsic difference between hard and soft ULXs, either related to different mass-transfer rates and/or the nature of the accretor. However, a statistical sample of ULXs with inferred UV luminosities is needed to fully determine the distinguishing features between hard and soft ULXs. Finally, we discuss ULXs ionizing role in the context of the nebular He ii λ4686 line observed in star-forming metal-poor galaxies.},
    issn = {0035-8711},
    doi = {10.1093/mnras/stae1336},
    url = {https://doi.org/10.1093/mnras/stae1336},
    eprint = {https://academic.oup.com/mnras/article-pdf/531/3/3118/58139949/stae1336.pdf},
}
```

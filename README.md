You are welcome to use this scripts. They allow things such as correcting the astrometry of HST images, performing aperture correction as well as SED fitting with very basic functions (e.g. a constant, a powerlaw). All scripts can be run from the terminal with the -h option to see the list of input parameters and their description.

astro_corr.py --> will correct the WCS of the HST image 


aperture_photometry.py --> To retrieve fluxes and magntiudes from point like sources using aperture photometry


mast.py --> Downloads observations and creates a latex table with them


aperture_correction.py --> Allows to determine the PSF aperture correction to infinity from stars in the field


derive_fluxes.py --> Minimizes the difference between an absorbed powerlaw and the observed count rates


derive_fluxes_bayesian.py --> Does a similar thing but allows to incorporate priors on the extinction and uses MCMC instead. 


derive_fluxes_Yang2011.py --> Does a similar thing but following an approach similar to Yang et al. 2011


If these scripts are useful to you we would appreciate a citation for the papers these were developed.

## Acknowledging

```
@ARTICLE{2024arXiv240514512G,
       author = {{G{\'u}rpide}, Andr{\'e}s and {Castro Segura}, Noel},
        title = "{Quasi-isotropic UV Emission in the ULX NGC\raisebox{-0.5ex}\textasciitilde1313\raisebox{-0.5ex}\textasciitildeX--1}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - High Energy Astrophysical Phenomena, Astrophysics - Astrophysics of Galaxies},
         year = 2024,
        month = may,
          eid = {arXiv:2405.14512},
        pages = {arXiv:2405.14512},
          doi = {10.48550/arXiv.2405.14512},
archivePrefix = {arXiv},
       eprint = {2405.14512},
 primaryClass = {astro-ph.HE},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024arXiv240514512G},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@ARTICLE{2024arXiv240513714G,
       author = {{G{\'u}rpide}, Andr{\'e}s and {Castro Segura}, Noel and {Soria}, Roberto and {Middleton}, Matthew},
        title = "{Absence of nebular He\{\textbackslashsc ii\} $\lambda$4686 constrains the UV emission from the Ultraluminous X-ray pulsar NGC\raisebox{-0.5ex}\textasciitilde1313\raisebox{-0.5ex}\textasciitildeX--2}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - High Energy Astrophysical Phenomena},
         year = 2024,
        month = may,
          eid = {arXiv:2405.13714},
        pages = {arXiv:2405.13714},
          doi = {10.48550/arXiv.2405.13714},
archivePrefix = {arXiv},
       eprint = {2405.13714},
 primaryClass = {astro-ph.HE},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024arXiv240513714G},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

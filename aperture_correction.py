# @Author: Andrés Gúrpide <agurpide>
# @Date:   19-10-2020
# @Email:  a.gurpide-lasheras@soton.ac.uk
# @Last modified by:   agurpide
# @Last modified time: 22-02-2023

import os
from regions import Regions
from photutils.aperture import aperture_photometry, ApertureStats, CircularAperture
from astropy.io import fits
import argparse
from astropy.stats import SigmaClip
import matplotlib.pyplot as plt
from astropy import wcs
import hst_utils as hst_ut
import numpy as np
from photutils.centroids import centroid_2dg
from astropy.visualization import simple_norm



parser = argparse.ArgumentParser(description='Determines the aperture correction by extracting count rates from various stars in the field.')
parser.add_argument("image", help="Image for which the aperture correction is to be determined", nargs=1, type=str)
parser.add_argument("-r", "--regions", type=str, help='Regions enclosing some stars for aperture correction determination', nargs=1)
args = parser.parse_args()

star_regs = Regions.read(args.regions[0], format="ds9")

image_file = args.image[0]

radii = np.arange(2, 15, 1)

if os.path.isfile(image_file):
    hst_hdul = fits.open(image_file)
    pixel_size = hst_hdul[0].header["D001SCAL"] # in arcseconds
    image_data = hst_hdul[1].data
    norm = simple_norm(image_data, 'linear', percent=99)
    hst_wcs = wcs.WCS(hst_hdul[1].header)
    for reg in star_regs:
        star_aperture = hst_ut.region_to_aperture(reg, hst_wcs)
        mask = star_aperture.to_mask()
        mask_image = mask.to_image(image_data.shape)
        #aperture_stat = ApertureStats(image_data, star_aperture, mask=mask.to_image(image_data.shape))

        x, y = centroid_2dg(image_data, mask=~mask_image.astype(bool))
        outputs = "#r\tarcsec\tcounts\n"
        fig, ax = plt.subplots(1 , subplot_kw={'projection': hst_wcs})
        plt.imshow(image_data, norm=norm, interpolation='nearest', cmap="cividis")
        plt.scatter(x, y, marker="x", color="red")
        plt.xlim(x - 50, x + 50)
        plt.ylim(y - 50, y + 50)

        for r in radii:
            aperture = CircularAperture([x, y], r)
            total_counts = aperture_photometry(image_data, aperture)["aperture_sum"]
            outputs += "%.2f\t%.2f\t%.5f\n" % (r, r * pixel_size, total_counts)
            ap_patches = aperture.plot(color='white', lw=1,
                                       label='%d' % r)
        outfile = reg.meta["text"].replace(" ", "").strip()
        plt.savefig(outfile + ".png", dpi=200)
        plt.close(fig)
        file = open(outfile + ".dat", "w+")
        file.write(outputs)
        file.close()

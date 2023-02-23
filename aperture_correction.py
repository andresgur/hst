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
from astropy.time import Time
import matplotlib.pyplot as plt
from astropy import wcs
import hst_utils as hst_ut
import numpy as np
from photutils.centroids import centroid_2dg
from astropy.visualization import simple_norm
import glob

def get_aperture_correction(obs_mode, hst_filter, radius):
    """Parameters
        ---------
        radius: pixel
    """

    if "HRC" in obs_mode and "ACS" in obs_mode:
        table = np.genfromtxt("%s/acs_hrc_ee_table.dat" % args.dir, names=True, skip_header=3,
                                dtype=("S6, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8"))
        corrections = table["%d" % radius]
    elif "UVIS2" in obs_mode:
        table = np.genfromtxt("%s/wfc3uvis2_aper_007_syn.csv"  % args.dir, names=True, delimiter=",",
                      dtype=("S6, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8"
                            ", f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8,"
                            "f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8"), deletechars="")

        corrections = table["APER%.2f" % (radius * 0.04)]
    elif "UVIS1" in obs_mode:
        table = np.genfromtxt("%s/wfc3uvis1_aper_007_syn.csv"  % args.dir, names=True, delimiter=",",
                      dtype=("S6, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8"
                            ", f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8,"
                            "f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8"), deletechars="")

        corrections = table["APER%.2f" % (radius * 0.04)]
    elif "WFC" in obs_mode and "ACS" in obs_mode:
        table = np.genfromtxt("%s/acs_wfc_ee_table.dat" % args.dir, skip_header=3, names=True,
                              dtype=("S10, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8"))
        corrections = table["%d" % radius]
    hst_filters = np.char.decode(table["FILTER"])
    index = np.where(hst_filters == hst_filter)[0]
    aperture_correction = corrections[index]
    if not aperture_correction:
        raise ValueError("Filter %s not found!" % hst_filter)
    print("Aperture correction found for obsmode:%s and r=%d: %.3f" % (obs_mode, radius, aperture_correction))
    return aperture_correction


parser = argparse.ArgumentParser(description='Determines the aperture correction at 10 pixels by extracting count rates from various stars in the field. For HRC and WFC "inifinite" aperture is defined to be 5.5 arcsec')
parser.add_argument("image", help="Image for which the aperture correction is to be determined", nargs=1, type=str)
parser.add_argument("--regions", type=str, help='Regions enclosing some stars for aperture correction determination', nargs=1)
parser.add_argument("-r", "--radius", type=float,
                    help='The radius for which the aperture correction is to be determined', nargs=1)
parser.add_argument("-d", "--dir", help="Directory with the stsci EE files", default="%s/scripts/pythonscripts/hst" % os.getenv("HOME"))
args = parser.parse_args()


star_regs = Regions.read(args.regions[0], format="ds9")
image_file = args.image[0]
default_radius = 10
radii = np.arange(2, 15, 0.5)

if os.path.isfile(image_file):
    hst_hdul = fits.open(image_file)
    date = hst_hdul[0].header["DATE-OBS"]
    obsdate = Time(date).mjd
    pixel_size = hst_hdul[0].header["D001SCAL"] # in arcseconds
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
    print("Obs mode %s" % obsmode)
    hst_filter = keywords[2]
    detector = hst_hdul[0].header["DETECTOR"]
    obs_mode = hst_hdul[1].header["PHOTMODE"]
    exp_time = float(hst_hdul[0].header["EXPTIME"])
    units = hst_hdul[1].header["BUNIT"]
    # divide by the exposure time if needed
    if "/S" not in units:
        print("Units: %s. Applying exposure time correction" % units)
        image_data /= exp_time
    image_data = hst_hdul[1].data
    norm = simple_norm(image_data, 'linear', percent=99)
    hst_wcs = wcs.WCS(hst_hdul[1].header)
    fig_psf, ax_psf = plt.subplots()
    ax_psf.set_ylabel("Normalized Counts")
    ax_psf.set_xlabel("r (pixels)")

    for reg in star_regs:
        star_aperture = hst_ut.region_to_aperture(reg, hst_wcs)
        mask = star_aperture.to_mask()
        #mask_image = mask.to_image(image_data.shape)
        cutout = mask.cutout(image_data)
        err_cutout = mask.cutout(np.sqrt(image_data * exp_time) / exp_time)
        #aperture_stat = ApertureStats(image_data, star_aperture, mask=mask.to_image(image_data.shape))
        print("Determining centroid for region %s" % reg.meta["text"])
        mask_cutout = cutout <= 0
        x_cut, y_cut = centroid_2dg(cutout, error=err_cutout, mask=mask_cutout)
        # convert to image coordinates
        x, y = x_cut + star_aperture.bbox.ixmin, y_cut + star_aperture.bbox.iymin
        outputs = "#r\tarcsec\tcounts\terr\n"
        fig, ax = plt.subplots(1 , subplot_kw={'projection': hst_wcs})
        ax.imshow(image_data, norm=norm, interpolation='nearest', cmap="cividis")
        ax.scatter(x, y, marker="x", color="red", lw=0.5)
        plt.xlim(x - 50, x + 50)
        plt.ylim(y - 50, y + 50)

        line = None
        apertures = [CircularAperture([x, y], r=r) for r in radii]
        apt_phots = aperture_photometry(image_data * exp_time, apertures)
        total_counts = np.array([apt_phots["aperture_sum_%d" % i][0] for i, aperture in enumerate(apertures)])

        [aperture.plot(color="white", lw=0.5) for aperture in apertures]
        index = np.argmin(np.abs(radii - default_radius))
        # plot the target aperture
        apertures[index].plot(color="magenta", lw=0.5, label="r=%d pixels" % default_radius)
        ax_psf.errorbar(radii, total_counts / max(total_counts), yerr=np.sqrt(total_counts) / max(total_counts), label=reg.meta["text"])

        # save results
        outputs = np.array([radii, radii*pixel_size, total_counts, np.sqrt(total_counts)])
        outfile = reg.meta["text"].replace(" ", "").strip()
        np.savetxt("apt_corr_" +  outfile + ".dat", outputs.T, header="#r\tarcsec\tcounts\terr",
                   delimiter="\t", fmt="%.2f\t%.2f\t%.1f\t%.1f")

        ax.legend()

        fig.savefig(outfile + ".png", dpi=200)
        plt.close(fig)

    ax_psf.legend()
    fig_psf.savefig("psf.png", dpi=200)
    out_apt_files = glob.glob("apt_corr*.dat")

    radii = np.genfromtxt(out_apt_files[0], names=True)["r"]

    target_radius = np.argmin(np.abs(radii - args.radius[0]))

    default_radius = 10

    large_radius = np.argmin(np.abs(radii - default_radius))

    samples = len(out_apt_files)

    counts_at_target = np.array([np.genfromtxt("apt_corr_" + reg.meta["text"].replace(" ", "").strip() + ".dat" , names=True)["counts"][target_radius] for reg in star_regs])
    err_counts_at_target = np.sum(np.array([np.genfromtxt("apt_corr_" + reg.meta["text"].replace(" ", "").strip() + ".dat" , names=True)["err"][target_radius]  for reg in star_regs]))

    counts_at_large = np.array([np.genfromtxt("apt_corr_" + reg.meta["text"].replace(" ", "").strip() + ".dat", names=True)["counts"][large_radius] for reg in star_regs])
    err_counts_at_large = np.sum(np.array([np.genfromtxt("apt_corr_" + reg.meta["text"].replace(" ", "").strip() + ".dat" , names=True)["err"][large_radius] for reg in star_regs]))

    corr_factors = counts_at_target / counts_at_large
    err_corr_factors = np.sqrt((err_counts_at_target / counts_at_large)**2 + (err_counts_at_large * counts_at_target / counts_at_large**2)**2)
    mean_corr_factor =  np.mean(corr_factors)

    err_corr_factor = np.sqrt(np.sum(err_corr_factors**2)) / samples
    to_inf = get_aperture_correction(obs_mode, hst_filter, default_radius)

    outfile = open("aperture_correction.dat", "w+")
    outfile.write("#mean\terr\tr\n%.4f\t%.5f\t%.1f\n" % (mean_corr_factor, err_corr_factor,
                  args.radius[0]))
    outfile.write("%.4f\t%.4f\tinf\n" % ((mean_corr_factor * to_inf), (err_corr_factor * to_inf))) # we multiply to get the total "dividing" factor to apply to the final flux
    to_input_radius = get_aperture_correction(obs_mode, hst_filter, args.radius[0])
    outfile.write("#%.3f\t_\t%d\n" % (to_input_radius, args.radius[0]))

    [outfile.write("#%.3f\t%s\n" % (corr_factor, reg.meta["text"])) for corr_factor, reg in zip(corr_factors, star_regs)]
    outfile.close()

    print("Mean aperture correction factor from %d to %d pixels: %.2f+-%.2f" % (args.radius[0],
         default_radius, mean_corr_factor, err_corr_factor))

    print("Run 'python %s/scripts/pythonscripts/hst/aperture_photometry.py -s ngc1313x1.reg -b background.reg -a %.4f --uncertainty %.4f %s'" % (os.getenv("HOME"),(mean_corr_factor * to_inf), (err_corr_factor * to_inf), image_file))
else:
    print("File %s not found" % image_file)

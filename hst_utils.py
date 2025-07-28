# @Author: Andrés Gúrpide <agurpide>
# @Date:   11-12-2021
# @Email:  agurpidelash@irap.omp.eu
# @Last modified by:   agurpide
# @Last modified time: 21-03-2022

from photutils.aperture import CircularAperture, CircularAnnulus, EllipticalAperture, SkyCircularAperture, SkyCircularAnnulus, SkyEllipticalAperture
from regions import Regions
import astropy.units as u


def get_image_filter(header):
    if "FILTER" in header:
        return header["FILTER"]
    elif "FILTNAM1" in header:
        return header["FILTNAM1"]
    else:
        return header["FILTER1"]



def region_to_aperture(region, wcs=None):
    """Convert region object to photutils.aperture.aperture_photometry object. The wcs object is needed only if the input regions are in sky coordinates.
    Parameters
    ----------
    region: regions.Region
        Output of read_ds9 method or str
    wcs: astropy.wcs.WCS
        A world coordinate system if the region in sky coordinates IS needed to convert it to pixels.
    """

    if type(region)==str:
        region= Regions.read(region, format="ds9")
    print(region)
    region_type = type(region).__name__
    if "Pixel" in region_type:
        source_center = (region.center.x, region.center.y)
        if region_type == 'CirclePixelRegion':
            return CircularAperture(source_center, r=region.radius)
        elif region_type == "CircleAnnulusPixelRegion":
            return CircularAnnulus(source_center, r_in=region.inner_radius, r_out=region.outer_radius)
        elif region_type == "EllipsePixelRegion":
            # to be tested
            return EllipticalAperture(source_center, a=region.width / 2, b=region.height / 2, theta=region.angle.to(u.rad).value)
        elif region_type == "EllipsePixelRegion":
            # to be tested
            return EllipticalAperture(source_center, a=region.width / 2, b=region.height / 2, theta=region.angle.to(u.rad).value)
    elif "Sky" in region_type:
        if wcs is None:
            print("Error, cannot obtain aperture without a wcs.")
            return None
        center = region.center.fk5
        if region_type == "CircleSkyRegion":
            return SkyCircularAperture(center, r=region.radius).to_pixel(wcs)
        elif region_type == "EllipseSkyRegion":
            return SkyEllipticalAperture(center, a=region.width / 2, b=region.height / 2, theta=region.angle).to_pixel(wcs)
        elif region_type == "CircleAnnulusSkyRegion":
            return SkyCircularAnnulus(center, r_in=region.inner_radius, r_out=region.outer_radius).to_pixel(wcs)
    else:
        print("Error region not implemented")
        return None

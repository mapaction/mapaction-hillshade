import itertools
import logging
import os

import numpy as np
import rasterio
from osgeo import gdal
from scipy import ndimage
from scipy.signal import fftconvolve
from skimage.measure import label

logger = logging.getLogger(__name__)
# Hill shading is determine by a combination of the slope (steepness) and the
# slope direction -> using http://people.csail.mit.edu/bkph/papers/Hill-Shading.pdf


def set_working_directory(path_to_output_file):
    """
    Basically make a subdirectory where the output file is located with the
    working files needed to construct a given hillshade.
    :param path_to_output_file: path to the output file
    :return:
    """
    working_dir = os.path.dirname(path_to_output_file) + os.sep + "hillshade_working"
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
    return working_dir


def set_temporary_uri(original_uri, working_dir, calculation, extra_suffix=None):
    """
    Sets the temporary working uri for a specific task
    :param working_dir:
    :param original_file:
    :return:
    """
    original_basename = os.path.basename(original_uri)

    if extra_suffix is None:
        extra_suffix = ""
    suffix = ("_" + "_".join([calculation, extra_suffix])).rstrip("_")
    temp_uri = working_dir + os.sep + suffix.join(os.path.splitext(original_basename))
    return temp_uri


def get_neighbouring_pixels(x=0, y=0, xlim=(-1, 1), ylim=(-1, 1), radius=1):
    """
    Yields coordinates of pixels in square surrounding this one, exclude edges.
    By default, outputs relative pixel coordinate positions around centre (0,0).

    Args (default +/- 1 pixel around <0,0>):
        x       - x-coordinate of pixel
        y       - y-coordinate of pixel
        xlim    - (minimum, maximum) tuple of x-coordinate pixels
        ylim    - (minimum, maximum) tuple of y-coordinate pixels
        radius  - number of pixels above/below (x,y) that are returned
    """
    intradius = int(np.ceil(radius))
    for dx, dy in itertools.product(
        (np.arange(-intradius, intradius + 1)), (np.arange(-intradius, intradius + 1))
    ):
        if (dx == 0) and (dy == 0):  # exclude the coordinate of pixel itself
            continue
        # if surrounding pixel is within array limits, return it :)
        if (min(xlim) <= (x + dx) <= max(xlim)) & (min(ylim) <= (y + dy) <= max(ylim)):
            yield tuple([int(x + dx), int(y + dy)])


def patch_raster_nodata_within_boundary(
    input_raster_uri, output_raster_uri=None, patch_resampling="nearest"
):
    """
    TODO: Can skimage do this better? Hav not researched...
    Fill in nodata values of raster with some value calculated out of raster
    pixel neighbours
    Args:
        input_raster_uri: original raster with nodata "holes"
        output_raster_uri: output raster with input nodata holes patched
                    if this is None, the input raster is changed in place)
        patch_resampling: choice of 'nearest' 'min' and 'max'

    Returns:
        None
    """
    with rasterio.open(input_raster_uri, "r") as input_raster:
        input_data = input_raster.read(1)
        input_meta = input_raster.meta
        input_meta.update(compress="lzw")

    good_data = np.where(input_data != input_meta["nodata"])

    if patch_resampling in ["min", "max"]:

        if patch_resampling == "min":
            patch_value = np.min(input_data[good_data])
        elif patch_resampling == "max":
            patch_value = np.max(input_data[good_data])

        filled = np.where(input_data != input_meta["nodata"], input_data, patch_value)

    elif patch_resampling == "nearest":

        bad = np.where(input_data == input_meta["nodata"])
        if len(bad[0]) == 0:
            logger.debug(
                f"No bad pixels to fix in {os.path.basename(input_raster_uri)}"
            )
            return None
        badval = list(zip(bad[0], bad[1]))
        xsize = input_data.shape[0]
        ysize = input_data.shape[1]
        filled = input_data.copy()
        logging.debug(
            "%d bad values to fix in %s"
            % (len(bad[0]), os.path.basename(input_raster_uri))
        )
        # on first loop, all nodata values are candidates for in-filling
        # during first loop, this is reduced to those nodata pixels that
        # have a neighbouring pixel with valid data.
        # Each subsequent loop are then filled with the neighbours of those
        # pixels that have no values until there are none left.
        while len(badval) > 0:
            logging.debug(
                "...searching %d of %d values for error"
                % (len(badval), input_data.size)
            )
            tofix = []
            nextbadval = []
            for bidx in np.arange(len(badval)):
                if filled[badval[bidx]] != input_meta["nodata"]:
                    continue
                goodnbor = np.array(
                    [
                        tuple(coord)
                        for coord in get_neighbouring_pixels(
                            x=badval[bidx][0],
                            y=badval[bidx][1],
                            xlim=(0, xsize - 1),
                            ylim=(0, ysize - 1),
                            radius=1,
                        )
                        if filled[tuple(coord)] != input_meta["nodata"]
                    ]
                )
                if len(goodnbor) > 0:
                    goodval = [filled[tuple(coord)] for coord in goodnbor]
                    goodval.sort()
                    tofix.append([tuple(badval[bidx]), goodval[int(len(goodval) / 2)]])

                    poornbor = np.array(
                        [
                            tuple(coord)
                            for coord in get_neighbouring_pixels(
                                x=badval[bidx][0],
                                y=badval[bidx][1],
                                xlim=(0, xsize - 1),
                                ylim=(0, ysize - 1),
                                radius=1,
                            )
                            if filled[tuple(coord)] == input_meta["nodata"]
                        ]
                    )
                    for poorpx in poornbor:
                        nextbadval.append(tuple(poorpx))

            logging.debug("... fixing %d values" % (len(tofix)))
            for fix in tofix:
                filled[fix[0]] = fix[1]
            badval = list(set(nextbadval))

    if output_raster_uri is None:
        output_raster_uri = input_raster_uri

    input_meta.update(dtype="float32")
    with rasterio.open(output_raster_uri, "w", **input_meta) as out_raster:
        out_raster.write_band(1, filled.astype(input_meta["dtype"]))


def calculate_curvature_3by3(arr, xy_resolution=1):
    """
    This works as a bilinear interpolation of the second-order terms of
    a 3x3 surface described by a fourth-order polynomial surface (see
    http://help.arcgis.com/en/arcgisdesktop/10.0/help/index.html#//00q90000000t000000)
    akin to an osculating circle
    :param arr: 3x3 array of either slope steepness or aspect (in radians)
    :param xy_resolution: resolution of the raster in metres.
    :return:
    """
    horizontal = ((arr[1, 2] + arr[1, 0]) / 2 - arr[1, 1]) / (xy_resolution ** 2)
    vertical = ((arr[2, 1] + arr[0, 1]) / 2 - arr[1, 1]) / (xy_resolution ** 2)

    curvature = -2 * (horizontal + vertical) * 100

    return curvature


def calculate_basic_hillshade(
    aspect_deg_array, slope_deg_array, altitude_deg=45.0, azimuth_deg=315.0
):
    """
    Calculates a basic hillshade based on equation 2 of
    http://myweb.liu.edu/~pkennell/reprints/peer-reviewed/geomorph_08.pdf
    :param aspect_deg_array:
    :param slope_deg_array:
    :param altitude_deg:
    :param azimuth_deg:
    :return:
    """

    # 1) altitude in degrees -> zenith angle in radians
    zenith_rad = (90.0 - altitude_deg) * np.pi / 180.0

    # 2) azimuth in radians
    azimuth_rad = ((360.0 - azimuth_deg + 90) % 360) * np.pi / 180.0

    # 3) aspect in radians
    aspect_rad_array = aspect_deg_array * np.pi / 180.0

    # 4) slope in radians
    slope_rad_array = slope_deg_array * np.pi / 180.0

    hillshade = 255.0 * (
        (np.cos(zenith_rad) * np.cos(slope_rad_array))
        + (
            np.sin(zenith_rad)
            * np.sin(slope_rad_array)
            * np.cos(azimuth_rad - aspect_rad_array)
        )
    )
    return hillshade


def get_slope_steepness_uri(input_dem_uri, working_dir, extra_suffix=""):

    temp_slope_uri = set_temporary_uri(
        input_dem_uri, working_dir, "slopeangle", extra_suffix=extra_suffix
    )
    return temp_slope_uri


def get_slope_steepness_deg(
    input_dem_uri, working_dir, overwrite_temp_files=False, extra_suffix=""
):

    # Set up file naming conventions
    temp_slope_uri = get_slope_steepness_uri(
        input_dem_uri, working_dir, extra_suffix=extra_suffix
    )

    # See if the data has already been saved to default temp sub-folder
    if not os.path.exists(temp_slope_uri) or overwrite_temp_files:
        logger.info("Creating slope raster")
        gdal.DEMProcessing(temp_slope_uri, input_dem_uri, "slope")
    with rasterio.open(temp_slope_uri, "r") as tempsl:
        slope_angle = tempsl.read(1)

    return slope_angle


def get_slope_aspect_uri(input_dem_uri, working_dir, extra_suffix=""):

    temp_aspect_uri = set_temporary_uri(
        input_dem_uri, working_dir, "aspect", extra_suffix=extra_suffix
    )
    return temp_aspect_uri


def get_slope_aspect_deg(
    input_dem_uri, working_dir, overwrite_temp_files=False, extra_suffix=""
):

    # Set up file naming conventions
    temp_aspect_uri = get_slope_aspect_uri(
        input_dem_uri, working_dir, extra_suffix=extra_suffix
    )

    # See if the data has already been saved to default temp sub-folder
    if not os.path.exists(temp_aspect_uri) or overwrite_temp_files:
        logger.info("Creating aspect raster")
        gdal.DEMProcessing(temp_aspect_uri, input_dem_uri, "aspect")
    with rasterio.open(temp_aspect_uri, "r") as tempas:
        aspect = tempas.read(1)

    return aspect


def get_basic_hillshade(
    output_hillshade_uri,
    input_dem_uri,
    altitude_deg=45.0,
    azimuth_deg=315.0,
    overwrite_temp_files=True,
):

    # Set up file naming conventions
    working_dir = set_working_directory(output_hillshade_uri)

    # Get slope angle and aspect
    aspect = get_slope_aspect_deg(
        input_dem_uri, working_dir, overwrite_temp_files=overwrite_temp_files
    )
    slope_angle = get_slope_steepness_deg(
        input_dem_uri, working_dir, overwrite_temp_files=overwrite_temp_files
    )

    # Calculate hillshade
    logger.info("Creating basic hillshade raster")
    hillshade = calculate_basic_hillshade(
        aspect, slope_angle, altitude_deg=altitude_deg, azimuth_deg=azimuth_deg
    )

    with rasterio.open(input_dem_uri, "r") as input_r:
        inputmeta = input_r.meta

    logger.info(f"Saving basic hillshade raster: {output_hillshade_uri}")
    with rasterio.open(output_hillshade_uri, "w", **inputmeta) as outraster:
        outraster.write_band(1, hillshade.astype(inputmeta["dtype"]))


def calculate_slope_curvature(slope_angle, xy_resolution=1):
    """
    SLOW: ABANDONED
    Nice explanation of curvature here:
    https://www.esri.com/arcgis-blog/products/product/imagery/understanding-curvature-rasters/
    This function calculates the profile curvature of slopes. i.e., the rate
    of change of slope magnitude. THIS IS SLOW
    :param slope_angle: the steepness/aspect of the slope
    :param xy_resolution: the length of one dimension of a given raster cell
    :return:
    """
    slope_curvature = slope_angle.copy()
    slope_curvature[:] = 0.0
    extra_keywords = {"xy_resolution": xy_resolution}

    ndimage.generic_filter(
        slope_angle,
        calculate_curvature_3by3,
        size=(3, 3),
        output=slope_curvature,
        mode="nearest",
        extra_keywords=extra_keywords,
    )

    return slope_curvature


def get_profile_curvature(input_dem_uri, working_dir, overwrite_temp_files=False):
    """
    SLOW: ABANDONED
    """

    # Set up file naming conventions
    temp_profcurve_uri = set_temporary_uri(
        input_dem_uri, working_dir, "profilecurvature"
    )

    with rasterio.open(input_dem_uri, "r") as input_r:
        inputmeta = input_r.meta
    xy_resolution = abs(inputmeta["transform"][0])

    # See if the data has already been saved to default temp sub-folder
    if not os.path.exists(temp_profcurve_uri):
        logger.info("Creating profile curvature raster")
        slope_angle = get_slope_steepness_deg(
            input_dem_uri, working_dir, overwrite_temp_files=overwrite_temp_files
        )
        profile_curvature = calculate_slope_curvature(
            slope_angle, xy_resolution=xy_resolution
        )
    else:
        with rasterio.open(temp_profcurve_uri, "r") as tempas:
            profile_curvature = tempas.read(1)

    if overwrite_temp_files or not os.path.exists(temp_profcurve_uri):
        logger.info(f"Saving profile curvature raster: {temp_profcurve_uri}")
        with rasterio.open(temp_profcurve_uri, "w", **inputmeta) as handle:
            handle.write_band(1, profile_curvature)

    return profile_curvature


def get_planform_curvature(input_dem_uri, working_dir, overwrite_temp_files=False):
    """
    SLOW: ABANDONED
    """

    # Set up file naming conventions
    temp_plancurve_uri = set_temporary_uri(
        input_dem_uri, working_dir, "planformcurvature"
    )

    with rasterio.open(input_dem_uri, "r") as input_r:
        inputmeta = input_r.meta
    xy_resolution = abs(inputmeta["transform"][0])

    # See if the data has already been saved to default temp sub-folder
    if not os.path.exists(temp_plancurve_uri):
        logger.info("Creating planform curvature raster")
        slope_aspect = get_slope_aspect_deg(
            input_dem_uri, working_dir, overwrite_temp_files=overwrite_temp_files
        )
        planform_curvature = calculate_slope_curvature(
            slope_aspect, xy_resolution=xy_resolution
        )
    else:
        with rasterio.open(temp_plancurve_uri, "r") as tempas:
            planform_curvature = tempas.read(1)

    if overwrite_temp_files or not os.path.exists(temp_plancurve_uri):
        logger.info(f"Saving planform curvature raster: {temp_plancurve_uri}")
        with rasterio.open(temp_plancurve_uri, "w", **inputmeta) as handle:
            handle.write_band(1, planform_curvature)

    return planform_curvature


def get_curvature_hillshade(
    output_curveshade_uri,
    input_dem_uri,
    altitude_deg=45.0,
    azimuth_deg=315.0,
    overwrite_temp_files=False,
):
    """
    WOULDN'T USE THIS -> scipy.ndimage is slow for large rasters. Based on
    http://myweb.liu.edu/~pkennell/reprints/peer-reviewed/geomorph_08.pdf
    and convolving curvature with basic hillshade.
    """

    # Set up file naming conventions
    working_dir = set_working_directory(output_curveshade_uri)

    # Get slope angle and aspect
    aspect = get_slope_aspect_deg(
        input_dem_uri, working_dir, overwrite_temp_files=overwrite_temp_files
    )
    slope_angle = get_slope_steepness_deg(
        input_dem_uri, working_dir, overwrite_temp_files=overwrite_temp_files
    )

    # Calculate hillshade
    hillshade = calculate_basic_hillshade(
        aspect, slope_angle, altitude_deg=altitude_deg, azimuth_deg=azimuth_deg
    )
    # Calculate curvature
    planform_curv = get_planform_curvature(
        input_dem_uri, working_dir, overwrite_temp_files=overwrite_temp_files
    )
    profile_curv = get_profile_curvature(
        input_dem_uri, working_dir, overwrite_temp_files=overwrite_temp_files
    )

    # rescale curvature to 0 - 255
    renorm_planform = 255.0 * abs(planform_curv) / max(abs(planform_curv))
    renorm_profile = 255.0 * abs(profile_curv) / max(abs(profile_curv))

    hillshade_curv = 0.2 * (renorm_planform + renorm_profile) + 0.6 * hillshade

    with rasterio.open(input_dem_uri, "r") as input_r:
        inputmeta = input_r.meta
    with rasterio.open(output_curveshade_uri, "w", **inputmeta) as outraster:
        outraster.write_band(1, hillshade_curv)


def gaussian_blur(input_raster, pixel_radius=10):
    """
    Gaussian blur using FFT convolution algorithm in
    From https://gis.stackexchange.com/posts/10467/revisions
    :param input_raster:
    :param pixel_radius:
    :return:
    """
    # expand input_raster to fit edge of kernel
    padded_array = np.pad(input_raster, pixel_radius, "symmetric")

    # build kernel
    x, y = np.mgrid[-pixel_radius : pixel_radius + 1, -pixel_radius : pixel_radius + 1]
    g = np.exp(-(x ** 2 / float(pixel_radius) + y ** 2 / float(pixel_radius)))
    g = (g / g.sum()).astype(input_raster.dtype)

    # do the Gaussian blur
    return fftconvolve(padded_array, g, mode="valid")


def get_gaussian_blur(
    input_dem_uri, working_dir, blur_radius, overwrite_temp_files=False
):
    """
    Wrapper function for gaussian_blur() saving data to a file in a regular
    format.
    :param input_dem_uri:
    :param working_dir:
    :param blur_radius:
    :param overwrite_temp_files:
    :return:
    """
    # Set up file naming conventions
    temp_blurred_uri = set_temporary_uri(
        input_dem_uri, working_dir, "blur", extra_suffix=str(blur_radius) + "px"
    )

    # See if the data has already been saved to default temp sub-folder
    if not os.path.exists(temp_blurred_uri) or overwrite_temp_files:
        logger.info(
            "Creating %d pixel blurred raster of %s" % (blur_radius, input_dem_uri)
        )
        with rasterio.open(input_dem_uri, "r") as tempdem:
            demmeta = tempdem.meta
            demdata = tempdem.read(1)
        blurred_raster = gaussian_blur(demdata, pixel_radius=blur_radius)
        with rasterio.open(temp_blurred_uri, "w", **demmeta) as tempblur:
            tempblur.write_band(1, blurred_raster.astype(demmeta["dtype"]))
    else:
        with rasterio.open(temp_blurred_uri, "r") as tempblur:
            blurred_raster = tempblur.read(1)

    return blurred_raster


def calculate_multiscale_hillshade(
    output_mshillshade_uri,
    downloaded_dem_uri,
    altitude_deg=45.0,
    azimuth_deg=315.0,
    delete_temp_files=True,
):
    """
    Saw this youtube: https://youtu.be/pFDLFldNj9c on how to hack ambient
    occlusion on hillshades and it looks boss. However, I'm sticking to
    grayscale so let's have a party and see if this works.
    The recipe seems to be:
        1) sandwich blurred DEMs to accentuate large scale features
        2) use blurred slopes as a fake directionless curvature indication
        3) calculate meta-hillshade using blurred slopes as DEM
        4) darken areas at lower altitude
    Adiitionally, I:
        5) apply mask of sea and other nodata values to output
        6) save and tidy up intermediary files
    :param output_mshillshade_uri: output file path
    :param downloaded_dem_uri: input dem file path
    :param altitude_deg: altitude in degrees
    :param azimuth_deg: sun azimuth in degrees
    :return:
    """
    pixel_blur = [10, 20, 50]

    # fetch the downloaded data to
    with rasterio.open(downloaded_dem_uri, "r") as indem:
        dl_data = indem.read(1)
        dl_meta = indem.meta
    dl_goodmask = np.where(dl_data != dl_meta["nodata"])
    dl_badmask = np.where(dl_data == dl_meta["nodata"])

    # Set up file naming conventions
    working_dir = set_working_directory(output_mshillshade_uri)
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
    keep_file = []
    if delete_temp_files:
        for rt, dr, fls in os.walk(working_dir):
            for fl in fls:
                keep_file.append(os.path.join(rt, fl))

    # patch the downloaded DEM (so nodata values don't corrupt blurred dems)
    working_dem_uri = working_dir + os.sep + os.path.basename(downloaded_dem_uri)
    if not os.path.exists(working_dem_uri):
        patch_raster_nodata_within_boundary(
            downloaded_dem_uri,
            output_raster_uri=working_dem_uri,
            patch_resampling="nearest",
        )

    # read in the patched (working dem)
    with rasterio.open(working_dem_uri, "r") as indem:
        working_dem = indem.read(1)
        working_meta = indem.meta

    # set up arrays to hold reused data
    base_dem = []
    steepness = []
    steepness_uri = {}
    hillshade = []

    # set up the arrays with unblurred DEM values
    base_dem.append(working_dem)
    logger.info("Calculating aspect")
    this_aspect = get_slope_aspect_deg(working_dem_uri, working_dir)
    logger.info("Calculating steepness")
    this_steepness = get_slope_steepness_deg(working_dem_uri, working_dir)
    steepness.append(this_steepness)
    steepness_uri[0] = get_slope_steepness_uri(working_dem_uri, working_dir)
    logger.info("Calculating hillshade")
    hillshade.append(
        calculate_basic_hillshade(
            this_aspect,
            this_steepness,
            altitude_deg=altitude_deg,
            azimuth_deg=azimuth_deg,
        )
    )
    # save0_hillshade = hillshade[0].copy()

    # Calculate hillshades for blurred DEMs
    for pxbl in pixel_blur:
        logger.info("Applying %d pixel radius blur to DEM" % pxbl)
        base_dem.append(get_gaussian_blur(working_dem_uri, working_dir, pxbl))
        blurred_dem_uri = set_temporary_uri(
            working_dem_uri, working_dir, "blur", extra_suffix=str(pxbl) + "px"
        )
        logger.info("\tCalculating aspect with %d pixel radius blur" % pxbl)
        this_aspect = get_slope_aspect_deg(blurred_dem_uri, working_dir)
        logger.info("\tCalculating steepness with %d pixel radius blur" % pxbl)
        this_steepness = get_slope_steepness_deg(blurred_dem_uri, working_dir)
        steepness.append(this_steepness)
        steepness_uri[pxbl] = get_slope_steepness_uri(blurred_dem_uri, working_dir)
        logger.info("\tCalculating hillshade with %d pixel radius blur" % pxbl)
        hillshade.append(
            calculate_basic_hillshade(
                this_aspect,
                this_steepness,
                altitude_deg=altitude_deg,
                azimuth_deg=azimuth_deg,
            )
        )

    # turn lists into numpy arrays
    base_dem = np.array(base_dem)
    steepness = np.array(steepness)
    hillshade = np.array(hillshade)

    # average the hillshades up, rescale, and then brighten them...
    logger.info("Averaging multiple scale hillshade")
    hillshade_average = np.sum(hillshade, axis=0) / len(hillshade)
    logger.info("Brightening multiple scale hillshade")
    ha_scale = np.max(hillshade_average) - np.min(hillshade_average)
    hillshade_average = (
        255.0 * (hillshade_average - np.min(hillshade_average)) / ha_scale
    )
    multiscale_hillshade = 85.0 + (
        170.0 * hillshade_average / np.max(hillshade_average)
    )
    # save1_hillshade = multiscale_hillshade.copy()

    # take hi res hillshade and use the bright end to brighten up/ white out
    # highlighted areas
    logger.info("Adding highlights to multiple scale hillshade")
    mean_hillshade = np.mean(hillshade[0, :, :])
    highlight = np.where(
        hillshade[0, :, :] > mean_hillshade,
        hillshade[0, :, :] - mean_hillshade + multiscale_hillshade,
        multiscale_hillshade,
    )
    multiscale_hillshade = (
        255.0
        * (highlight - np.min(highlight))
        / (np.max(highlight) - np.min(highlight))
    )
    # save2_hillshade = multiscale_hillshade.copy()

    # 2) in the demo above it goes from black (steepest) to transparent black
    # (shallowest) with the slopes, need to invert values (want steepest areas
    # to be darkest) then average them together - the directionless gives a fake
    # ambient occlusion effect
    logger.info("Applying multiple scale steepness shading to multiple scale hillshade")
    slope_average = np.sum(steepness, axis=0) / len(steepness)
    stp_bad_mask_arr = np.ones(multiscale_hillshade.shape)
    stp_bad_mask_arr[dl_badmask] = 0.0
    for sidx in range(len(steepness)):
        bad = np.where(steepness[sidx, :, :] == working_meta["nodata"])
        stp_bad_mask_arr[bad] = 0.0

    stp_good_mask = np.where(stp_bad_mask_arr == 1)
    stp_bad_mask = np.where(stp_bad_mask_arr == 0)
    weighted_slope_average = (
        255.0 * slope_average / np.max(slope_average[stp_good_mask])
    )
    multiscale_hillshade[stp_good_mask] = (
        multiscale_hillshade[stp_good_mask] - weighted_slope_average[stp_good_mask]
    )
    multiscale_hillshade = (
        255.0
        * (multiscale_hillshade - np.min(multiscale_hillshade))
        / (np.max(multiscale_hillshade) - np.min(multiscale_hillshade))
    )
    # save3_hillshade = multiscale_hillshade.copy()

    # 3) now take hillshade of the _slope_ at all resolutions.
    # so to do this we need to treat slope as a DEM and loop to create fake
    # fake slope aspect and fake slope steepness
    # N.B. only want the high end and low end to accent our final hillshade
    logger.info("Making meta-hillshade from slope steepness at multiple scales")
    meta_slope_shade = []
    temp_slope_hillshade_uri = working_dir + os.sep + "tempmetahillshade.tif"
    meta_bad_mask_arr = np.ones(multiscale_hillshade.shape)
    meta_bad_mask_arr[dl_badmask] = 0.0
    for sidx in steepness_uri.keys():
        meta_dem_uri = steepness_uri[sidx]
        # I don't want to save all these items so just have them overwrite
        logger.info("\t\tMaking meta-hillshade at %d pixel blur radius" % sidx)
        get_basic_hillshade(
            temp_slope_hillshade_uri,
            meta_dem_uri,
            altitude_deg=altitude_deg,
            azimuth_deg=azimuth_deg,
            overwrite_temp_files=True,
        )
        with rasterio.open(temp_slope_hillshade_uri, "r") as tempslopeshade:
            meta_slope_shade.append(tempslopeshade.read(1))
        meta_bad_mask_arr = np.where(
            meta_slope_shade[-1] == working_meta["nodata"], 0.0, meta_bad_mask_arr
        )
    logger.info("\tAveraging meta-hillshade from slope steepness")
    meta_slope_shade = np.array(meta_slope_shade)
    average_slope_shade = np.sum(meta_slope_shade, axis=0) / len(meta_slope_shade)
    # rescale the slope shade to 255.
    avsl_good_mask = np.where(meta_bad_mask_arr == 1)
    avsl_badmask = np.where(meta_bad_mask_arr == 0)
    average_slope_shade = (
        255.0
        * (average_slope_shade - np.min(average_slope_shade[avsl_good_mask]))
        / (
            np.max(average_slope_shade[avsl_good_mask])
            - np.min(average_slope_shade[avsl_good_mask])
        )
    )
    # at this point, I want to block out the original nodata values to figure out
    # the bright ends (additive) and dark ends (subtractive)
    logger.info("\tDetermining bright and dark ends of meta-hillshade")
    midpoint_slope_shade = (
        np.min(average_slope_shade[avsl_good_mask])
        + np.max(average_slope_shade[avsl_good_mask])
    ) / 2
    upper_quartile_slope_shade = (
        np.max(average_slope_shade[avsl_good_mask]) + midpoint_slope_shade
    ) / 2
    lower_quartile_slope_shade = (
        np.min(average_slope_shade[avsl_good_mask]) + midpoint_slope_shade
    ) / 2
    bright_slope_shade = np.where(
        average_slope_shade >= upper_quartile_slope_shade,
        average_slope_shade - upper_quartile_slope_shade,
        0.0,
    )
    dark_slope_shade = np.where(
        average_slope_shade <= lower_quartile_slope_shade,
        average_slope_shade - lower_quartile_slope_shade,
        0.0,
    )
    logger.info(
        "\tApplying bright and dark ends of meta-hillshade to" + " multiscale hillshade"
    )
    multiscale_hillshade[avsl_good_mask] += bright_slope_shade[avsl_good_mask]
    multiscale_hillshade[avsl_good_mask] -= dark_slope_shade[avsl_good_mask]
    # save4_hillshade = multiscale_hillshade.copy()

    # 4) penultimately, with original DEM, take the lower 50% of elevated areas and
    # proportionally darken to lower elevation
    dark_factor = 0.66  # up/down to make this more/less dramatic
    logger.info(f"\tDarkening lower elevation areas by {dark_factor:.2f}")
    threshold_elevation = (
        np.max(working_dem[dl_goodmask]) + np.min(working_dem[dl_goodmask])
    ) / 3
    height_darken = np.where(
        working_dem <= threshold_elevation, threshold_elevation - working_dem, 0.0
    )
    height_darken = (
        dark_factor
        * 255.0
        * (height_darken - np.min(height_darken[dl_goodmask]))
        / (np.max(height_darken[dl_goodmask]) - np.min(height_darken[dl_goodmask]))
    )
    multiscale_hillshade[dl_goodmask] -= height_darken[dl_goodmask]
    multiscale_hillshade = (
        255.0
        * (multiscale_hillshade - np.min(multiscale_hillshade))
        / (np.max(multiscale_hillshade) - np.min(multiscale_hillshade))
    )
    # save5_hillshade = multiscale_hillshade.copy()

    # 5) and finally, we would like to excise non-land pixels from the hillshade
    # raster. Sea values are typically zero, but land height can also pass
    # below zero. We can however, use our 50 pixel blurred dem to determine
    # those areas of zeros with negligent variation and then return to the
    # original dem and find other zeros that are contiguous. Then combine with
    # other masks and set to nodata.
    # sea mask if sea = 1, and if land = 0
    sea_mask_primed = np.where(base_dem[-1, :, :] == 0.0, 1, 0)
    n_primed = len(np.where(sea_mask_primed == 1)[0])
    sea_mask_to_test = np.where(base_dem[0, :, :] == 0.0, 1, 0)
    sea_labeled = label(sea_mask_to_test, background=0)

    # before getting started, can already blank out no data areas as non-land
    sea_mask = np.zeros(sea_mask_to_test.shape)
    sea_mask[avsl_badmask] = 1
    sea_mask[dl_badmask] = 1
    sea_mask[stp_bad_mask] = 1
    indexes = list(set(list(sea_labeled.ravel())))
    iidx = 0
    while n_primed > 0:
        index = indexes[iidx]
        if index == 0:
            iidx += 1
            continue
        to_mask = np.where(sea_labeled == index)
        sea_mask_primed[to_mask] = 0
        new_n_primed = len(np.where(sea_mask_primed == 1)[0])
        # only mask as sea if it covers sea-like (contiguous 50px zero) area
        if new_n_primed != n_primed:
            n_primed = new_n_primed
            sea_mask[to_mask] = 1
        iidx += 1
        if iidx >= len(indexes):
            break

    ###########################################################################
    # 6) save
    out_meta = working_meta.copy()
    out_meta.update(dtype="int16")
    out_meta.update(compress="lzw")
    multiscale_hillshade[np.where(sea_mask == 1)] = out_meta["nodata"]

    with rasterio.open(output_mshillshade_uri, "w", **out_meta) as outmshade:
        outmshade.write_band(1, multiscale_hillshade.astype(out_meta["dtype"]))

    logger.info(f"Multiscale hillshade saved to {output_mshillshade_uri}")

    # 7) tidy up if asked
    if delete_temp_files:
        logger.warning("Deleting temporary files")
        for rt, dr, fls in os.walk(working_dir):
            for fl in fls:
                fl_uri = os.path.join(rt, fl)
                if fl_uri not in keep_file:
                    os.remove(fl_uri)

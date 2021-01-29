import os
import logging
import zipfile
import glob
import getpass


import numpy as np
import geopandas as gpd
import rasterio
from rasterio import warp
import affine

import src.utils.requests_api as requests_api
from requests.exceptions import HTTPError

logger = logging.getLogger(__name__)


def get_srtm30_root_url():
    """
    The root url to download SRTM 30m zip files from
    """
    srtm30_url = 'http://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/2000.02.11/'
    return srtm30_url


def get_earthdata_login_credentials():
    """
    Tuple of (username, password) for logging into NASA EarthData. Ideally,
    this will be handled somewhere more secure!
    """
    username = input("Earth Data username: ")
    password = getpass.getpass("Earth Data password: ")

    return tuple([username, password])


def get_srtm90_root_url():
    """
    The root url to download SRTM 90m zip files from
    """
    srtm90_url = 'https://srtm.csi.cgiar.org/' + \
                 'wp-content/uploads/files/srtm_5x5/TIFF/'
    
    return srtm90_url


def longitude_to_srtm90_grid_long(longitude: float):
    """
    Determine longitudes within SRTM 5-degree grid
    SRTM longitude is 72 divisions
    Grid starts from 01 at 180E (Pacific Date Line)
    """
    return int(1 + (180 + longitude - (longitude % 5.)) / 5)


def latitude_to_srtm90_grid_lat(latitude: float):
    """
    Determine latitudes within SRTM 5-degree grid
    SRTM latitude is 24 divisions starting at 01 from 55N to 60S
    (using 60N as zero index)
    """
    return int((60 - (latitude - (latitude % 5.)))/5)


def longitude_to_srtm30_text_long(longitude: float):
    """
    Determine longitudes within SRTM 1-degree grid
    EW Meridian is E000 and tiles indexed by western-most coordinate
    """

    flong = np.floor(longitude)
    if flong < 0.:
        tlong = "W%03d" % (abs(flong))
    else:
        tlong = "E%03d" % (abs(flong))

    return tlong


def latitude_to_srtm30_text_lat(latitude: float):
    """
    Determine latitudes within SRTM 1-degree grid
    NS Equator is N00 and tiles indexed by southern-most coordinate
    """

    flat = np.floor(latitude)
    if flat < 0.:
        tlat = "S%02d" % (abs(flat))
    else:
        tlat = "N%02d" % (abs(flat))

    return tlat


def get_country_bounds(country_gpkg=None):
    """
    :param country_gpkg: geopackage of country border
    :return: bounds of country outline
    N.B. adding this as separate module to allow insertion of tests
    """

    if country_gpkg is None:
        # TODO: have some error behaviour when no file supplied
        logger.info("Defaulting to Yemen")
        country_bnd = np.array([41.81458282, 12.10819435,
                                54.53541565, 18.99999809])
    else:
        country = gpd.read_file(country_gpkg)
        assert country.crs == {'init': 'epsg:4326'}, \
            "Country outline is not in EPSG 4326 (WGS84)"
        country_bnd = country.total_bounds

    return country_bnd


def determine_srtm_90_zip(country_gpkg=None):
    """
    :param country_gpkg: geopackage of country border
    SRTM is stored in 5x5 degree grids, this script determines which zip
    files to download to cover the country
    """
    country_bnd = get_country_bounds(country_gpkg=country_gpkg)

    # TODO: Not dealt with countries that cross 180E/W longitude
    tru_long = [country_bnd[0], country_bnd[2]]
    tru_lat = [country_bnd[1], country_bnd[3]]

    # determine max / min lats longs 5 degree grid origins
    grid_long = list(range(longitude_to_srtm90_grid_long(min(tru_long)),
                           longitude_to_srtm90_grid_long(max(tru_long)) + 1))
    # NOTE: grid latitude increases with more southerly latitude
    grid_lat = list(range(latitude_to_srtm90_grid_lat(max(tru_lat)),
                          latitude_to_srtm90_grid_lat(min(tru_lat)) + 1))

    # TODO: check tile coincides with country outline
    for glon in grid_long:
        for glat in grid_lat:
            gcoord = '%02d_%02d' % (glon, glat)
            yield 'srtm_' + gcoord + '.zip'


def determine_srtm_30_zip(country_gpkg=None):
    """
    :param country_gpkg: geopackage of country border
    Determine longitudes within SRTM 1-degree grid
    More straightforward than SRTM 90!
    """
    country_bnd = get_country_bounds(country_gpkg=country_gpkg)

    # TODO: Not dealt with countries that cross 180E/W longitude
    tru_long = [country_bnd[0], country_bnd[2]]
    tru_lat = [country_bnd[1], country_bnd[3]]

    # TODO: check tile coincides with country outline
    glon = min(tru_long)
    while glon <= max(tru_long):
        glat = min(tru_lat)
        while glat <= max(tru_lat):
            gcoord = latitude_to_srtm30_text_lat(glat) + \
                     longitude_to_srtm30_text_long(glon)
            yield '.'.join([gcoord, 'SRTMGL1', 'hgt', 'zip'])
            glat += 1.
        glon += 1.


def fetch_srtm_30_zips(destination_folder: str, country_gpkg=None,
                       force_download=False):
    """
    Fetch relevant srtm 30 zip files for a given country boundary from the web
    to a local folder
    :param destination_folder: folder to which zip files should be downloaded
    :param country_gpkg: geopackage of country border
    :param  force_download: download zip if download already exists
    :return: list of zip files
    """

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    srtm_root_uri = get_srtm30_root_url()
    zip_list_out = []
    session = None

    for zipf in determine_srtm_30_zip(country_gpkg=country_gpkg):
        zip_web_uri = srtm_root_uri + zipf
        zip_local_uri = os.path.join(destination_folder, zipf)
        # TODO: skip file if (web hosted) zip file not found
        if force_download or not os.path.exists(zip_local_uri):
            logger.info("Downloading %s to %s" %
                        (zip_web_uri, zip_local_uri))
            if session is None:
                username, password = get_earthdata_login_credentials()
                session = requests_api.SessionWithHeaderRedirection(username,
                                                                    password)
            # if is_valid_accessible_url(zip_web_uri):
            response = session.get(zip_web_uri, stream=True)
            # raise an exception in case of http errors
            try:
                response.raise_for_status()
                with open(zip_local_uri, 'wb') as zl:
                    for chunk in response.iter_content(chunk_size=1024*1024):
                        zl.write(chunk)
                zip_list_out.append(zip_local_uri)
                logger.info("Downloaded %s to %s" %
                            (zipf, zip_local_uri))
            except HTTPError:
                logger.warning("Couldn't download %s from %s (%s error)" %
                               (zipf, zip_web_uri, response.status_code))

    return zip_list_out


def fetch_srtm_90_zips(destination_folder: str, country_gpkg=None,
                       force_download=False):
    """
    Fetch relevant srtm 90 zip files for a given country boundary from the web
    to a local folder
    :param destination_folder: folder to which zip files should be downloaded
    :param country_gpkg: geopackage of country border
    :param  force_download: download zip if download already exists
    :return: list of zip files
    """

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    srtm_root_uri = get_srtm90_root_url()
    zip_list_out = []

    for zipf in determine_srtm_90_zip(country_gpkg=country_gpkg):
        zip_web_uri = srtm_root_uri + zipf
        print(zip_web_uri)
        zip_local_uri = os.path.join(destination_folder, zipf)
        # TODO: skip file if (web hosted) zip file not found
        if force_download or not os.path.exists(zip_local_uri):
            requests_api.download_url(zip_web_uri, zip_local_uri,
                                      chunk_size=1024*1024)
        zip_list_out.append(zip_local_uri)

    return zip_list_out


def unzip_raster_to_folder(zip_list: list, out_folder: str, ext=".tif"):
    """
    Unzips a list of raster files into a folder
    :param zip_list: lits of zip files generated by fetch_srtm_##_zips
    :param out_folder: folder for zip contents to be extracted to.
    :param ext: file extension identifying the raster type
    :return: list of raster files extracted from zipfiles
    """

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    if not ext.startswith("."):
        logger.warning("Amending extension to '.%s'" % ext)
        ext = '.' + ext

    existing_raster = glob.glob(out_folder + os.sep + '*' + ext)

    for zipf in zip_list:
        arkiv = zipfile.ZipFile(zipf)
        arkiv.extractall(path=out_folder)

    unzipped_raster = glob.glob(out_folder + os.sep + '*' + ext)
    # unzipped_raster = [uzt for uzt in unzipped_raster
    #                    if uzt not in existing_raster]

    return unzipped_raster


def unzip_tif_to_folder(zip_list: list, out_folder: str):
    """
    Unzips a list of zip files into a folder
    :param zip_list: lits of zip files generated by fetch_srtm_##_zips
    :param out_folder: folder for zip contents to be extracted to.
    :return: list of .tif files extracted from zipfiles
    """
    return unzip_raster_to_folder(zip_list, out_folder, ext='.tif')


def mosaic_rasters(input_raster_uri_list, output_geotiff_uri, nullvalue=-9999):
    """
    Joins list of rasters together into a new tif file.
    :param input_raster_uri_list: list of rasters to mosaic together
    :param output_geotiff_uri: output name of mosaicked raster
    :param nullvalue: a no-data value (defaulting to -9999 to clear it from
           viable topogaphy heights)
    :return: None
    """

    if not os.path.exists(os.path.dirname(output_geotiff_uri)):
        os.makedirs(os.path.dirname(output_geotiff_uri))

    with rasterio.open(input_raster_uri_list[0], 'r') as inraster:
        mainmeta = inraster.meta
        maintransform = inraster.transform
        allbound = inraster.bounds.__dict__
        if allbound == {}:
            allbound = inraster.bounds._asdict()
        if inraster.res[0] != inraster.res[1]:
            raise Exception('Raster pixels not equal in x- and y- resolution')
        else:
            logger.info('Pixel resolution of %d units (assumed metres)' %
                        inraster.res[0])
            mainres = inraster.res

    # use mainmeta and maintransform as info going into new raster

    # first fix up metadata
    mainmeta.update(compress='lzw')
    if 'nodata' not in mainmeta.keys():
        mainmeta.update(nodata=nullvalue)
    elif mainmeta['nodata'] is None:
        mainmeta.update(nodata=nullvalue)
    if 'crs' not in mainmeta.keys():
        raise NotImplementedError('Cannot deal with lack of CRS in %s' %
                                  input_raster_uri_list[0])

    datacube = []  # container for raster layers

    # loop through DEMs and record data; adjust allbound too
    logger.info('Calculating combined boundary')
    for one_raster_uri in input_raster_uri_list:
        with rasterio.open(one_raster_uri, 'r') as inraster:
            datacube.append(tuple([inraster.read(1), inraster.meta]))
            # check subsequent CRS are consistent
            if inraster.meta['crs'] != mainmeta['crs']:
                err_msg = 'Cannot deal with lack of CRS in %s' % one_raster_uri
                logger.error(err_msg)
                raise NotImplementedError(err_msg)
            allbound['left'] = min([allbound['left'], inraster.bounds.left])
            allbound['right'] = max([allbound['right'], inraster.bounds.right])
            allbound['bottom'] = min([allbound['bottom'], inraster.bounds.bottom])
            allbound['top'] = max([allbound['top'], inraster.bounds.top])

    fullbound = tuple([abs(allbound['top'] - allbound['bottom']),
                       abs(allbound['right'] - allbound['left'])])

    # note that maps order resolution northing-easting, whilst python likes x-y
    newdims = [np.int(np.ceil(fullbound[0] / mainres[1])),
               np.int(np.ceil(fullbound[1] / mainres[0]))]
    newlayer = np.zeros(newdims)

    mainlayer = newlayer.copy()
    # save the new layer with appropriate geo information
    newtransform = affine.Affine(maintransform[0], maintransform[1],
                                 allbound['left'],
                                 maintransform[3], maintransform[4],
                                 allbound['top'])
    if 'affine' in mainmeta.keys():
        mainmeta.update(affine=newtransform)

    if mainmeta['driver'] != "GTiff":
        mainmeta.update(driver="GTiff")
    mainmeta.update(width=newlayer.shape[1])
    mainmeta.update(height=newlayer.shape[0])
    mainmeta.update(transform=newtransform)

    logger.info('Processing raster data to single file')
    # 1igure out where each layer should go and the size of the output raster.
    for layer in datacube:
        newlayer[:] = nullvalue
        src_transform = layer[1]['transform']
        warp.reproject(layer[0], newlayer, src_transform=src_transform,
                       src_crs=layer[1]['crs'], src_nodata=layer[1]['nodata'],
                       dst_transform=mainmeta['transform'],
                       dst_crs=mainmeta['crs'],
                       dst_nodata=nullvalue,
                       resampling=warp.Resampling.nearest)
        gooddata = np.where(newlayer != nullvalue)
        mainlayer[gooddata] = newlayer[gooddata]

    mainmeta.update(nodata=nullvalue)
    with rasterio.open(output_geotiff_uri, 'w', **mainmeta) as outraster:
        outraster.write_band(1, mainlayer.astype(mainmeta['dtype']))
    logger.info(f'Wrote to {output_geotiff_uri}')


def transform_raster_to_metre_projection(country_geotiff_uri: str,
                                         destination_epsg: str,
                                         reprojected_geotiff_uri: str):
    """
    Mosaicked(or single tile) SRTM files are projected in WGS84 degrees.
    To be able to calculate slope, we need the raster in a local projection
    in metres
    :param country_geotiff_uri: the geotiff raster for the country
    :param destination_epsg: the target projection as string "EPSG:#####"
    :param reprojected_geotiff_uri: the raster for the country in the new
           projection
    """

    with rasterio.open(country_geotiff_uri) as src:
        dst_transform, dst_width, dst_height = \
            warp.calculate_default_transform(src.crs, destination_epsg, src.width,
                                             src.height, *src.bounds)
        src_data = src.read(1)
        src_meta = src.meta
        src_crs = src.crs
        src_transform = src.transform

    dst_meta = src_meta.copy()
    dst_meta.update(crs=destination_epsg)
    dst_meta.update(transform=dst_transform)
    dst_meta.update(width=dst_width)
    dst_meta.update(height=dst_height)
    dst_meta.update(compress='lzw')

    with rasterio.open(reprojected_geotiff_uri, 'w', **dst_meta) as dst:
        warp.reproject(source=src_data,
                       destination=rasterio.band(dst, 1),
                       src_transform=src_transform,
                       src_crs=src_crs,
                       dst_transform=dst_transform,
                       dst_crs=destination_epsg,
                       resampling=warp.Resampling.nearest)


def get_srtm30_for_country(output_geotiff_uri: str,
                           download_folder: str, destination_epsg: str,
                           country_gpkg=None, nullvalue=-9999,
                           force_download=False):
    """
    Fetches SRTM 30m data for a given country. Defaults to Yemen
    Have exposed null value and  a force_download flag for rerunning but
    they aren't used in the snakemake setup
    :param download_folder: where raw srtm contents to be downloaded/extracted
    :param destination_epsg: string of "EPSG:#####" for desired projection
    :param output_geotiff_uri: mosaic'ed data to end up here
    :param country_gpkg: geopackage of desired country outline (s1 ad0)
    :param nullvalue: 'no data' value of mosaicked raster
    :param force_download: if zip files must be downloaded again
                           (e.g. when zip files corrupted)
    """

    zip_download_folder = download_folder + os.sep + 'zip'
    srtm30_zip_list = fetch_srtm_30_zips(zip_download_folder,
                                         country_gpkg=country_gpkg,
                                         force_download=force_download)
    srtm30_hgt_list = unzip_raster_to_folder(srtm30_zip_list, download_folder,
                                             ext=".hgt")
    temp_mosaic_uri = '_wgs84'.join(os.path.splitext(output_geotiff_uri))
    mosaic_rasters(srtm30_hgt_list, temp_mosaic_uri, nullvalue=nullvalue)
    transform_raster_to_metre_projection(temp_mosaic_uri, destination_epsg,
                                         output_geotiff_uri)
    os.remove(temp_mosaic_uri)
    return output_geotiff_uri


def get_srtm90_for_country(output_geotiff_uri: str,
                           download_folder: str, destination_epsg: str,
                           country_gpkg=None, nullvalue=-9999,
                           force_download=False):
    """
    Fetches SRTM 90m data for a given country. Have exposed null value and
    a force_download flag for rerunning but they aren't used in the snakemake
    setup
    :param download_folder: where raw srtm contents to be downloaded/extracted
                            zips are downloaded to a subfolder entitiled "zip"
    :param destination_epsg: string of "EPSG:#####" for desired projection
    :param output_geotiff_uri: mosaic'ed data to end up here
    :param country_gpkg: path to polygon geopackage of desired country outline
    :param nullvalue: 'no data' value of mosaicked raster
    :param force_download: if zip files must be downloaded again
                           (e.g. when zip files corrrupted)
    """

    zip_download_folder = download_folder + os.sep + 'zip'
    srtm90_zip_list = fetch_srtm_90_zips(zip_download_folder,
                                         country_gpkg=country_gpkg,
                                         force_download=force_download)
    srtm90_tif_list = unzip_tif_to_folder(srtm90_zip_list, download_folder)
    temp_mosaic_uri = '_wgs84'.join(os.path.splitext(output_geotiff_uri))
    mosaic_rasters(srtm90_tif_list, temp_mosaic_uri, nullvalue=nullvalue)
    transform_raster_to_metre_projection(temp_mosaic_uri, destination_epsg,
                                         output_geotiff_uri)
    os.remove(temp_mosaic_uri)
    return output_geotiff_uri

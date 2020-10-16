import gdal
from osgeo import gdal_array
import ogr, osr
import numpy as np
import os
import subprocess
from tempfile import TemporaryDirectory


def get_raster_bounds(raster):
    """
    Returns a wkbPolygon geometry with the bounding rectangle of a raster calculated from its geotransform.

    Parameters
    ----------
    raster : gdal.Image
        A gdal.Image object

    Returns
    -------
    boundary : ogr.Geometry
        An ogr.Geometry object containing a single wkbPolygon with four points defining the bounding rectangle of the
        raster.

    Notes
    -----
    Bounding rectangle is obtained from raster.GetGeoTransform(), with the top left corners rounded
    down to the nearest multiple of of the resolution of the geotransform. This is to avoid rounding errors in
    reprojected geotransformations.
    """
    raster_bounds = ogr.Geometry(ogr.wkbLinearRing)
    geotrans = raster.GetGeoTransform()
    # We can't rely on the top-left coord being whole numbers any more, since images may have been reprojected
    # So we floor to the resolution of the geotransform maybe?
    top_left_x = floor_to_resolution(geotrans[0], geotrans[1])
    top_left_y = floor_to_resolution(geotrans[3], geotrans[5]*-1)
    width = geotrans[1]*raster.RasterXSize
    height = geotrans[5]*raster.RasterYSize * -1  # RasterYSize is +ve, but geotransform is -ve
    raster_bounds.AddPoint(top_left_x, top_left_y)
    raster_bounds.AddPoint(top_left_x + width, top_left_y)
    raster_bounds.AddPoint(top_left_x + width, top_left_y - height)
    raster_bounds.AddPoint(top_left_x, top_left_y - height)
    raster_bounds.AddPoint(top_left_x, top_left_y)
    bounds_poly = ogr.Geometry(ogr.wkbPolygon)
    bounds_poly.AddGeometry(raster_bounds)
    return bounds_poly


def floor_to_resolution(input, resolution):
    """
    Returns input rounded DOWN to the nearest multiple of resolution. Used to prevent float errors on pixel boarders.

    Parameters
    ----------
    input : number
        The value to be rounded
    resolution : number
        The resolution

    Returns
    -------
    output : number
        The largest value between input and 0 that is wholly divisible by resolution.

    Notes
    -----
    Uses the following formula: ``input-(input%resolution)``
    If resolution is less than 1, then this assumes that the projection is in decmial degrees and will be rounded to 6dp
    before flooring. However, it is not recommended to use this function in that situation.

    """
    if resolution > 1:
        return input - (input%resolution)
    else:
        print("Low resolution detected, assuming in degrees. Rounding to 6 dp.\
                Probably safer to reproject to meters projection.")
        resolution = resolution * 1000000
        input = input * 1000000
        return (input-(input%resolution))/1000000


def get_aoi_intersection(raster, aoi):
    """
    Returns a wkbPolygon geometry with the intersection of a raster and a shpefile containing an area of interest

    Parameters
    ----------
    raster : gdal.Image
        A raster containing image data
    aoi : ogr.DataSource
        A datasource with a single layer and feature
    Returns
    -------
    intersection : ogr.Geometry
        An ogr.Geometry object containing a single polygon with the area of intersection

    """
    raster_shape = get_raster_bounds(raster)
    aoi.GetLayer(0).ResetReading()  # Just in case the aoi has been accessed by something else
    aoi_feature = aoi.GetLayer(0).GetFeature(0)
    aoi_geometry = aoi_feature.GetGeometryRef()
    return aoi_geometry.Intersection(raster_shape)


def bound_area(dataset):
    """Returns the bounds (x_min, y_min, x_max, y_max) for non-zero data of an array. Assumes that
    the LAST two dimensions of dataset are x (lon) and y (lat)"""
    non_zero_x, non_zero_y = np.nonzero(dataset[0,:,:])  # Gets the non-zero of the las
    x_min = non_zero_x.min()
    x_max = non_zero_x.max()
    y_min = non_zero_y.min()
    y_max = non_zero_y.max()
    return x_min, y_min, x_max, y_max


def mask_dataset_to_region(dataset, lats, lons, region_path,
                           dataset_epsg = 4326, crop_output = True, output_nodata = 0):
    """Replaces every value outside of the region at region_path with a"""
    with TemporaryDirectory() as td:
        # Creating temporary raster
        tmp_raster_path = os.path.join(td, 'raster')
        tmp_out_path = os.path.join(td, "out")
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(dataset_epsg)
        meatball_geotransform = [lats[0], lats[1]-lats[0], 0,
                        lons[0], 0, -1*(lons[1]-lons[0])]   # This will NOT be accurate over more than a few km
        driver = gdal.GetDriverByName("GTiff")
        type_code = gdal_array.NumericTypeCodeToGDALTypeCode(dataset.dtype)
        raster = driver.Create(
            tmp_raster_path,
            xsize=dataset.shape[2],
            ysize=dataset.shape[1],
            bands=dataset.shape[0],
            eType=type_code
        )
        raster.SetGeoTransform(meatball_geotransform)
        raster.SetProjection(srs.ExportToWkt())
        ras_array = raster.GetVirtualMemArray(eAccess=gdal.GA_Update)
        ras_array[...] = dataset

        # Clipping array
        aoi = ogr.Open(region_path)
        intersection = get_aoi_intersection(raster, aoi)
        min_x_geo, max_x_geo, min_y_geo, max_y_geo = intersection.GetEnvelope()
        width_pix = int(np.floor(max_x_geo - min_x_geo)/meatball_geotransform[1])
        height_pix = int(np.floor(max_y_geo - min_y_geo)/np.absolute(meatball_geotransform[5]))
        clip_spec = gdal.WarpOptions(
            format="GTiff",
            cutlineDSName=region_path,
            cropToCutline=False,
            width=width_pix,
            height=height_pix,
            dstSRS=srs,
            dstNodata=output_nodata
        )
        temp_out = gdal.Warp(tmp_out_path, raster, options = clip_spec)
        clipped_dataset = temp_out.ReadAsArray().copy()

        # If required, clipping array to non-zero bounds
        if crop_output:
            x_min, y_min, x_max, y_max = bound_area(clipped_dataset)
            clipped_dataset = clipped_dataset[..., x_min:x_max, y_min:y_max]
            lats = lats[y_min:y_max]
            lons = lons[x_min:x_max]

        #This is GDAL, you need to close everything by hand or you get segfaults. In python! It's ridiculous.
        temp_out = None
        ras_array = None
        raster = None

        return clipped_dataset, lats, lons

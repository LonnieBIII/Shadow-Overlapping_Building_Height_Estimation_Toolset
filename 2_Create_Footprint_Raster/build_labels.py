#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

Build Labels Helper function
--------------------------------------------------

Purpose: Creates building footprint raster from vector footprint SHP or GEOJSON file in the same size as the original image raster

Author:
- Lonnie Byrnside III

Additional Notes:
- Create building mask function pulled from:
    Adam Van Etten
    https://gist.github.com/avanetten/b295e89f6fa9654c9e9e480bdb2e4d60
    2017
- Spatial reference management function pulled from:
    Schuyler Erle, Frank Warmerdam, Even Rouault
    https://svn.osgeo.org/gdal/trunk/gdal/swig/python/samples/gdalcopyproj.py
    2005

"""

import os
from osgeo import gdal
from create_building_mask import create_building_mask


def main(src_raster_path, src_vector_path, dst_path):
    create_building_mask(
            src_raster_path, src_vector_path, npDistFileName=dst_path,
            noDataValue=0, burn_values=255
    )


src_raster_path = input("Enter original image raster path: ")
src_vector_path = input("Enter vector building footprint SHP/GEOJSON path: ")
dst_path_fldr = input("Enter output folder path: ")
dst_path_name = input("Enter output image name: ")
dst_path = os.path.join(dst_path_fldr, dst_path_name + ".tif")


main(src_raster_path, src_vector_path, dst_path)


dataset = gdal.Open( src_raster_path )
projection   = dataset.GetProjection()
geotransform = dataset.GetGeoTransform()

if projection is not None and geotransform is not None:
    dataset2 = gdal.Open( dst_path, gdal.GA_Update )
    if geotransform is not None and geotransform != (0,1,0,0,0,1):
        dataset2.SetGeoTransform( geotransform )
    if projection is not None and projection != '':
        dataset2.SetProjection( projection )
    gcp_count = dataset.GetGCPCount()
    if gcp_count != 0:
        dataset2.SetGCPs( dataset.GetGCPs(), dataset.GetGCPProjection() )

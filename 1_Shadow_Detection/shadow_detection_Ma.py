# -*- coding: utf-8 -*-

"""
Paper:
H. Ma, Q. Qin, and X. Shen
Shadow segmentation and compensation in high resolution satellite images
2008

Code:
Nassim REFES
https://github.com/NassimREFES/shadow-detection-algorithms
2022

"""

import cv2
import numpy
import numpy as np
from matplotlib import pyplot as plt
import math
import copy
import sys
import os
from osgeo import gdal


def nsvdi(img):
    _index = []

    rows = img.shape[0]
    cols = img.shape[1]


    h, s, v = cv2.split(img)

    for row in range(rows):
        _index.append([])
        for col in range(cols):
            if s[row][col] + v[row][col] == 0.0:
                _index[row].append(0)
            else:
                _index[row].append( (s[row][col] - v[row][col]) / (s[row][col] + v[row][col]) )

    return np.array(_index, dtype=np.float64)
    #return img

def main(input_image, folder_path, base_name):

    img = cv2.imread(input_image)

    # RGB to HSV
    bgr_to_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    bgr_to_hsv = bgr_to_hsv.astype(np.float64)

    # normalize HSV in [0, 1]
    cv2.normalize(bgr_to_hsv, bgr_to_hsv, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    #bgr_to_hsv = bgr_to_hsv / 255
    
    # build index NSVDI
    _index = nsvdi(bgr_to_hsv)

    normlize_index = np.zeros((_index.shape[0], _index.shape[1]))
    cv2.normalize(_index, normlize_index, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    normlize_index = np.array(np.round(normlize_index), dtype=np.uint8)
    otsu, res_otsu = cv2.threshold(normlize_index, 0, 255, type=cv2.THRESH_OTSU)

    bgr_to_hsv = bgr_to_hsv.astype(np.uint8)


    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(res_otsu, kernel, iterations=1)
    #dilation = cv2.dilate(erosion, kernel, iterations=1)

    
    cv2.imwrite(os.path.join(folder_path, base_name + '_Ma.tif'), erosion)

    dataset = gdal.Open( input_image )
    projection   = dataset.GetProjection()
    geotransform = dataset.GetGeoTransform()

    if projection is not None and geotransform is not None:
        dataset2 = gdal.Open( os.path.join(folder_path, base_name + '_Ma.tif'), gdal.GA_Update )
        if geotransform is not None and geotransform != (0,1,0,0,0,1):
            dataset2.SetGeoTransform( geotransform )
        if projection is not None and projection != '':
            dataset2.SetProjection( projection )
        gcp_count = dataset.GetGCPCount()
        if gcp_count != 0:
            dataset2.SetGCPs( dataset.GetGCPs(), dataset.GetGCPProjection() )


if __name__ == '__main__':
    main(input_image, folder_path, base_name)

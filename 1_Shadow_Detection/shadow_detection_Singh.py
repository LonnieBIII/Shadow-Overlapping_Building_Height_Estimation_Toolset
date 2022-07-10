"""
Paper:
K. K. Singh, K. Pal, and M. J. Nigam
Shadow detection and removal from remote sensing images using NDI and morphological operators
2012

Code:
Megh Thakkar
https://github.com/scikit-image/scikit-image/issues/3214
2018

"""

from skimage import data, exposure, filters, morphology, io, color, img_as_float, img_as_ubyte
import skimage
import cv2
import os
from osgeo import gdal
from skimage.measure import label
from PIL import Image
import numpy as np

np.seterr(divide='ignore', invalid='ignore')


def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out

def imbinarize(img, threshold):
    img[img >= threshold] = 1
    img[img < threshold] = 0
    return img

def main(input_image, folder_path, base_name):
    image = img_as_float(io.imread(input_image))
    hsv_img  = color.rgb2hsv(image)

    H = hsv_img[:, :, 0]
    S = hsv_img[:, :, 1]
    V = hsv_img[:, :, 2]
    NDI = (S - V)/(S + V)

    NDI[np.argwhere(np.isnan(NDI))] = 0	# prevent NaN

    level = filters.threshold_otsu(NDI)
    binary_mask = imbinarize(NDI, level)

    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(binary_mask, kernel, iterations=1)
    #dilation = cv2.dilate(erosion, kernel, iterations=1)

    output = img_as_ubyte(erosion)

    cv2.imwrite(os.path.join(folder_path, base_name + '_Singh.tif'), output)

    dataset = gdal.Open( input_image )
    projection   = dataset.GetProjection()
    geotransform = dataset.GetGeoTransform()

    if projection is not None and geotransform is not None:
        dataset2 = gdal.Open( os.path.join(folder_path, base_name + '_Singh.tif'), gdal.GA_Update )
        if geotransform is not None and geotransform != (0,1,0,0,0,1):
            dataset2.SetGeoTransform( geotransform )
        if projection is not None and projection != '':
            dataset2.SetProjection( projection )
        gcp_count = dataset.GetGCPCount()
        if gcp_count != 0:
            dataset2.SetGCPs( dataset.GetGCPs(), dataset.GetGCPProjection() )


if __name__ == '__main__':
    main(input_image, folder_path, base_name)


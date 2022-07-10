"""
Paper:
G. F. Silva, G. B. Carneiro, R. Doth, L. A. Amaral, and D. F. G. Azevedo
Near real-time shadow detection and removal in aerial motion imagery application
2018

Code:
Thomas Wang Wei Hong
https://github.com/ThomasWangWeiHong/Shadow-Detection-Algorithm-for-Aerial-and-Satellite-Images
2019

"""

import cv2
import os
import gc
import numpy as np
import pandas as pd
import rasterio
from skimage.color import lab2lch, rgb2lab
from skimage.exposure import rescale_intensity
from skimage.morphology import disk
from sklearn.cluster import KMeans



def shadow_detection(image_file, shadow_mask_file, convolve_window_size, num_thresholds):
    """
    This function is used to detect shadow - covered areas in an image
    
    Inputs:
    - image_file: Path of image to be processed for shadow removal. It is assumed that the first 3 channels are ordered as
                  Red, Green and Blue respectively
    - shadow_mask_file: Path of shadow mask to be saved
    - convolve_window_size: Size of convolutional matrix filter to be used for blurring of specthem ratio image
    - num_thresholds: Number of thresholds to be used for automatic multilevel global threshold determination
    
    Outputs:
    - shadow_mask: Shadow mask for input image
    
    """
    
    if (convolve_window_size % 2 == 0):
        raise ValueError('Please make sure that convolve_window_size is an odd integer')
        
    buffer = int((convolve_window_size - 1) / 2)
    
    
    with rasterio.open(image_file) as f:
        metadata = f.profile
        img = rescale_intensity(np.transpose(f.read(tuple(np.arange(metadata['count']) + 1)), [1, 2, 0]), out_range = 'uint8')
        img = img[:, :, 0 : 3]
    
    
    lch_img = np.float32(lab2lch(rgb2lab(img)))
    
    
    l_norm = rescale_intensity(lch_img[:, :, 0], out_range = (0, 1))
    h_norm = rescale_intensity(lch_img[:, :, 2], out_range = (0, 1))
    sr_img = (h_norm + 1) / (l_norm + 1)
    log_sr_img = np.log(sr_img + 1)
    
    del l_norm, h_norm, sr_img
    gc.collect()

    

    avg_kernel = np.ones((convolve_window_size, convolve_window_size)) / (convolve_window_size ** 2)
    blurred_sr_img = cv2.filter2D(log_sr_img, ddepth = -1, kernel = avg_kernel)
      
    
    del log_sr_img
    gc.collect()
    
                
    flattened_sr_img = blurred_sr_img.flatten().reshape((-1, 1))
    labels = KMeans(n_clusters = num_thresholds + 1, max_iter = 10000).fit(flattened_sr_img).labels_
    flattened_sr_img = flattened_sr_img.flatten()
    df = pd.DataFrame({'sample_pixels': flattened_sr_img, 'cluster': labels})
    threshold_value = df.groupby(['cluster']).min().max()[0]
    df['Segmented'] = np.uint8(df['sample_pixels'] >= threshold_value)
    
    
    del blurred_sr_img, flattened_sr_img, labels, threshold_value
    gc.collect()
    
    
    shadow_mask_initial = np.array(df['Segmented']).reshape((img.shape[0], img.shape[1]))
    #struc_elem = np.ones((3, 3), np.uint8)
    #erosion = cv2.erode(np.uint8(shadow_mask_initial), struc_elem, iterations=1)
    #dilation = cv2.dilate(erosion, struc_elem, iterations=1)
    shadow_mask = np.expand_dims(shadow_mask_initial, axis = 0)
    
    
    del df, shadow_mask_initial #, struc_elem
    gc.collect()
    
    shadow_mask *= 255

    metadata['count'] = 1
    with rasterio.open(shadow_mask_file, 'w', **metadata) as dst:
        dst.write(shadow_mask)


def main(input_image, folder_path, base_name):
    shadow_detection(input_image, os.path.join(folder_path, base_name + '_Silva.tif'), convolve_window_size = 1, num_thresholds = 4)

    #orig
    #shadow_detection(input_image, os.path.join(folder_path, base_name + '_Silva.tif'), convolve_window_size = 5, num_thresholds = 3, struc_elem_size = 5)


if __name__ == '__main__':
    main(input_image, folder_path, base_name)


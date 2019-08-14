import cv2
import numpy as np
from os.path import relpath
from os import makedirs
import pymeanshift as pms
import re

import scipy
import scipy.misc

def mean_shift_transform(im_num, image, f2):
    # Get the raw RGB values from the hdf5 image
    dem = (list(image["georef_img"]["layers"]['dem']['array']))
    ndvi = (list(image["georef_img"]["layers"]['ndvi']['array']))

    dem_image = np.asarray(dem)
    ndvi_image = np.asarray(ndvi)

    # Convert to Mean Shift
    (mean_shift_image, labels_image, number_regions) = pms.segment(ndvi_image, spatial_radius=1, range_radius=1, min_density=300)
    mean_shift_image = np.dstack((dem_image, mean_shift_image))
    save_nmp_array(im_num, mean_shift_image, 'mean_shift', f2)

def edge_detector_transform(f2, im_num, image, sigma=0.33):
    # Get the raw RGB values from the hdf5 image
    dem = (list(image["georef_img"]["layers"]['dem']['array']))
    nir = (list(image["georef_img"]["layers"]['nir']['array']))

    dem_image = np.asarray(dem)
    nir_image = np.asarray(nir)

    # compute the median of the single channel pixel intensities
    v = np.median(nir_image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    edge_detector_image = cv2.Canny(nir_image,lower,upper)

    edge_detector_image = np.dstack((dem_image, edge_detector_image))
    save_nmp_array(im_num, edge_detector_image, 'edge_detector', f2)

# Performs histogram equalization on nir
def hist_equal_transform(im_num, image, f2):
    dem = (list(image["georef_img"]["layers"]['dem']['array']))
    ndvi = (list(image["georef_img"]["layers"]['ndvi']['array']))

    dem_image = np.asarray(dem)
    ndvi_image = np.asarray(ndvi)

    ndvi_image = cv2.equalizeHist(ndvi_image)

    equal_img = np.dstack((dem_image, ndvi_image))
    save_nmp_array(im_num, equal_img, 'hist_equal', f2)

def rgb(f2, im_num, image):
    rgb_image = (list(image["georef_img"]["layers"]['visible']['array']))
    dem = (list(image["georef_img"]["layers"]['dem']['array']))
    rgb_image = np.asarray(rgb_image)
    dem_image = np.asarray(dem)

    rgb_image = np.dstack((dem_image, rgb_image))
    save_nmp_array(im_num, rgb_image, 'rgb', f2)
    
def save_nmp_array(im_num, new_image, folder, f2):

    # Save the Lab image as a numpy array to preserve accuracy - Tensorflow will need to read in these images with numpy
    # We will also need a way of either saving the tree masks, or retrieving them from the original image
    # np.save("../img/%s/%s%s" % (folder, im_num, '.npy'), new_image)
    cv2.imwrite("../img/%s/%s%s" % (folder, im_num, '.png'), new_image)
    f2.write('img/' + folder + '/' + str(im_num) + '.png img/mask/' + str(im_num) + '.png' + "\n")

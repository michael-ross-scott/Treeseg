import cv2
import numpy as np
from os.path import relpath
from os import makedirs
import pymeanshift as pms
import re


def mean_shift_transform(im_num, image):
    # Get the raw RGB values from the hdf5 image
    ndvi_image = (list(image["georef_img"]["layers"]['ndvi']['array']))
    np_image = np.asarray(ndvi_image)

    # Convert to Mean Shift
    (mean_shift_image, labels_image, number_regions) = pms.segment(np_image, spatial_radius=1, range_radius=1, min_density=300)
    save_nmp_array(im_num, mean_shift_image, 'mean_shift')


def edge_detector_transform(im_num, image, sigma=0.33):
    # Get the raw RGB values from the hdf5 image
    nir_image = (list(image["georef_img"]["layers"]['nir']['array']))
    np_image = np.asarray(nir_image)

    # compute the median of the single channel pixel intensities
    v = np.median(nir_image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    edge_detector_image = cv2.Canny(np_image,lower,upper)

    save_nmp_array(im_num, edge_detector_image, 'edge_detector')


# Performs histogram equalization on nir
def hist_equal_transform(im_num, image):
    ndvi_image = (list(image["georef_img"]["layers"]['ndvi']['array']))
    np_image = np.asarray(ndvi_image)

    equal_img = cv2.equalizeHist(np_image)

    save_nmp_array(im_num, equal_img, 'hist_equal')


def save_nmp_array(im_num, new_image, folder):

    # Save the Lab image as a numpy array to preserve accuracy - Tensorflow will need to read in these images with numpy
    # We will also need a way of either saving the tree masks, or retrieving them from the original image
    cv2.imwrite("../img/%s/%s%s" % (folder, im_num, '.png'), new_image)

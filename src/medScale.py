import cv2
import numpy as np
from os.path import relpath
from os import makedirs
import pymeanshift as pms

def mean_shift_transform(image):
    # Get the raw RGB values from the hdf5 image
    ndvi_image = (list(image["georef_img"]["layers"]['ndvi']['array']))
    np_image = np.asarray(ndvi_image)

    # Convert to Mean Shift
    (mean_shift_image, labels_image, number_regions) = pms.segment(np_image, spatial_radius=3, range_radius=4.5, min_density=300)
    save_nmp_array(image, mean_shift_image, 'mean_shift')

def edge_detector_transform(image, sigma=0.33):
    # Get the raw RGB values from the hdf5 image
    nir_image = (list(image["georef_img"]["layers"]['nir']['array']))
    np_image = np.asarray(nir_image)

    # compute the median of the single channel pixel intensities
    v = np.median(nir_image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    edge_detector_image = cv2.Canny(np_image,lower,upper)

    save_nmp_array(image, edge_detector_image, 'edge_detector')

# Performs histogram equalization on nir
def hist_equal_transform(image):
    ndvi_image = (list(image["georef_img"]["layers"]['ndvi']['array']))
    np_image = np.asarray(ndvi_image)

    equal_img = cv2.equalizeHist(np_image)

    save_nmp_array(image, equal_img, 'hist_equal')
    
#dir to make needs to be updated at the end to work on all machines. Had some trouble on Windows and just hardcoded it for testing
def save_nmp_array(hd5image, new_image, folder):
    # Get relative path for output directory, minus the file extension
    rel_path = relpath(hd5image.filename)[14:-3]

    # Work out which parent directories need to be created before writing the file
    try:
        #directory_to_make = re.search(r'^([0-9a-zA-Z/ ]*/[0-9a-zA-Z\-]*/[0-9a-zA-Z\-]*)/[a-zA-Z0-9]*$', rel_path).groups(1)
        directory_to_make = rel_path
    except:
        raise Exception("Invalid directory path")

    # Make the parent directories
    makedirs("../img/%s/%s" % (folder, directory_to_make), exist_ok=True)

    # Save the Lab image as a numpy array to preserve accuracy - Tensorflow will need to read in these images with numpy
    # We will also need a way of either saving the tree masks, or retrieving them from the original image
    np.save("../img/%s/%s%s" % (folder, rel_path, '.npy'), new_image)
    cv2.imwrite("../img/%s/%s%s" % (folder, rel_path, '.png'), new_image)

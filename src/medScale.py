import cv2
import numpy as np
from os.path import relpath
from os import makedirs
import pymeanshift as pms

def mean_shift_transform(image):
    # Get the raw RGB values from the hdf5 image
    rgb_image = (list(image["georef_img"]["layers"]['visible']['array']))
    rgb_image = np.asarray(rgb_image)

    # Convert to Mean Shift
    (mean_shift_image, labels_image, number_regions) = pms.segment(rgb_image, spatial_radius=3, range_radius=4.5, min_density=300)
    save_nmp_array(image, mean_shift_image, 'mean_shift')

def edge_detector_transform(image):
    # Get the raw RGB values from the hdf5 image
    rgb_image = (list(image["georef_img"]["layers"]['nir']['array']))
    # v = np.median(rgb_image)
    rgb_image = np.asarray(rgb_image)

    #Convert to Canny Edge Detector
    higher_threshold = 275
    lower_threshold = 230
    # sigma = 0.33
    # lower_threshold = int(max(0, (1.0 - sigma) * v))
    # higher_threshold = int(min(255, (1.0 + sigma) * v))
    aperture = 3

    edge_detector_image = cv2.Canny(rgb_image, lower_threshold, higher_threshold, aperture)
    save_nmp_array(image, edge_detector_image, 'edge_detector')

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

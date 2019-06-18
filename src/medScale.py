import cv2 as cv
import numpy as np
from os.path import relpath
from os import makedirs
from sklearn.cluster import MeanShift, estimate_bandwidth

def mean_shift_transform(image):
    # Get the raw RGB values from the hdf5 image
    rgb_image = (list(image["georef_img"]["layers"]['visible']['array']))
    rgb_image = np.asarray(rgb_image)

    # Convert to Mean Shift
    original_shape = rgb_image.shape
    X = np.reshape(rgb_image, [-1, 3])
    bandwidth = estimate_bandwidth(X, quantile=0.1, n_samples=100)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(X)
    labels = ms.labels_

    mean_shift_image = np.reshape(labels, original_shape[:2])  # Just take size, ignore RGB channels.
    save_nmp_array(image, mean_shift_image, 'mean_shift')

def edge_detector_transform(image):
    # Get the raw RGB values from the hdf5 image
    rgb_image = (list(image["georef_img"]["layers"]['visible']['array']))
    rgb_image = np.asarray(rgb_image)

    #Convert to Canny Edge Detector
    edge_detector_image = cv.Canny(rgb_image, 100, 200)
    save_nmp_array(image, edge_detector_image, 'edge_detector')

#Stolen from Fudges
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

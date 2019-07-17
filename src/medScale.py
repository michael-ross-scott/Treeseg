import cv2
from pylab import *

from os.path import relpath
from os import makedirs
# import re


def mean_shift_transform(image):
    pass


def edge_detector_transform(image):
    pass


# Performs histogram equalization on nir
def hist_equal_transform(image):
    ndvi_image = (list(image["georef_img"]["layers"]['ndvi']['array']))
    np_image = np.asarray(ndvi_image)

    equal_img = cv2.equalizeHist(np_image)

    save_nmp_array(image, equal_img, 'hist_equal')


def save_nmp_array(hd5image, new_image, folder):
    # Get relative path for output directory, minus the file extension
    rel_path = relpath(hd5image.filename)[0:-3]

    # Work out which parent directories need to be created before writing the file
    # directory_to_make = re.search(r'^([0-9a-zA-Z/ ]*/[0-9a-zA-Z\-]*/[0-9a-zA-Z\-]*)/[a-zA-Z0-9]*$', rel_path).group(1)
    directory_to_make = rel_path
    if not directory_to_make:
        # Ya messed up real bad
        raise Exception("Invalid directory path")

    # Make the parent directories
    makedirs("img/%s/%s" % (folder, directory_to_make), exist_ok=True)

    # Save the Lab image as a numpy array to preserve accuracy - Tensorflow will need to read in these images with numpy
    # We will also need a way of either saving the tree masks, or retrieving them from the original image
    np.save("../img/%s/%s%s" % (folder, rel_path, '.npy'), new_image)
    cv2.imwrite("../img/%s/%s%s" % (folder, rel_path, '.png'), new_image)
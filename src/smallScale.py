import numpy as np
from skimage import color
from os.path import relpath
from os import makedirs
import re


def lab_transform(image):
    # Get the raw RGB values from the hdf5 image
    rgb_image = (list(image["georef_img"]["layers"]['visible']['array']))
    rgb_image = np.asarray(rgb_image)

    # Convert to lab
    lab_image = color.rgb2lab(rgb_image)
    save_nmp_array(image, lab_image, 'lab')


def hsi_transform(image):
    # Get the raw RGB values from the hdf5 image
    rgb_image = (list(image["georef_img"]["layers"]['visible']['array']))
    rgb_image = np.asarray(rgb_image)

    # Convert to HSI - Note this saves H as a fraction, instead of degrees
    hsi_image = color.rgb2hsv(rgb_image)
    save_nmp_array(image, hsi_image, 'hsi')


def save_nmp_array(hd5image, new_image, folder):
    # Get relative path for output directory, minus the file extension
    rel_path = relpath(hd5image.filename)[6:-3]

    # Work out which parent directories need to be created before writing the file
    directory_to_make = re.search(r'^([0-9a-zA-Z/ ]*/[0-9a-zA-Z\-]*/[0-9a-zA-Z\-]*)/[a-zA-Z0-9]*$', rel_path).group(1)
    if not directory_to_make:
        # Ya messed up real bad
        raise Exception("Invalid directory path")

    # Make the parent directories
    makedirs("../img/%s/%s" % (folder, directory_to_make), exist_ok=True)

    # Save the Lab image as a numpy array to preserve accuracy - Tensorflow will need to read in these images with numpy
    # We will also need a way of either saving the tree masks, or retrieving them from the original image
    np.save("../img/%s/%s%s" % (folder, rel_path, '.npy'), new_image)
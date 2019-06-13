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

    # Get relative path for output directory, minus the file extension
    rel_path = relpath(image.filename)[6:-3]

    # Work out which parent directories need to be created before writing the file
    directory_to_make = re.search(r'^([0-9a-zA-Z/ ]*/[0-9a-zA-Z\-]*/[0-9a-zA-Z\-]*)/[a-zA-Z0-9]*$', rel_path).group(1)
    if not directory_to_make:
        # Ya messed up real bad
        raise Exception("Invalid directory path")

    # Make the parent directories
    makedirs("../img/lab/%s" % directory_to_make, exist_ok=True)

    # Save the Lab image as a numpy array to preserve accuracy - Tensorflow will need to read in these images with numpy
    np.save("../img/lab/%s%s" % (rel_path, '.npy'), lab_image)


def hsi_transform(image):
    pass

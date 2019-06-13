import numpy as np
from skimage import color
from os.path import relpath
from os import makedirs
import re


def lab_transform(image):
    rgb_image = (list(image["georef_img"]["layers"]['visible']['array']))
    rgb_image = np.asarray(rgb_image)
    lab_image = color.rgb2lab(rgb_image)

    rel_path = relpath(image.filename)[6:-3]
    directory_to_make = re.search(r'^([0-9a-zA-Z/ ]*/[0-9a-zA-Z\-]*/[0-9a-zA-Z\-]*)/[a-zA-Z0-9]*$', rel_path).group(1)
    if not directory_to_make:
        raise Exception("Invalid directory path")
    makedirs("../img/lab/%s" % directory_to_make, exist_ok=True)

    # lab_scaled = (lab_image + [0, 128, 128]) / [100, 255, 255]
    np.save("../img/lab/%s%s" % (rel_path, '.npy'), lab_image)


def hsi_transform(image):
    pass

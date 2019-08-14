import numpy as np


def get_layers(im_transform, image, ndarray):
    """
    :param im_transform: list of transforms that needs to be done
    :param image: h5 image file
    :param ndarray: list containing numpy images
    :return: list containing numpy images
    """

    if 'rgb' in im_transform:
        im = rgb(image)
        ndarray.append(im)
    if 'nir' in im_transform:
        im = nir(image)
        ndarray.append(im)
    if 'ci' in im_transform:
        im = ci(image)
        ndarray.append(im)
    if 'dem' in im_transform:
        im = dem(image)
        ndarray.append(im)
    if 'reg' in im_transform:
        im = reg(image)
        ndarray.append(im)
    if 'ndvi' in im_transform:
        im = ndvi(image)
        ndarray.append(im)
    if 'red' in im_transform:
        im = red(image)
        ndarray.append(im)
    return ndarray


def rgb(image):
    """
    :param image: h5 image
    :return: numpy array
    """

    rgb_image = (list(image["georef_img"]["layers"]['visible']['array']))
    rgb_image = np.asarray(rgb_image)

    return rgb_image


def ci(image):
    """
    :param image: h5 image
    :return: numpy array
    """

    ci_image = (list(image["georef_img"]["layers"]['ci']['array']))
    np_ci = np.asarray(ci_image)

    return np_ci


def dem(image):
    """
    :param image: h5 image
    :return: numpy array
    """

    dem_image = (list(image["georef_img"]["layers"]['dem']['array']))
    np_dem = np.asarray(dem_image)

    return np_dem


def reg(image):
    """
    :param image: h5 image
    :return: numpy array
    """

    reg_image = (list(image["georef_img"]["layers"]['reg']['array']))
    np_reg = np.asarray(reg_image)

    return np_reg


def ndvi(image):
    """
    :param image: h5 image
    :return: numpy array
    """

    ndvi_image = (list(image["georef_img"]["layers"]['ndvi']['array']))
    np_ndvi = np.asarray(ndvi_image)

    return np_ndvi


def nir(image):
    """
    :param image: h5 image
    :return: numpy array
    """

    nir_image = (list(image["georef_img"]["layers"]['nir']['array']))
    np_nir = np.asarray(nir_image)

    return np_nir


def red(image):
    """
    :param image: h5 image
    :return: numpy array
    """

    red_image = (list(image["georef_img"]["layers"]['red']['array']))
    np_red = np.asarray(red_image)

    return np_red


def mask(image):
    """
    :param image: h5 image
    :return: numpy array
    """

    mask_image = (list(image["georef_img"]["layers"]['tree_global_mask']['array']))
    mask_image = np.asarray(mask_image)

    mask_image = mask_image + 1
    return mask_image

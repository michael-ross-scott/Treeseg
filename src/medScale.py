import cv2
import numpy as np
import pymeanshift as pms


def run_transform(im_transform, image, ndarray):
    """
    :param im_transform: list of transforms that needs to be done
    :param image: h5 image file
    :param ndarray: list containing numpy images
    :return: list containing numpy images
    """

    if 'hist_equal' in im_transform:
        im = hist_equal_transform(image)
        ndarray.append(im)
    if 'mean_shift' in im_transform:
        im = mean_shift_transform(image)
        ndarray.append(im)
    if 'edge_detector' in im_transform:
        im = edge_detector_transform(image)
        ndarray.append(im)
    if 'morph_closing' in im_transform:
        im = morph_closing_transform(image)
        ndarray.append(im)
    return ndarray


def mean_shift_transform(image):
    """
    :param image: h5 image
    :return: numpy array
    """

    ndvi_image = (list(image["georef_img"]["layers"]['visible']['array']))
    np_image = np.asarray(ndvi_image)

    (mean_shift_image, labels_image, number_regions) = pms.segment(np_image, spatial_radius=1, range_radius=1, min_density=300)
    return mean_shift_image


def morph_closing_transform(image):
    """
    :param image: h5 image
    :return: numpy array
    """

    nir_image = (list(image["georef_img"]["layers"]['ndvi']['array']))

    np_image = np.asarray(nir_image)

    # Produces binary image
    th = cv2.adaptiveThreshold(np_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 0)
    th1 = np.invert(th)

    # Removes noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 4))
    morph_img = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, kernel)

    return morph_img


def edge_detector_transform(image, sigma=0.33):
    """
    :param image: h5 image
    :param sigma: value to tweak mean_shift transform
    :return: numpy array
    """

    nir_image = (list(image["georef_img"]["layers"]['nir']['array']))
    np_image = np.asarray(nir_image)

    # compute the median of the single channel pixel intensities
    v = np.median(nir_image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    edge_detector_image = cv2.Canny(np_image,lower,upper)

    return edge_detector_image


def hist_equal_transform(image):
    """
    :param image: h5 image
    :return: numpy array
    """

    ndvi_image = (list(image["georef_img"]["layers"]['ndvi']['array']))
    np_image = np.asarray(ndvi_image)

    equal_img = cv2.equalizeHist(np_image)

    return equal_img


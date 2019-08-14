import numpy as np
from skimage import color
import cv2
from sklearn.decomposition import FastICA
import matplotlib
from matplotlib import colors


def run_transform(im_transform, image, ndarray):
    """
    :param im_transform: list of transforms that needs to be done
    :param image: h5 image file
    :param ndarray: list containing numpy images
    :return: list containing numpy images
    """

    if 'hsl' in im_transform:
        im = hsl_transform(image)
        ndarray.append(im)
    if 'lab' in im_transform:
        im = lab_transform(image)
        ndarray.append(im)
    if 'hsi' in im_transform:
        im = ica_transform(image)
        ndarray.append(im)
    if 'pca' in im_transform:
        im = pca_transform(image)
        ndarray.append(im)
    if 'ica' in im_transform:
        im = ica_transform(image)
        ndarray.append(im)
    return ndarray


def hsi_transform(image):
    """
    :param image: h5 image
    :return: numpy array
    """

    rgb_image = (list(image["georef_img"]["layers"]['visible']['array']))
    rgb_image = np.asarray(rgb_image)

    hsi_image = color.rgb2hsv(rgb_image)
    return hsi_image


def lab_transform(image):
    """
    :param image: h5 image
    :return: numpy array
    """

    rgb_image = (list(image["georef_img"]["layers"]['visible']['array']))

    rgb_image = np.asarray(rgb_image)
    lab_image = color.rgb2lab(rgb_image)

    return lab_image


def hsl_transform(image):
    """
    :param image: h5 image
    :return: numpy array
    """

    rgb_image = (list(image["georef_img"]["layers"]['visible']['array']))
    np_image = np.asarray(rgb_image)

    hsl_image = matplotlib.colors.rgb_to_hsv(np_image)
    return hsl_image


def ica_transform(image):
    """
    :param image: h5 image
    :return: numpy array
    """

    rgb_image = (list(image["georef_img"]["layers"]['visible']['array']))
    dem_image = (list(image["georef_img"]["layers"]['dem']['array']))

    np_image = np.array(rgb_image)
    np_image1 = np.asarray(dem_image)

    np_image = np.dstack((np_image, np_image1))

    gray_img = cv2.cvtColor(np_image, cv2.COLOR_BGRA2GRAY)

    Ica = FastICA(n_components=50)

    # Reconstruct image with independent components
    image_ica = Ica.fit_transform(gray_img)
    image_restored = Ica.inverse_transform(image_ica)

    return image_restored


def pca_transform(image):
    """
    :param image: h5 image
    :return: numpy array
    """

    rgb_image = (list(image["georef_img"]["layers"]['visible']['array']))

    np_image = np.array(rgb_image)
    img_r_orig, img_g_orig, img_b_orig = np_image[:, :, 0], np_image[:, :, 1], np_image[:, :, 2]

    # PCA of RGB separately
    img_r_pca, img_g_pca, img_b_pca = comp_2d(img_r_orig), comp_2d(img_g_orig), comp_2d(img_b_orig)

    # Combining rgb channels
    color_img = np.dstack((img_r_pca, img_g_pca, img_b_pca))
    return color_img


def comp_2d(image_2d):
    """
    :param image_2d: single channel numpy image
    :return: reduced colour numpy image
    """

    cov_mat = image_2d - np.mean(image_2d, axis=1)
    eig_val, eig_vec = np.linalg.eigh(np.cov(cov_mat))
    p = np.size(eig_vec, axis=1)
    idx = np.argsort(eig_val)
    idx = idx[::-1]
    eig_vec = eig_vec[:, idx]
    eig_val = eig_val[idx]
    numpc = 50

    if numpc < p or numpc > 0:
        eig_vec = eig_vec[:, range(numpc)]
    score = np.dot(eig_vec.T, cov_mat)

    # Normalise to make image quality better
    recon = np.dot(eig_vec, score) + np.mean(image_2d, axis=1).T

    # Controls eigenvalues
    recon_img_mat = np.uint8(np.absolute(recon))
    return recon_img_mat

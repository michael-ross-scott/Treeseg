import numpy as np
from skimage import color
from os.path import relpath
from os import makedirs
import re
import cv2
from sklearn.decomposition import FastICA
import matplotlib
from matplotlib import colors
from PIL import Image


def lab_transform(im_num, image):
    # Get the raw RGB values from the hdf5 image
    rgb_image = (list(image["georef_img"]["layers"]['visible']['array']))
    dem = (list(image["georef_img"]["layers"]['dem']['array']))

    rgb_image = np.asarray(rgb_image)
    dem_image = np.asarray(dem)

    # Convert to lab
    lab_image = color.rgb2lab(rgb_image)
    lab_dem_image = np.dstack((dem_image, lab_image))

    save_nmp_array(im_num, lab_dem_image, 'lab')


def mask(im_num,image):
    mask_image = (list(image["georef_img"]["layers"]['tree_global_mask']['array']))
    mask_image = np.asarray(mask_image)

    # Do this so we can see wtf we are doing
    mask_image = mask_image * 255

    save_nmp_array(im_num, mask_image, 'mask')


def hsi_transform(im_num, image):
    # Get the raw RGB values from the hdf5 image
    rgb_image = (list(image["georef_img"]["layers"]['visible']['array']))
    rgb_image = np.asarray(rgb_image)

    # Convert to HSI - Note this saves H as a fraction, instead of degrees
    hsi_image = color.rgb2hsv(rgb_image)
    save_nmp_array(im_num, hsi_image, 'hsi')
    
    
# Converts image to hsl colour space
def hsl_transform(im_num,image):
    rgb_image = (list(image["georef_img"]["layers"]['visible']['array']))
    np_image = np.asarray(rgb_image)

    # Convert to hsl
    hsl_image = matplotlib.colors.rgb_to_hsv(np_image)
    save_nmp_array(image, hsl_image, 'hsl')


# Performs independent component analysis on image
def ica_transform(im_num, image):
    rgb_image = (list(image["georef_img"]["layers"]['visible']['array']))
    np_image = np.array(rgb_image)

    gray_img = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)

    Ica = FastICA(n_components=100)

    # Reconstruct image with independent components
    image_ica = Ica.fit_transform(gray_img)
    image_restored = Ica.inverse_transform(image_ica)

    save_nmp_array(im_num, image_restored, 'ica')


# Performs principal component analysis on image
def pca_transform(im_num, image):
    rgb_image = (list(image["georef_img"]["layers"]['visible']['array']))

    np_image = np.array(rgb_image)
    img_r_orig, img_g_orig, img_b_orig = np_image[:, :, 0], np_image[:, :, 1], np_image[:, :, 2]

    # PCA of RGB separately
    img_r_pca, img_g_pca, img_b_pca = comp_2d(img_r_orig), comp_2d(img_g_orig), comp_2d(img_b_orig)

    # Combining rgb channels
    color_img = np.dstack((img_r_pca, img_g_pca, img_b_pca))
    save_nmp_array(im_num, color_img, 'pca')


# Performs principal component analysis on individual colour channels
def comp_2d(image_2d):
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


def save_nmp_array(im_num, new_image, folder):
    # Save the Lab image as a numpy array to preserve accuracy - Tensorflow will need to read in these images with numpy
    # We will also need a way of either saving the tree masks, or retrieving them from the original image
    cv2.imwrite("../img/%s/%s%s" % (folder, im_num, '.png'), new_image)

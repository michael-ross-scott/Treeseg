import pandas as pd
import h5py
import datetime
import cv2
import numpy as np
import os
from PIL import Image
import imageio
import random
from src import norm_layers, smallScale, medScale

# absolute path to the data
image_data_root = "/home/user/PycharmProjects/Apricot"
image_data_root1 = "/home/user/PycharmProjects/ApricotV2"
# image_data_root2 = "/home/user/PycharmProjects/Selected"
image_data_root2 = "/home/user/PycharmProjects/Selected001"


# The transform options, these are denoted in a space-separated string
'''
transforms options: 
    norm_layers = {ci, dem, ndvi, red, reg, mask, rgb}
    small_Scale = {hsi, lab, hsl, pca, ica}
    med_scale   = {mean_shift, hist_equal, edge_detector, morph_closing}
'''
transforms = "rgb mask"

# Keeps track of image names for the neural network
'''
Files:
    trainval: all image names are written to this file
    train   : writes seperate train and validation as well as trainval
    val:    : images that are for validation are written to this file
'''
train_val = open("trainval.txt", "w")
train = open("train.txt", "w")
val = open("val.txt", "w")

# Choose what type of files to write, one or the other
'''
File options:
    trainval: all image names are written to this file
    all     : writes seperate train and val files, as well as trainval
'''
train_files = "all"

# Choose training and evaluation split
'''
train_split: double ~ (0,1) specifies how much data to train and how much to test
'''
train_split = 0.7

# Choose save options for image array
'''
save options:
    gif: save as gif
    png: save as png
    npy: save as numpy array
'''
save = "png"

def main():

    print("Performing transforms for: %s" % transforms)

    start_time = datetime.datetime.now()

    image_path = get_image_paths(image_data_root)
    i = perform_transforms(image_path, image_data_root)

    image_path1 = get_image_paths(image_data_root1)
    i = perform_transforms(image_path1, image_data_root1, i)

    image_path2 = get_image_paths(image_data_root2)
    perform_transforms(image_path2, image_data_root2, i)


    if "all" in train_files:
        write_all(i)
    else:
        write_trainval(i)

    print("Time Taken: %ss" % (round((datetime.datetime.now() - start_time).total_seconds())))


def perform_transforms(image_paths, im_root, i=0):
    """
    :param image_paths: paths to all the images
    :param im_root: Place where hdf5 file resides
    :param i: keeps track of images that have been written
    :return: i
    """

    for im in image_paths:
        i += 1
        image = get_annotated_image(im, im_root)

        # List of numpy images
        array_of_images = []

        # Folder name to write our images
        scale = ""

        if 'rgb' in transforms or 'dem' in transforms or 'nir' in transforms or 'ndvi' in transforms or \
                'red' in transforms or 'reg' in transforms or 'ci' in transforms:
            array_of_images = norm_layers.get_layers(transforms, image, array_of_images)
            scale += "norm"

        if 'hsl' in transforms or 'hsi' in transforms or 'lab' in transforms or 'ica' in transforms or \
                'pca' in transforms:
            array_of_images = smallScale.run_transform(transforms, image, array_of_images)
            scale += "small"

        if 'hist_equal' in transforms or 'mean_shift' in transforms or 'morph_closing' in transforms or 'edge_detector'\
                in transforms:
            array_of_images = medScale.run_transform(transforms, image, array_of_images)
            scale += "med"

        if 'mask' in transforms:
            mask = norm_layers.mask(image)
            save_im(i, mask, "mask")

        if 'png' in save:
            nd_arr = rollup_images(array_of_images)
            save_im(i, nd_arr, scale)

        if 'gif' in save:
            save_gif(i, array_of_images, scale)

        if 'np' in save:
            nd_arr = rollup_images(array_of_images)
            save_nmp_array(i, nd_arr, scale)
    return i


def get_image_paths(im_root):
    """
    :param im_root: Place where hdf5 file resides
    :return: paths to all the images
    """

    print("Getting manifest.hd5 at: %s/manifest.md5" % im_root)

    manifest = pd.read_hdf("%s/manifest.h5" % im_root)
    image_paths = list(manifest["annotated_image_path"])
    print("Success, retrieved %d image paths" % len(image_paths))
    print("Layers included in image: %s" % list(get_annotated_image(image_paths[0], im_root)["georef_img"]["layers"]))

    test = list(get_annotated_image(image_paths[0], im_root)["georef_img"]["layers"]['visible']['array'])
    print("Dimensions: %s x %s" % (len(test[0]), len(test)))

    return image_paths


def get_annotated_image(im, im_root):
    """
    :param im: test h5 image
    :param im_root: paths to all images
    :return: h5py file
    """
    true_path = "%s/%s" % (im_root, im)
    return h5py.File(true_path, 'r')


def rollup_images(array_of_images):
    """
    :param array_of_images: list that contains individual numpy arrays
    :return: numpy ndstack
    """

    nd_arr = array_of_images[0]
    for i in range(1, len(array_of_images)):
        nd_arr = np.dstack((nd_arr, array_of_images[i]))
    return nd_arr


def save_im(im_num, new_image, folder):
    """
    :param im_num: filename
    :param new_image: numpy file to be written
    :param folder: folder where it needs to be written
    :return: None
    """

    print("Saving image", im_num, "to image path", "../img/%s/%s%s" % (folder, im_num, '.png'))
    cv2.imwrite("../img/%s/%s%s" % (folder, im_num, '.png'), new_image)


def save_gif(im_num, new_image, folder):
    """
    :param im_num: filename
    :param new_image: numpy file to be written
    :param folder: folder where it needs to be written
    :return: None
    """

    im_list = []
    for i in new_image:
        im = Image.fromarray(i)
        im_list.append(im)

    print("Saving image", im_num, "to image path", "../img/%s/%s%s" % (folder, im_num, '.gif'))

    im_path = "../img/" + folder + "/" + str(im_num) + ".gif"
    imageio.mimsave(im_path,im_list)
    os.system("convert " + im_path + " -coalesce " + im_path)


def save_nmp_array(im_num, new_image, folder):
    """
    :param im_num: filename
    :param new_image: numpy file to be written
    :param folder: folder where it needs to be written
    :return: None
    """

    print("Saving image", im_num, "to image path", "../img/%s/%s%s" % (folder, im_num, '.npy'))
    np.save("../img/%s/%s%s" % (folder, im_num, '.npy'), new_image)


def write_trainval(num_images):
    """
    :param num_images: number of image names to write
    :return: None
    """
    for i in range(1, num_images):
        train_val.write(str(i)+'\n')
    train_val.close()


def write_all(num_images):
    """
    :param num_images: number of image names to write
    :return: None
    """
    upper = int(num_images * train_split)
    im_list = []

    for i in range(0, num_images):
        im_list.append(i)

    random.shuffle(im_list)

    for i in range(1, upper):
        train.write(str(im_list[i]) + '\n')
    train.close()

    for i in range(upper + 1, num_images):
        val.write(str(im_list[i]) + "\n")
    val.close()

    write_trainval(num_images)


main()

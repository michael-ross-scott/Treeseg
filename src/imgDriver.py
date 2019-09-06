import pandas as pd
import h5py
import datetime
import numpy as np
from src import norm_layers, smallScale, medScale, writer

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
transforms = "rgb"


# Choose what type of files to write, one or the other
'''
File Save Options:
    "" : does not save a file
    trainval: all image names are written to this file
    all : writes separate train and val files, as well as trainval. This is unshuffled.
    rand : writes separate train and val files, as well as trainval. Randomly shuffled for evaluation
    fcn  : writes separate train and val files, as well as trainval. Taking every 10/100 images for evaluation 
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
    gif: Save transform as gif
    png: Save transform as png
    npy: Save transform as numpy array
    png_mask: Save the mask as a png 
    npy_mask: Save the mask as a separate npy array
'''
save = "npy npy_mask"


def main():

    print("Performing transforms for: %s" % transforms)

    start_time = datetime.datetime.now()

    # Folder name to write images
    folder = transforms.replace(" ", "_")
    writer.create_transform_dir(folder)

    image_path = get_image_paths(image_data_root)
    i = perform_transforms(image_path, image_data_root, folder)

    image_path1 = get_image_paths(image_data_root1)
    i = perform_transforms(image_path1, image_data_root1, folder, i)

    image_path2 = get_image_paths(image_data_root2)
    i = perform_transforms(image_path2, image_data_root2, folder, i)

    if "all" in train_files:
        writer.write_all(i, train_split)
    elif "rand" in train_files:
        writer.write_all_rand(i, train_split)
    elif "fcn" in train_files:
        writer.write_all_fcn(i, folder, "mask")
    elif "trainvail" in train_files:
        writer.write_trainval(i)
    print("Time Taken: %ss" % (round((datetime.datetime.now() - start_time).total_seconds())))


def perform_transforms(image_paths, im_root, folder, i=0):
    """
    :param image_paths: paths to all the images
    :param im_root: Place where hdf5 file resides
    :param i: keeps track of images that have been written
    :param folder: the place where the transforms will be written
    :return: i
    """

    for im in image_paths:
        i += 1
        image = get_annotated_image(im, im_root)

        # List of numpy images
        array_of_images = []

        if 'rgb' in transforms or 'dem' in transforms or 'nir' in transforms or 'ndvi' in transforms or \
                'red' in transforms or 'reg' in transforms or 'ci' in transforms:
            array_of_images = norm_layers.get_layers(transforms, image, array_of_images)

        if 'hsl' in transforms or 'hsi' in transforms or 'lab' in transforms or 'ica' in transforms or \
                'pca' in transforms:
            array_of_images = smallScale.run_transform(transforms, image, array_of_images)

        if 'hist_equal' in transforms or 'mean_shift' in transforms or 'morph_closing' in transforms or \
                'edge_detector' in transforms:
            array_of_images = medScale.run_transform(transforms, image, array_of_images)

        if 'png_mask' in save:
            mask = norm_layers.mask(image)
            writer.save_im(i, mask, "mask")

        if "npy_mask" in save:
            mask = norm_layers.np_mask(image)
            np.save("../img/%s/%s_mask%s" % ("mask", i, '.npy'), mask)

        if 'png' in save:
            nd_arr = rollup_images(array_of_images)
            writer.save_im(i, nd_arr, folder)

        if 'gif' in save:
            writer.save_gif(i, array_of_images, folder)

        if 'npy' in save:
            nd_arr = rollup_images(array_of_images)
            writer.save_nmp_array(i, nd_arr, folder)

    return i


def get_image_paths(im_root):
    """
    :param im_root: Place where hdf5 file resides
    :return: list of paths to all the h5 images
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

  
main()

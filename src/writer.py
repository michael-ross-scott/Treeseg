import cv2
from PIL import Image
import numpy as np
import imageio
import os
import random


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
    :return: outputs gif file of nd_arr
    """

    im_list = []
    for i in new_image:
        im = Image.fromarray(i)
        im_list.append(im)

    print("Saving image", im_num, "to image path", "../img/%s/%s%s" % (folder, im_num, '.gif'))
    imageio.mimsave("../img/%s/%s%s" % (folder, im_num, '.png'), new_image)


def save_nmp_array(im_num, new_image, folder):
    """
    :param im_num: filename
    :param new_image: numpy file to be written
    :param folder: folder where it needs to be written
    :return: outputs numpy file
    """

    print("Saving image", im_num, "to image path", "../img/%s/%s%s" % (folder, im_num, '.npy'))
    np.save("../img/%s/%s%s" % (folder, im_num, '.npy'), new_image)


def write_trainval(num_images, folder=os.getcwd()):
    """
    :param num_images: number of image names to write
    :param folder: path where image files are written
    :trainval: file used to keep track of all image paths
    :return: files with training and evaluation image names
    """
    train_val = open(folder + "trainval.txt", "w")

    for i in range(1, num_images + 1):
        train_val.write(str(i)+'\n')
    train_val.close()


def write_all(num_images, train_split, shuffle=True, folder=os.getcwd()):
    """
    :param num_images: number of image names to write
    :param train_split: percentage of data dedicated to training
    :param shuffle: boolean that decides whether training data needs to be randomly arranged
    :param folder: path where image files are written
    :train: file that keeps track of all the training image paths
    :val: file that keeps track of all the evaluation image paths
    :return: files with training and evaluation image names
    """

    train = open(folder + "train.txt", "w")
    val = open(folder + "val.txt", "w")

    upper = int(num_images * train_split)
    im_list = []

    for i in range(1, num_images + 1):
        print(i)
        im_list.append(i)

    if shuffle:
        random.shuffle(im_list)

    for i in range(0, upper + 1):
        print(i)
        train.write(str(im_list[i]) + '\n')
    train.close()

    for i in range(upper + 1, len(im_list)):
        print(i)
        val.write(str(im_list[i]) + "\n")
    val.close()

    write_trainval(num_images)

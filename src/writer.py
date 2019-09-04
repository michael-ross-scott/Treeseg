import cv2
from PIL import Image
import numpy as np
import imageio
import os
import random

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


def create_transform_dir(folder):
    """
    :param folder: the directory that image transforms will be written to
    :return: None
    """
    try:
        os.mkdir("../img/" + folder + "/")
    except:
        print("Directory already exists")


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
    imageio.mimsave(im_path, im_list)
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
    for i in range(1, num_images + 1):
        train_val.write(str(i)+'\n')
    train_val.close()


def write_all_rand(num_images, train_split):
    """
    :param num_images: number of image names to write
    :param train_split: split of images used to train/evaluate
    :return: None
    """

    upper = int(num_images * train_split)
    im_list = []

    for i in range(1, num_images + 1):
        print(i)
        im_list.append(i)

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


def write_all(num_images, train_split):
    """
    :param num_images: number of image names to write
    :param train_split: split of images used to train/evaluate
    :return: None
    """

    upper = int(num_images * train_split)
    im_list = []

    for i in range(1, num_images + 1):
        print(i)
        im_list.append(i)

    for i in range(0, upper + 1):
        print(i)
        train.write(str(im_list[i]) + '\n')
    train.close()

    for i in range(upper + 1, len(im_list)):
        print(i)
        val.write(str(im_list[i]) + "\n")
    val.close()

    write_trainval(num_images)


def write_all_fcn(num_images, transform, mask):
    """
    :param num_images: number of image names to write
    :param transform: path to the transform folder
    :param mask: path to the mask folder
    :return:
    """
    remainder = num_images % 100
    current = 1

    for i in range(1, num_images + 1):
        transform_path = "img/" + transform + "/" + i + ".png "
        mask_path = "img/" + mask + "/" + i + ".png\n"
        write_path = transform_path + mask_path
        if num_images-i >= 100:
            if current > (num_images * 0.1) / (num_images / 100):
                train.write(write_path)
            else:
                val.write(write_path)
        else:
            if current > remainder * 0.1:
                train.write(write_path)
            else:
                val.write(write_path)
        if current == 100:
            current = 1
        current += 1

        train.close()
        val.close()
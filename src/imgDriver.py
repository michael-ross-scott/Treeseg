import sys
import random
import pandas as pd
import h5py
import matplotlib.pyplot as plt


def main():
    global image_data_root
    image_data_root = sys.argv[1]
    print("Getting manifest.hd5 at: %s/manifest.md5" % image_data_root)

    manifest = pd.read_hdf("%s/manifest.h5" % image_data_root)
    image_paths = list(manifest["annotated_image_path"])
    print("Success, retrieved %d image paths" % len(image_paths))

    test = get_annotated_image(image_paths[0])


def get_annotated_image(image_path):
    true_path = "%s/%s" % (image_data_root, image_path)
    return h5py.File(true_path, 'r')

main()
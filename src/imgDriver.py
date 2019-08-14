import sys
import pandas as pd
import h5py
import datetime
import os

from src import smallScale, medScale

def main():
    dirpath = os.getcwd()
    trainfcn = os.path.join(dirpath[:-3], 'trainfcn.txt')

    f = open("trainval.txt", "w+")
    f2 = open(trainfcn, "w+")

    global image_data_root
    image_data_root = sys.argv[1]

    transforms = sys.argv[2]

    print("Performing transforms for: %s" % transforms)

    image_paths = get_image_paths()

    start_time = datetime.datetime.now()
    i = 0
    for image_path in image_paths:
        image = get_annotated_image(image_path)
        if 'lab' in transforms:
            smallScale.lab_transform(i, image, f2)
        if 'hsi' in transforms:
            smallScale.hsi_transform(i, image, f2)
        if 'hsl' in transforms:
            smallScale.hsl_transform(i, image, f2)
        if 'pca' in transforms:
            smallScale.pca_transform(i, image, f2)
        if 'ica' in transforms:
            smallScale.ica_transform(i, image, f2)
        if 'hist_equal' in transforms:
            medScale.hist_equal_transform(i, image, f2)
        if 'mean_shift' in transforms:
            medScale.mean_shift_transform(i, image, f2)
        if 'edge_detector' in transforms:
            medScale.edge_detector_transform(f2, i, image)
        if 'rgb' in transforms:
            medScale.rgb(f2, i, image)
        if 'mask' in transforms:
            smallScale.mask(i, image)
            f.write(str(i) + "\n")
        print("Completed image %d" % i)
        i += 1
    print("Time Taken: %ss" % (round((datetime.datetime.now() - start_time).total_seconds())))

def get_image_paths():
    print("Getting manifest.hd5 at: %s/manifest.md5" % image_data_root)

    manifest = pd.read_hdf("%s/manifest.h5" % image_data_root)
    image_paths = list(manifest["annotated_image_path"])
    print("Success, retrieved %d image paths" % len(image_paths))
    print("Layers included in image: %s" % list(get_annotated_image(image_paths[0])["georef_img"]["layers"]))

    test = list(get_annotated_image(image_paths[0])["georef_img"]["layers"]['visible']['array'])
    print("Dimensions: %s x %s" % (len(test[0]), len(test)))

    return image_paths

def get_annotated_image(image_path):
    true_path = "%s/%s" % (image_data_root, image_path)
    try:
        return h5py.File(true_path, 'r')
    except OSError:
            print("File not found, assuming removed: " + image_path)

main()
import sys
import pandas as pd
import h5py

from src import smallScale, medScale


def main():
    global image_data_root
    image_data_root = sys.argv[1]

    transforms = sys.argv[2]
    print("Performing transforms for: %s" % transforms)

    image_paths = get_image_paths()

    for image_path in image_paths:
        image = get_annotated_image(image_path)

        if 'lab' in transforms:
            smallScale.lab_transform(image)
        if 'hsi' in transforms:
            smallScale.hsi_transform(image)
        if 'hsl' in transforms:
            smallScale.hsl_transform(image)
        if 'pca' in transforms:
            smallScale.pca_transform(image)
        if 'ica' in transforms:
            smallScale.ica_transform(image)
        if 'hist_equal' in transforms:
            medScale.hist_equal_transform(image)
        if 'mean_shift' in transforms:
            medScale.mean_shift_transform(image)
        if 'edge_detector' in transforms:
            medScale.edge_detector_transform(image)


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
    return h5py.File(true_path, 'r')


main()

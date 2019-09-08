# Treeseg

## Small scale transforms

- Lab colorspace
- HSI colorspace
- HSL colorspace
- PCA on RGB
- ICA on RGB

Implemented using `skimage`, `OpenCV`, `Matplotlib` & `sklearn`; saved into a numpy array binary format for import into TensorFlow.

## Medium scale transforms

- Mean Shift
- Histogram Equalization
- Thresholding and Morphological closing
- Edge Detection

Implemented using `OpenCV` and [Mean Shift C++](https://github.com/fjean/pymeanshift)

## Prerequisites

To set up venv in Ubuntu:

`virtualenv venv && source venv/bin/activate && pip install -r requirements.txt`

### Pymeanshift Installation

0. If on windows download Visual Studio C++ build tools.

1.  The Python extension is compiled as follows: 
    * Linux and MacOS X (in a terminal window): bash cd path-to-pymeanshift-sources ./setup.py build 
    * Windows (at the command prompt): bash cd path-to-pymeanshift-sources python setup.py build

2.  The wrapper module and the extension can be installed as follows: * Systemwide, for all users (admin privileges are needed):
    * Linux and MacOS X (in a terminal window): sudo ./setup.py install
    * Windows (at the command prompt, from an admin account): python setup.py install

3.  If everything went fine, it should be possible to import the pymeanshift module in Python code. The module provides a function named segment and a class named Segmenter.

## Usage
Data to be processed should be next to this repo.

We assume data to be in `.h5` format, and a `manifest.h5` in the root of the source directory to enumerate 
filenames.

Output images will be stored in `img/`, keeping their original structure

### Running ImgDriver

Modify strings inputted into imgDriver to achieve desired functionality

**image_data_root:** path to data
</br>
**transforms:** Transforms or various image layers
</br>
**train_files:** Files to save paths of images written. Also used to denote which images are used to train Neural Networks
</br>
**transform_save_options:** File format to save transforms (PNG, npy or GIF)
</br>
**mask_save_options:** File format to save ground truth of images (PNG or npy)


## Troubleshooting
### Known Windows Issue 1:

If you run into this [issue](https://github.com/pandas-dev/pandas/issues/19666):

`pip uninstall h5py` 
</br>

`pip install h5py==2.8.0rc1`

### Known Windows Issue 2:

Windows works using `\` to denote paths, whilst Linux uses `/`. Convert these to suit your native file system

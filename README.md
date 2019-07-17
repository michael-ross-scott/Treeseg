# Treeseg

To set up venv in Ubuntu:

`virtualenv venv && source venv/bin/activate && pip install -r requirements.txt`

## Usage
Data to be processed should be next to this repo.

We assume data to be in `.h5` format, and a `manifest.h5` in the root of the source directory to enumerate 
filenames.

Assuming this repo is `/home/fergus/Treeseg`.

CLI arg1: Path to data, e.g. `"/home/fergus/apricot"`.

CLI arg2: Transforms to process, one or many of:

`lab hsi edge_detector mean_shift hsl hist_equal pca ica`

Output images will be stored in `img/`, keeping their original structure

## Small scale transforms

- Lab colorspace
- HSI colorspace

Implemented using `skimage`, saved into a numpy array binary format for import into TensorFlow.

## Troubleshooting

If you run into this [issue](https://github.com/pandas-dev/pandas/issues/19666):

`pip uninstall h5py` 
</br>

`pip install h5py==2.8.0rc1`

## Prerequisites
### Pymeanshift Installation

0. If on windows download Visual Studio C++ build tools.

1.  The Python extension is compiled as follows: 
    * Linux and MacOS X (in a terminal window): bash cd path-to-pymeanshift-sources ./setup.py build 
    * Windows (at the command prompt): bash cd path-to-pymeanshift-sources python setup.py build

2.  The wrapper module and the extension can be installed as follows: * Systemwide, for all users (admin privileges are needed):
    * Linux and MacOS X (in a terminal window): sudo ./setup.py install
    * Windows (at the command prompt, from an admin account): python setup.py install

3.  If everything went fine, it should be possible to import the pymeanshift module in Python code. The module provides a function named segment and a class named Segmenter.

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

`lab hsi edge_detector mean_shift hough template`

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
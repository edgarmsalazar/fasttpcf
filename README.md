# Fast TPCF

This code computes the two-point correlation function of tracer fields from cosmological simulations. Works as a wrapper to [`Corrfunc`](https://github.com/manodeep/Corrfunc) providing jackknife error estimations.

## Requirements
An `environment.yml` file is supplied as an example but feel free to use your own. 
```sh
$ conda env create -f environment.yml
```
Alternatively, create a conda environment including the following packages:
```sh
$ conda create --name <env name> python>=3.4 numpy cython mpi4py gcc gsl
```

### Installing `Corrfunc`
To compute correlation functions with multithreading, you'll need to install `corrfunc` from source. You can follow the installation process specified in the `corrfunc` repo to run tests and make sure everything is working as intended. If you wish to simply install the package do the following in another directory:
```sh
$ git clone https://github.com/manodeep/Corrfunc.git
$ cd Corrfunc
$ make
$ make install
```
Then install the package.
```sh
$ conda activate <env name>
$ python -m pip install .
```
I do recommend you run the tests as suggested in the [`corrfunc` package](https://github.com/manodeep/Corrfunc?tab=readme-ov-file#method-1-source-installation-recommended).

If you get an error where `crypt.h` could not be found, simply copy it into your environment.
```sh
$ cp /usr/include/crypt.h /home/<user>/miniconda/env/<env name>/include/python3.XX/
```

## Installation
Build from source
```sh
$ git clone https://github.com/edgarmsalazar/fasttpcf.git
$ cd fasttpcf
$ python -m pip install .
```

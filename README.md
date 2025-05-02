# Fast TPCF

This code computes the two-point correlation function of tracer fields from cosmological simulations. Uses [`Corrfunc`](https://github.com/manodeep/Corrfunc) for pair counting and provides jackknife error estimations. 

## Usage

You will probably only need two functions for most cases: 
- `cross_tpcf_jk`: computes the cross (or auto) correlation between fields given their X/Y/Z coordinates and weights (e.g. mass) for each field.
- `cross_tpcf_jk_radial`: computes the cross (or auto) correlation between fields given the X/Y/Z coordinates for field 1 and the radial distance $r$ of field 2 with respect to field 1 coordinates. Since this function was specifically written for halo-matter computations  it assumes same mass particles.

The `gridsize` parameter sets the side length of a volume partition in the same units as `boxsize`. The number of jackknife samples computed is `boxsize`//`gridsize` cubed.

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

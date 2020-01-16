#!/bin/bash
# run as "source build_tomopy.sh"
# First swap out IBM's XL compiler with GCC
module load gcc
# Next load the fftw module:
module load fftw
# This module will be necessary to install the netcdf4 python package later
module load netcdf
# The following two modules are needed for astra toolbox - GPU reconstructions
module load boost
module load cuda
# Finally, load and use the IBM WML CE module as the base
module load ibm-wml-ce/1.6.2-0
echo "Loaded all necessary modules"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
module list
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Now clone the environment
conda create --prefix /gpfs/alpine/stf011/world-shared/mcclure/mcclure-conda --clone ibm-wml-ce-1.6.2-0
# Activate clone:
conda activate /gpfs/alpine/stf011/world-shared/mcclure/mcclure-conda
# install cython - available via conda for power pc: https://anaconda.org/anaconda/cython
# Doing this early allows pip installation of other packages
conda install -y cython
# pywavelets came with the ibm module. No need to install
# install simple packages via pip:
pip install tifffile scikit-build cftime pyfftw
# dxchange does not have a pip installer nor power-pc builds on conda so install from source:
git clone https://github.com/data-exchange/dxchange.git ./packages/dxchange
cd ./packages/dxchange
python setup.py install
cd ../..
# Follow instructions from here - https://betterscientificsoftware.github.io/python-for-hpc/summit-mpi4py-note/ to install mpi4py from source
CC=mpicc 
MPICC=mpicc 
pip install mpi4py --no-binary mpi4py
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Install optional dependencies:
conda install -y astropy netcdf4
# dxfile cannot be installed via pip or conda. Install from source:
git clone https://github.com/data-exchange/dxfile ./packages/DXfile
cd packages/DXfile/
python setup.py install
cd ../..
# unable to find easy and simple installers for edffile, spefile, olefile
# Maybe these packages are not required after all
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Tomopy at talks about astra toolbox for GPU reconstruction
# Install ASTRA Toolbox based on instructions from here - https://www.astra-toolbox.com/docs/install.html#linux-from-source
git clone https://github.com/astra-toolbox/astra-toolbox.git ./packages/astra-toolbox
cd ./packages/astra-toolbox/build/linux/
./autogen.sh   # when building a git version
./configure --with-cuda=/sw/summit/cuda/10.1.168 --with-python --with-install-type=module
make
make install
cd ../../../..
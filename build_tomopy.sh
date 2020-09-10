#!/bin/bash
# run as "source build_tomopy.sh"
# First swap out IBM's XL compiler with GCC
module load gcc/6.4.0
# Next load the fftw module:
module load fftw/3.3.8
# This module will be necessary to install the netcdf4 python package later
module load netcdf/4.6.2
# The following two modules are needed for astra toolbox - GPU reconstructions
module load boost/1.66.0
module load cuda/10.1.168
# Load additional modules necessary for timemory and tomopy
module load cmake
# module load gperftools
# Finally, load and use the IBM WML CE module as the base
module load ibm-wml-ce/1.6.2-0
echo "Loaded all necessary modules"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
module list
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Basic folder management
CWD=$(pwd)
CONDA_NAME='mcclure-conda'
PACK_DIR='packages'
CONDA_DIR=$CWD/$CONDA_NAME
PACK_DIR=$CWD/$PACK_DIR
echo "Conda environment will be located at: $CONDA_DIR"
echo "Temporary directory for installing dependency packages will be located at $PACK_DIR"
if [ -d "$CONDA_DIR" ]; then echo "$CONDA_DIR already exists. Removing"; rm -rf $CONDA_DIR; fi
if [ -d "$PACK_DIR" ]; then echo "$PACK_DIR already exists. Removing"; rm -rf $PACK_DIR; fi
mkdir $PACK_DIR
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo ""
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Now clone the environment
conda create --prefix $CONDA_DIR --clone ibm-wml-ce-1.6.2-0
# Activate clone:
conda activate $CONDA_DIR
# Update conda if necessary:
# Skip since user does not have sufficient priveleges
# conda update -n base -c defaults conda -y
# Before installing packages via conda make sure to append channels:
# General:
conda config --append channels conda-forge
# IBM specific:
conda config --prepend channels \
https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda/
# install cython - available via conda for power pc: https://anaconda.org/anaconda/cython
# Doing this early allows pip installation of other packages
# cython on conda forge works for linux-ppc64le
# https://anaconda.org/conda-forge/cython/
conda install -y cython
# Install ipython for easier development and testing
conda install -y ipython
# pywavelets came with the ibm module. No need to install
# pyfftw does not come with a ppc installer in conda - https://anaconda.org/search?q=platform%3Alinux-ppc64le+pyfftw
# Not sure if pip will pick up on Summit's module:
pip install pyfftw
# install simple packages via pip:
pip install tifffile scikit-build cftime
# dxchange does not have a pip installer nor power-pc builds on conda so install from source:
git clone https://github.com/data-exchange/dxchange.git ${PACK_DIR}/dxchange
cd ${PACK_DIR}/dxchange
python setup.py install
cd $CWD
# Follow instructions from here - https://betterscientificsoftware.github.io/python-for-hpc/summit-mpi4py-note/ to install mpi4py from source
CC=mpicc 
MPICC=mpicc 
pip install mpi4py --no-binary mpi4py
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Install other dependencies:
# All these have ppc64le builds
# Note - numexpr has been a pain point in the past. Not sure if this is hte appropriate way to build
conda install -y astropy netcdf4 numexpr
# dxfile cannot be installed via pip or conda. Install from source:
git clone https://github.com/data-exchange/dxfile ${PACK_DIR}/DXfile
cd ${PACK_DIR}/DXfile/
python setup.py install
cd $CWD
# unable to find easy and simple installers for edffile, spefile, olefile
# Maybe these packages are not required after all
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Tomopy at talks about astra toolbox for GPU reconstruction
# Install ASTRA Toolbox based on instructions from here - https://www.astra-toolbox.com/docs/install.html#linux-from-source
# This takes a LONG time to build
git clone https://github.com/astra-toolbox/astra-toolbox.git ${PACK_DIR}/astra-toolbox
cd ${PACK_DIR}/astra-toolbox/build/linux/
./autogen.sh   # when building a git version
./configure --with-cuda=/sw/summit/cuda/10.1.168 --with-python --with-install-type=module
make
make install
cd $CWD
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Finally install Tomopy from custom fork
# Instructions for GPU build - https://tomopy.readthedocs.io/en/latest/gpu.html
git clone https://github.com/ssomnath/tomopy.git ${PACK_DIR}/tomopy
cd ${PACK_DIR}/tomopy
# python setup.py install
cd $CWD

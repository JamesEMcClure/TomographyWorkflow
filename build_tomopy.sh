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
mkdir $PACK_DIR
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Now clone the environment
conda create --prefix $CONDA_DIR --clone ibm-wml-ce-1.6.2-0
# Activate clone:
conda activate $CONDA_DIR
# install cython - available via conda for power pc: https://anaconda.org/anaconda/cython
# Doing this early allows pip installation of other packages
conda install -y cython
# pywavelets came with the ibm module. No need to install
# install simple packages via pip:
pip install tifffile scikit-build cftime pyfftw
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
# Install optional dependencies:
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
git clone https://github.com/astra-toolbox/astra-toolbox.git ${PACK_DIR}/astra-toolbox
cd ${PACK_DIR}/astra-toolbox/build/linux/
./autogen.sh   # when building a git version
./configure --with-cuda=/sw/summit/cuda/10.1.168 --with-python --with-install-type=module
make
make install
cd $CWD
# Install ipython for easier development and testing
conda install -y ipython
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Finally install Tomopy from custom fork
# Instructions for GPU build - https://tomopy.readthedocs.io/en/latest/gpu.html
git clone https://github.com/ssomnath/tomopy.git ${PACK_DIR}/tomopy
cd ${PACK_DIR}/tomopy
# python setup.py install
cd $CWD

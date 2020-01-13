"""
Get 2D slices of a reconstructed 3D volume stored in 'data_file'.

The slices are defined by the axis (x, y, or z) that they are perpendicular to
and a slice number that lies between 0 and max_number.
Max_number can be found in the print out if the script is run without a slice number.
Results are stored in the binary numpy format (.npy)

Usage:
python get_slices.py [data_file] [axis] [slice number]

Example:
python get_slices.py james.volume x 500
"""

import sys
import os.path
import numpy as np
from netCDF4 import Dataset

if __name__ == '__main__':

    # Open file with 
    dset = Dataset(sys.argv[1], 'r', format = 'NETCDF3_64BIT')
    volume = np.array(dset['VOLUME'])
    print('volume shape: ', volume.shape)

    # Normalization
    maxint = 2**15 - 1
    maxrange = np.max(np.abs(volume))
    print('maxrange (float32):', maxrange)

    scale = maxint/maxrange
    volume = (volume*scale).astype(np.int16)
    maxrange = np.max(np.abs(volume))
    print('maxrange (int16):', maxrange)

    # Output to netcdf file
    f_name = sys.argv[1][:-7] + '.nc'
    ncfile = Dataset(f_name, 'w', format = 'NETCDF3_64BIT', clobber = True)
    NX = ncfile.createDimension('NX', volume.shape[2])
    NY = ncfile.createDimension('NY', volume.shape[1])
    NZ = ncfile.createDimension('NZ', volume.shape[0])
    volume_out = ncfile.createVariable('VOLUME', 'i2', ('NZ','NY','NX'))
    volume_out[:] = volume
    ncfile.close()

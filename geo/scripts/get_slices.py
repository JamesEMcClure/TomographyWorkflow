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
import netCDF4

if __name__ == '__main__':

    # Open file with 
    dset = netCDF4.Dataset(sys.argv[1], 'r', format = 'NETCDF3_64BIT')
    #dset = netCDF4.Dataset(sys.argv[1], 'r', format = 'NETCDF3_CLASSIC')
    print(dset)
    print('dimensions: ', dset.dimensions)
    print('volume shape: ', dset['VOLUME'].shape)
    #print(dset.dimensions['NX'])

    # Select section of the 3d volume
    axdir = {'x':0, 'y':1, 'z':2}
    if len(sys.argv) >= 3:
        axis = sys.argv[2]
        if len(sys.argv) == 4:
            section = int(sys.argv[3])
        else:
            print(f'max_number along axis {axis}: ', dset['VOLUME'].shape[axdir[axis]])
    else:
        axis = 'x'
        section = 600

    if axis == 'x':
        plane = np.array(dset['VOLUME'][section, :, :])
    elif axis == 'y':
        plane = np.array(dset['VOLUME'][:, section, :])
    elif axis == 'z':
        plane = np.array(dset['VOLUME'][:, :, section])

    print(plane.shape)
    print(type(plane), type(plane[0,0]))

    # Output to numpy binary file
    f_name = sys.argv[1] +'_' + axis + str(section) + '.npy'
    np.save(f_name, plane)


"""
TomoPy example script to reconstruct the APS 13-BM tomography
data as original netcdf files. To use, change fname to just
the file name (e.g. 'sample[2].nc' would be 'sample'.
Reconstructed dataset will be saved as float32 netcdf3.
"""

import sys
import os.path
import re
import glob
import numpy as np
import tomopy as tp
#import dxchange as dx
#import fnmatch
#import logging
import dxchange.reader as dxreader
from netCDF4 import Dataset

def read_aps_13bm(fname, format, proj=None, sino=None):
    """
    Read APS 13-BM standard data format. Searches directory for all necessary
    files, and then combines the separate flat fields.

    Parameters
    ----------
    fname : str
        Path to hdf5 file.

    format : str
        Data format. 'spe' or 'netcdf4'

    proj : {sequence, int}, optional
        Specify projections to read. (start, end, step)

    sino : {sequence, int}, optional
        Specify sinograms to read. (start, end, step)

    Returns
    -------
    ndarray
        3D tomographic data.
    """

    if format == 'netcdf4':
        files = glob.glob(fname[0:-5] + '*[1-3].nc')
        files.sort()
        #print('files', files)
        tomo = dxreader.read_netcdf4(files[1], 'array_data', slc=(proj, sino))

        flat1 = dxreader.read_netcdf4(files[0], 'array_data', slc=(proj, sino))
        flat2 = dxreader.read_netcdf4(files[2], 'array_data', slc=(proj, sino))
        flat = np.concatenate((flat1, flat2), axis = 0)
        del flat1, flat2

        setup = glob.glob(fname[0:-5] + '*.setup')
        setup = open(setup[0], 'r')
        setup_data = setup.readlines()
        result = {}
        for line in setup_data:
            words = line[:-1].split(':',1)
            result[words[0].lower()] = words[1]

        dark = float(result['dark_current'])
        dark = flat*0+dark

        #theta = np.linspace(0.0, np.pi, tomo.shape[0])

        theta = dxreader.read_netcdf4(files[1], 'Attr_SampleOmega')[:]/180.0*np.pi

    return tomo, flat, dark, theta

if __name__ == '__main__':
    ## Set path (without file suffix) to the micro-CT data to reconstruct.
    fname = './data/bentonite/Bent2_under_op__E_'

    ## Import Data.
    proj, flat, dark, theta = read_aps_13bm(fname, format = 'netcdf4')
    print(type(proj), type(flat), type(dark), type(theta))
    print('a', proj.shape, flat.shape, dark.shape, theta.shape)
    #f_name = 'proj.npy'
    #np.save(f_name, np.array(proj))

    ## Flat-field correction of raw data.
    proj = tp.normalize(proj, flat = flat, dark = dark, ncore=4)
    print('b', proj.shape, flat.shape, dark.shape, theta.shape)
    #f_name = 'proj_norm.npy'
    #np.save(f_name, np.array(proj))
 
    ## Additional flat-field correction of raw data to negate need to mask.
    proj = tp.normalize_bg(proj, air = 10, ncore=4)
    #f_name = 'proj_norm_bg.npy'
    #np.save(f_name, np.array(proj))
    print('c', proj.shape, flat.shape, dark.shape, theta.shape)

    nostripe = tp.remove_stripe_based_sorting(proj)
    nostripe = tp.remove_stripe_based_filtering(nostripe)
    #nostripe = tp.remove_all_stripe(proj)
    f_name = 'nostripe_sort_filt.npy'
    np.save(f_name, np.array(nostripe))

    sys.exit()
 
    ## Set rotation center.
#    rot_center = tp.find_center_vo(proj)
#    #print('Center of rotation: ', rot_center)
# 
#    tp.minus_log(proj, out = proj)
# 
#    # Reconstruct object using Gridrec algorith.
#    rec = tp.recon(proj, theta, center = rot_center, sinogram_order = False, algorithm = 'gridrec', filter_name = 'hann', ncore=4)
#    rec = tp.remove_nan(rec)
# 
#    ## Writing data in netCDF3 .volume.
#    #ncfile = Dataset('filename.volume', 'w', format = 'NETCDF3_64BIT', clobber = True)
#    ncfile = Dataset('filename.volume', 'w', format = 'NETCDF3_CLASSIC', clobber = True)
#    NX = ncfile.createDimension('NX', rec.shape[2])
#    NY = ncfile.createDimension('NY', rec.shape[1])
#    NZ = ncfile.createDimension('NZ', rec.shape[0])
#    volume = ncfile.createVariable('VOLUME', 'f4', ('NZ','NY','NX'))
#    volume[:] = rec
#    ncfile.close()

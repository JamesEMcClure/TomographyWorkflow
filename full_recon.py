
"""
TomoPy example script to reconstruct the APS 13-BM tomography
data as original netcdf files. To use, change fname to just
the file name (e.g. 'sample[2].nc' would be 'sample'.
Reconstructed dataset will be saved as float32 netcdf3.
"""

import sys
import os.path
import re
import glob, logging
import numpy as np
import tomopy as tp
import logging
import dxchange.reader as dxreader
import dxchange
from netCDF4 import Dataset
import time

if __name__ == '__main__':
    ## Set path of one of the micro-CT data files to reconstruct.
    # fname = '/gpfs/alpine/world-shared/geo136/tomopy_example/data/E'
    # fname = '/gpfs/alpine/world-shared/geo136/Bent2/Bent2_under_op_/Bent2_under_op__ZZH_1.nc'
    fname = '/gpfs/alpine/world-shared/geo136/Bent2/Bent2_under_op_/Bent2_under_op__E_1.nc'
    
    # #################################################################
    
    cores=42
    
    t_start = time.time()

    ## Import Data.
    t0 = time.time()
    proj, flat, dark, theta = dxchange.read_aps_13bm(fname, file_format='netcdf4')
    
    print('Loaded data from GPFS to memory in {:.2f} sec'.format(time.time() - t0))
    
    for arr, arr_name in zip([proj, flat, dark, theta],
                             ['proj', 'flat', 'dark', 'theta']):
        print(arr_name + ' : {} {} : {:.2f} GB'.format(arr.dtype, arr.shape, arr.nbytes / (1024 ** 3)))
        
    print('\nNormalizing now...')

    ## Flat-field correction of raw data.
    t0 = time.time()
    proj = tp.normalize(proj, flat = flat, dark = dark, ncore=cores)
    
    print('Completed normalization in {:.2f} sec'.format(time.time() - t0))
    print('Projection after normalization:')
    print('{} {} : {:.2f} GB'.format(proj.dtype, proj.shape, proj.nbytes / (1024 ** 3)))
 
    ## Additional flat-field correction of raw data to negate need to mask.
    t0 = time.time()
    proj = tp.normalize_bg(proj, air = 10, ncore=cores)
    
    print('Flat-field correction performed in {:.2f} sec'.format(time.time() - t0))
    
    print('Projection after flat-field correction:')
    print('{} {} : {:.2f} GB'.format(proj.dtype, proj.shape, proj.nbytes / (1024 ** 3)))

    # Stripe removal using sorting method (Vo, 2018, Algorithm 3)
    # Turns out it also effectively removes zingers and specks (defect clusters)
    t0 = time.time()
    proj = tp.remove_stripe_based_sorting(proj)
    
    print('Removed stripes in {:.2f} sec'.format(time.time() - t0))
    
    print('Projection after stripe removal:')
    print('{} {} : {:.2f} GB'.format(proj.dtype, proj.shape, proj.nbytes / (1024 ** 3)))

    ## Set rotation center.
    t0 = time.time()
    rot_center = tp.find_center_vo(proj)
    #print('Center of rotation: ', rot_center)
    print('Rotated to find center in {:.2f} sec'.format(time.time() - t0))
    
    print('After rotation center:')
    print('{} {} : {:.2f} GB'.format(rot_center.dtype, rot_center.shape, rot_center.nbytes / (1024 ** 3)))
 
    t0 = time.time()
    tp.minus_log(proj, out = proj)
    
    print('Minus log completed in {:.2f} sec'.format(time.time() - t0))
    print('{} {} : {:.2f} GB'.format(proj.dtype, proj.shape, proj.nbytes / (1024 ** 3)))
 
    # Reconstruct object using Gridrec algorith.
    t0 = time.time()
    rec = tp.recon(proj, theta, center = rot_center, sinogram_order = False, algorithm = 'gridrec', filter_name = 'hann', ncore=cores)
    
    print('Reconstruction completed in {:.2f} sec'.format(time.time() - t0))
    print('{} {} : {:.2f} GB'.format(rec.dtype, rec.shape, rec.nbytes / (1024 ** 3)))
    
    t0 = time.time()
    rec = tp.remove_nan(rec)
    
    print('NaNs removed from reconstruction in {:.2f} sec'.format(time.time() - t0))
    print('{} {} : {:.2f} GB'.format(rec.dtype, rec.shape, rec.nbytes / (1024 ** 3)))
    print('Writing results to NetCDF:')
    
    t0=time.time() 
    ## Writing data in netCDF3 .volume.
    ncfile = Dataset('sort_rec_2.volume', 'w', format = 'NETCDF3_64BIT', clobber = True)
    NX = ncfile.createDimension('NX', rec.shape[2])
    NY = ncfile.createDimension('NY', rec.shape[1])
    NZ = ncfile.createDimension('NZ', rec.shape[0])
    volume = ncfile.createVariable('VOLUME', 'f4', ('NZ','NY','NX'))
    volume[:] = rec
    ncfile.close()
    print('Wrote to NetCDF in {:.2f} sec'.format(time.time() - t0))
    
    time_mins = (time.time() - t_start) / 60
    
    print('Total execution time: {:.2f} min'.format(time_mins))
          

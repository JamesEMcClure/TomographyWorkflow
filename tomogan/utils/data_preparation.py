"""
Prepare the train/test data for TomoGAN from 3D reconstructed volumes.

Requires:
    - numpy 
    - netCDF4
    - h5py

Usage:
python data_preparation.py [data_dir] [axis] 
"""

import sys,glob
import os.path
import numpy as np
import netCDF4
import h5py 
import argparse


args = argparse.ArgumentParser()
args.add_argument('--data-dir', default=None, type=str,
                  help='directory to reconstruction files')
args.add_argument('--axis', default='z', type=str,
                  help='axis for slicing: x, y, or z')
args.add_argument('--shuffle', default=False, action='store_true',
                  help='whether to shuffle')
args.add_argument('--infer', action='store_true', default=False,
                    help='whether to prepare for inference')
args.add_argument('--output', default='clean', type=str,
                  help='output file name')

def grayscale(d_img):
    np.nan_to_num(d_img, copy=False)
    _min, _max = np.percentile(d_img, 0.05), np.percentile(d_img, 99.95)
    d_img = d_img.clip(_min, _max)
    if _max == _min:
        d_img -= _max
    else:
        d_img = (d_img - _min) * 255. / (_max - _min)
    return d_img.astype('uint8')

if __name__ == '__main__':
   
    args = args.parse_args()
    axdir = {'x':0, 'y':1, 'z':2}
    datadir = args.data_dir
    axis = args.axis 
    h5f = args.output 
    shuffle = args.shuffle
    infer = args.infer 

    if axis not in axdir:
        print("Error: Specify axis (x, y, or z)")
        sys.exit()
    try:
        ncfiles = glob.glob(datadir+"/*.nc")
        ncfiles.sort()
        #vol = None
        with netCDF4.Dataset(ncfiles[0],'r') as header:
            xdim = header.dimensions['tomo_xdim'].size
            ydim = header.dimensions['tomo_ydim'].size 
            zdim = header.dimensions['tomo_zdim'].size 
            dtype = header['tomo'].dtype
        vol = np.zeros((zdim*len(ncfiles),ydim,xdim),dtype=dtype)
        for ncfile in ncfiles:
            dset = netCDF4.Dataset(ncfile, 'r')
            zdim = dset.getncattr('zdim_range')
            vol_part = np.array(dset['tomo'])
            vol[zdim[0]:zdim[1]+1,:,:] = vol_part
            #vol_part = np.array(dset['tomo'])
            #if vol is None: 
            #    vol = vol_part
            #else:
            #    vol = np.concatenate((vol,vol_part),axis=0)
        print('volume shape: ', vol.shape)
    except:
        print("can't find reconstruction files in %s"%datadir)
        sys.exit()
    
    if axis == 'x':
        planes = np.transpose(vol, (2,1,0))
    elif axis == 'y':
        planes = np.transpose(vol, (1,0,2))
    else:
        planes = vol 
    for i in range(planes.shape[0]):
        planes[i,:,:] = grayscale(planes[i,:,:])

    if shuffle: 
        np.random.seed(123) 
        permutation = np.random.permutation(planes.shape[0])
    train_size = int(0.8*planes.shape[0])
    if infer:
        with h5py.File(h5f+'.h5','w') as h5:
            h5['images'] = planes
    else: 
        with h5py.File(h5f+'_train.h5','w') as h5:
            if shuffle:
                h5['images'] = planes[permutation[:train_size],:,:] 
            else:
                h5['images'] = planes[:train_size,:,:]
        with h5py.File(h5f+'_test.h5','w') as h5:
            if shuffle: 
                h5['images'] = planes[permutation[train_size:],:,:]
            else:        
                h5['images'] = planes[train_size:,:,:]

  

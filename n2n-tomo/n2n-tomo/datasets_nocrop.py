import sys
import os
from glob import glob
import numpy as np
import torch
import netCDF4
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class TomoDataset(Dataset):
    """Make datasets from pairs of reconstructions of the same system"""

    #@profile
    def __init__(self, root_dir, axis=0, select=None):
        """Initializes dataset"""

        super(TomoDataset, self).__init__()

        if select is None:
            select = slice(None, None, None)

        self.root_dir = root_dir
        self.imgs = []

        files_A = glob(os.path.join(root_dir, '*_A.nc'))
        files_B = glob(os.path.join(root_dir, '*_B.nc'))
        files_A.sort()
        files_B.sort()

        assert len(files_A) == len(files_B), "Number of source and target files does not match"
        for fileA, fileB in zip(files_A, files_B):
            assert fileA[:-4] == fileB[:-4], f"filenames {fileA} and {fileB} do not match"

        for files in [files_A, files_B]:
            images = []
            for fname in files:

                #volume = np.load(fname)
                dset = netCDF4.Dataset(fname, 'r', format = 'NETCDF3_64BIT')

                if axis == 0:
                    vol_select = np.array(dset['VOLUME'][select, :, :])
                    vol_select = vol_select[:,None,:,:]
                elif axis == 1:
                    vol_select = np.array(dset['VOLUME'][:, select, :])
                    vol_select = vol_select.transpose(1, 0, 2)[:,None,:,:]
                elif axis == 2:
                    vol_select = np.array(dset['VOLUME'][:, :, select])
                    vol_select = vol_select.transpose(2, 1, 0)[:,None,:,:]

                dset.close()

                vol_select = vol_select.astype(np.float32) 
                vol_select /= 2**15  # normalize between +1 and -1
                print(np.max(vol_select), np.min(vol_select), vol_select.shape)
    
                images.append(torch.from_numpy(vol_select))


            self.imgs.append(torch.cat(images))

        assert len(self.imgs[0]) == len(self.imgs[1]), "Number or images in the source and target sets does not agree"


    def __getitem__(self, index):
        """Retrieves image and slice direction (target) from data folder"""
        return self.imgs[0][index], self.imgs[1][index]


    def __len__(self):
        """Returns length of dataset."""
        return len(self.imgs[0])


def load_dataset(root_dir, batch_size, axis=None, select=None, shuffle=False):
    """Loads dataset and returns corresponding data loader."""

    dataset = TomoDataset(root_dir, axis=0, select=select)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader

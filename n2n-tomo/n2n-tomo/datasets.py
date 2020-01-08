import sys
import os
from glob import glob
import numpy as np
import torch
import netCDF4
from torch.utils.data import Dataset, DataLoader, Sampler, SequentialSampler
from torchvision import transforms


class TomoDataset(Dataset):
    """Make datasets from pairs of reconstructions of the same system"""

    #@profile
    def __init__(self, root_dir, axis=0, crop_size=None, n_crops=1,
            n_height=0, n_width=0, select=None, render=False, seed=None):
        """Initializes dataset"""

        super(TomoDataset, self).__init__()

        if select is None:
            select = slice(None, None, None)

        self.root_dir = root_dir
        self.crop_size = crop_size
        self.n_crops = n_crops
        self.n_height = n_height
        self.n_width = n_width
        self.seed = seed
        self.imgs = [[], []]
        self.render = render

        files_A = glob(os.path.join(root_dir, '*_A.nc'))
        files_B = glob(os.path.join(root_dir, '*_B.nc'))
        files_A.sort()
        files_B.sort()

        assert len(files_A) == len(files_B), "Number of source and target files does not match"
        for fileA, fileB in zip(files_A, files_B):
            assert fileA[:-4] == fileB[:-4], f"filenames {fileA} and {fileB} do not match"

        for files in zip(files_A, files_B):

            self.crop_choice = None
            for i, fname in enumerate(files):

                dset = netCDF4.Dataset(fname, 'r', format = 'NETCDF3_64BIT')

                if axis == 0:
                    vol_select = np.array(dset['VOLUME'][select, :, :])
                elif axis == 1:
                    vol_select = np.array(dset['VOLUME'][:, select, :]).transpose(1, 0, 2)
                elif axis == 2:
                    vol_select = np.array(dset['VOLUME'][:, :, select]).transpose(2, 1, 0)

                dset.close()

                print('shape:', vol_select.shape)
                if crop_size is not None:
                    if render:
                        vol_select = self._covered_crop(vol_select)
                    else:
                        vol_select = self._random_crop(vol_select)

                vol_select = vol_select.astype(np.float32) 
                #vol_select /= 2**15  # normalize between +1 and -1
                vol_select += 2**15  # normalize between 0 and 1
                vol_select /= 2**16

                print('max-min-shape:', np.max(vol_select), np.min(vol_select), vol_select.shape)
    
                self.imgs[i].append(torch.from_numpy(vol_select[:,None,:,:]))

        assert len(self.imgs[0]) == len(self.imgs[1]), "Number or images in the source and target sets does not agree"

        self.imgs[0] = torch.cat(self.imgs[0])
        self.imgs[1] = torch.cat(self.imgs[1])

        print(self.imgs[0].shape, self.imgs[1].shape)

    def _covered_crop(self, imgs, seed=None):
        """Performs square crop of fixed size covering the image..
        """

        if self.crop_choice is None:
            n, w, h = imgs.shape
            assert w >= self.crop_size and h >= self.crop_size, \
                f'Error: Crop size: {self.crop_size}, Image size: ({w}, {h})'

            n_height = self.n_height
            n_width = self.n_width

            self.nh = (h - self.crop_size)//(n_height - 1)
            self.nw = (w - self.crop_size)//(n_width - 1)

            self.crop_choice = []
            for i in range(n_height):
                for j in range(n_width):
                    ih = i*self.nh
                    jw = j*self.nw
                    self.crop_choice.append((ih, jw))

        total_crops = self.n_height*self.n_width
        #print('whtot', self.n_height, self.n_width, total_crops)
        cropped_imgs = []
        cnt = 0
        for k, img in enumerate(imgs):
            #print('img', k)
            for ih, jw in self.crop_choice:
                cropped_imgs.append(img[ih:ih+self.crop_size, jw:jw+self.crop_size])
                #print('crop', ih, jw)
                cnt += 1

        print("total images:", cnt, self.nh, self.nw)
        coords_dir = os.path.dirname(self.root_dir)
        save_path = os.path.join(coords_dir, 'compose')
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        fname = os.path.join(save_path, 'coords')
        np.save(fname, np.array(self.crop_choice))

        return np.array(cropped_imgs)


    def _random_crop(self, imgs, seed=None):
        """Performs random square crop of fixed size.
        Works with list so that all items get the same cropped window (e.g. for buffers).
        """

        if self.seed:
            np.random.seed(self.seed)

        if self.crop_choice is None:
            n, w, h = imgs.shape
            assert w >= self.crop_size and h >= self.crop_size, \
                f'Error: Crop size: {self.crop_size}, Image size: ({w}, {h})'

            self.crop_choice = []
            for _ in range(n*self.n_crops):
                i = np.random.randint(0, h - self.crop_size + 1)
                j = np.random.randint(0, w - self.crop_size + 1)
                self.crop_choice.append((i, j))

        cropped_imgs = []
        for k, img in enumerate(imgs):
            for i, j in self.crop_choice[k*self.n_crops: (k+1)*self.n_crops]:
                cropped_imgs.append(img[i:i+self.crop_size, j:j+self.crop_size])

        return np.array(cropped_imgs)


    def __getitem__(self, index):
        """Retrieves image and slice direction (target) from data folder"""
        if index % 2 == 0:
            return self.imgs[0][index//2], self.imgs[1][index//2]
        else:
            return self.imgs[1][index//2], self.imgs[0][index//2]


    def __len__(self):
        """Returns length of dataset."""
        return len(self.imgs[0]) + len(self.imgs[1])


class RandomPairSampler(Sampler):
    r"""Takes random pairs of related (subsequent) samples.

    Useful for denoising and removing artifacts if different noisy instances of the
    same ground truth are available.

    Arguments
    ----------
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.num_samples = len(data_source)
        assert self.num_samples % 2 == 0, f"Even number of samples is required for RandomPairSampler"

        perm = torch.randperm(self.num_samples).tolist()
        perm = [i for i in perm if i % 2 == 0]
        self.perm = []
        for i in perm:
            self.perm.extend([i, i+1])
        
    def __iter__(self):
        return iter(self.perm)

    def __len__(self):
        return self.num_samples


def load_dataset(params, select=None, shuffle=False, sample_type='random_pairs'):
    """Loads dataset and returns corresponding data loader."""

    root_dir = params.data
    axis = params.axis
    crop_size = params.crop_size
    n_crops = params.n_crops
    n_width = params.n_width
    n_height = params.n_height
    batch_size = params.batch_size
    seed = params.seed
    render = params.render

    dataset = TomoDataset(root_dir, axis=axis, crop_size=crop_size,
            n_crops=n_crops, n_width=n_width, n_height=n_height,
            select=select, seed=seed, render=True)
    if sample_type == 'random_pairs':
        sampler = RandomPairSampler(dataset)
    elif sample_type == 'sequential':
        sampler = SequentialSampler(dataset)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler)

    return data_loader

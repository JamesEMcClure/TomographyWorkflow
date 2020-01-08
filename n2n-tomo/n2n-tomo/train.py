#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
from glob import glob
import netCDF4

import numpy as np
import torch
import torch.nn as nn

from datasets import load_dataset
from noise2noise import Noise2Noise
from argparse import ArgumentParser


def parse_args():
    """Command-line argument parser for training."""

    # New parser
    parser = ArgumentParser(description='Noise2Noise adapted to X-ray microtomography')

    # Data parameters
    parser.add_argument('-d', '--data', help='dataset root path', default='../data')
    parser.add_argument('-x', '--axis', help='Axis along which slices will be taken', type=int)
    parser.add_argument('-cs', '--crop-size', help='Size of the cropped image', type=int)
    parser.add_argument('-nc', '--n-crops', help='Number of random crops from a single image', type=int)
    parser.add_argument('-tf', '--train-fraction', help='Fraction of the train data', type=float)
    parser.add_argument('-vf', '--valid-fraction', help='Fraction of the validation data out of train data', type=float)
    parser.add_argument('-tn', '--train-number', help='Number of train samples', type=int)
    parser.add_argument('-vn', '--valid-number', help='Number of validation steps', type=int)

    # Training hyperparameters
    parser.add_argument('-lr', '--learning-rate', help='learning rate', default=0.001, type=float)
    parser.add_argument('-a', '--adam', help='adam parameters', nargs='+', default=[0.9, 0.99, 1e-8], type=list)
    parser.add_argument('-b', '--batch-size', help='minibatch size', default=4, type=int)
    parser.add_argument('-e', '--n-epochs', help='number of epochs', default=100, type=int)
    parser.add_argument('-l', '--loss', help='loss function', choices=['l1', 'l2'], default='l2', type=str)
    parser.add_argument('-s', '--seed', help='fix random seed', type=int)

    parser.add_argument('--cuda', help='use cuda', action='store_true')
    parser.add_argument('--plot-stats', help='plot stats after every epoch', action='store_true')
    parser.add_argument('--ckpt-save-path', help='checkpoint save path', default='./../ckpts')
    parser.add_argument('--ckpt-overwrite', help='overwrite model checkpoint on save', action='store_true')
    parser.add_argument('--report-interval', help='batch report interval', default=500, type=int)

    return parser.parse_args()

def train_valid_test_split(params):

    # Check the array size
    file_name = glob(os.path.join(params.data, '*_A.nc'))[0]
    print('filename', file_name)
    dset = netCDF4.Dataset(file_name, 'r', format = 'NETCDF3_64BIT')
    n_sample = dset['VOLUME'].shape[params.axis]
    dset.close()

    if params.train_number > 0:
        if params.train_number + params.valid_number > n_sample:
            raise ValueError(f"More training/valid samples requested than data ({n_sample})")
        train_cut = params.train_number
        valid_cut = params.valid_number
    else:
        train_cut = int(params.train_fraction*n_sample)
        valid_cut = int(params.valid_fraction*n_sample)

    test_cut = n_sample - train_cut - valid_cut
    print(n_sample, train_cut, valid_cut, test_cut)

    select = np.random.permutation(n_sample)
    select_train = select[:train_cut]
    select_test = select[-test_cut:]
    select_valid = select[train_cut:-test_cut]

    print('axis: ', params.axis, 'lenght', n_sample)

    return select_train, select_valid, select_test


if __name__ == '__main__':
    """Trains Noise2Noise."""

    # Parse training parameters
    params = parse_args()

    select_train, select_valid, select_test = train_valid_test_split(params)
    selects = {'axis':params.axis, 'train':select_train, 'valid':select_valid, 'test':select_test}
    np.savez(os.path.join(params.data, f"data_split.npz"), **selects)
    print('Lens:', len(select_train), len(select_valid), len(select_test))

    # Train/valid datasets
    train_loader = load_dataset(params, select=select_train, shuffle=False)
    valid_loader = load_dataset(params, select=select_valid, shuffle=False)

    # Initialize model and train
    n2n = Noise2Noise(params, trainable=True)
    n2n.train(train_loader, valid_loader)

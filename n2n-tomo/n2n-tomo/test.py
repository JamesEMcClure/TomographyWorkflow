#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import torch.nn as nn

from datasets import load_dataset
from noise2noise import Noise2Noise
from argparse import ArgumentParser


def parse_args():
    """Command-line argument parser for testing."""

    # New parser
    parser = ArgumentParser(description='Noise2Noise adapted to X-ray microtomography')

    # Parameters
    parser.add_argument('-d', '--data', help='dataset root path', default='../data')
    parser.add_argument('--load-ckpt', help='load model checkpoint')
    parser.add_argument('-x', '--axis', help='Axis along which slices will be taken', type=int)
    parser.add_argument('-cs', '--crop-size', help='Size of the cropped image', type=int)
    parser.add_argument('-nc', '--n-crops', help='Number of random crops from a single image', type=int)
    parser.add_argument('-b', '--batch-size', help='minibatch size', default=4, type=int)
    parser.add_argument('-s', '--seed', help='fix random seed', type=int)
    parser.add_argument('--cuda', help='use cuda', action='store_true')
    parser.add_argument('--show-output', help='pop up window to display outputs', default=0, type=int)

    return parser.parse_args()


if __name__ == '__main__':
    """Tests Noise2Noise."""

    # Parse test parameters
    params = parse_args()
    if params.crop_size == 0:
        params.crop_size = None

    # get test split information
    selects = np.load(os.path.join(params.data, 'data_split.npz'))
    params.axis = selects['axis']
    select_test = selects['test']

    # Initialize model and test
    n2n = Noise2Noise(params, trainable=False)
    test_loader = load_dataset(params, select=select_test, shuffle=True)
    n2n.load_model(params.load_ckpt)
    n2n.test(test_loader, show=params.show_output)

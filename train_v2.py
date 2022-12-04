#!/usr/bin/env python

'''
MIT License

Copyright (c) 2021 Stephen Hausler, Sourav Garg, Ming Xu, Michael Milford and Tobias Fischer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Significant parts of our code are based on [Nanne's pytorch-netvlad repository]
(https://github.com/Nanne/pytorch-NetVlad/), as well as some parts from the [Mapillary SLS repository]
(https://github.com/mapillary/mapillary_sls)

This code trains the NetVLAD neural network used to extract Patch-NetVLAD features.
'''


from __future__ import print_function

import argparse
import configparser
import os
import random
import shutil
from os.path import join, isfile
from os import makedirs
from datetime import datetime
import tempfile

import torch
import torch.nn as nn
import torch.optim as optim

import h5py

from tensorboardX import SummaryWriter
import numpy as np

from patchnetvlad.training_tools.train_epoch import train_epoch
from patchnetvlad.training_tools.val import val
from patchnetvlad.training_tools.get_clusters import get_clusters
from patchnetvlad.training_tools.tools import save_checkpoint
from patchnetvlad.tools.datasets import input_transform
from patchnetvlad.models.models_generic import get_backend, get_model
from patchnetvlad.tools import PATCHNETVLAD_ROOT_DIR

from tqdm.auto import trange

from patchnetvlad.training_tools.msls import MSLS

from custom.dataset import DatasetFactory
from custom.training_utils import perform_clustering, perform_training


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Patch-NetVLAD-train')

    parser.add_argument('--config_path', 
        type=str, 
        default=join(PATCHNETVLAD_ROOT_DIR, 'configs/train.ini'),
        help='File name (with extension) to an ini file with model config.',
    )
    parser.add_argument('--cache_path', 
        type=str, 
        default=tempfile.mkdtemp(),
        help='Path to save cache, centroid data to.',
    )
    parser.add_argument('--save_path', 
        type=str, 
        default='',
        help='Path to save checkpoints to'
    )
    parser.add_argument('--resume_path', 
        type=str, 
        default='',
        help='Full path and name (with extension) to load checkpoint from.',
    )
    parser.add_argument('--cluster_path', 
        type=str, 
        default='',
        help='Full path and name (with extension) to load cluster data from.'
    )
    parser.add_argument('--dataset_root_dir', 
        type=str, 
        default='/work/qvpr/data/raw/Mapillary_Street_Level_Sequences',
        help='Root directory of dataset',
    )
    parser.add_argument('--identifier', 
        type=str, 
        default='mapillary_nopanos',
        help='Description of this model, e.g. mapillary_nopanos_vgg16_netvlad'
    )
    parser.add_argument('--nEpochs', 
        type=int, 
        default=30, 
        help='number of epochs to train for'
    )
    parser.add_argument('--start_epoch', 
        default=0, 
        type=int, 
        metavar='N',
        help='manual epoch number (useful on restarts)'
    )
    parser.add_argument('--save_every_epoch', 
        action='store_true', 
        help='Flag to set a separate checkpoint file for each new epoch'
    )
    parser.add_argument('--threads', 
        type=int, 
        default=6, 
        help='Number of threads for each data loader to use'
    )
    parser.add_argument('--nocuda', 
        action='store_true', 
        help='If true, use CPU only. Else use GPU.'
    )


    opt = parser.parse_args()
    print(opt)

    configfile = opt.config_path
    assert os.path.isfile(configfile)
    config = configparser.ConfigParser()
    config.read(configfile)

    cuda = not opt.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")

    device = torch.device("cuda" if cuda else "cpu")

    random.seed(int(config['train']['seed']))
    np.random.seed(int(config['train']['seed']))
    torch.manual_seed(int(config['train']['seed']))
    if cuda:
        # noinspection PyUnresolvedReferences
        torch.cuda.manual_seed(int(config['train']['seed']))

    optimizer = None
    scheduler = None

    print('===> Building model')


    # TODO: Perform clustering
    prepare_model(opt, config)

    if config['train']['optim'] == 'ADAM':
        optimizer = optim.Adam(filter(lambda par: par.requires_grad,
            model.parameters()), lr=float(config['train']['lr'])
        )
    elif config['train']['optim'] == 'SGD':
        optimizer = optim.SGD(filter(lambda par: par.requires_grad,
            model.parameters()), lr=float(config['train']['lr']),
            momentum=float(config['train']['momentum']),
            weight_decay=float(config['train']['weightDecay'])
        )

        scheduler = optim.lr_scheduler.StepLR(optimizer, 
            step_size=int(config['train']['lrstep']),
            gamma=float(config['train']['lrgamma'])
        )
    else:
        raise ValueError('Unknown optimizer: ' + config['train']['optim'])

    criterion = nn.TripletMarginLoss(
        margin=float(config['train']['margin']) ** 0.5, p=2, 
        reduction='sum'
    ).to(device)

    model = model.to(device)

    if opt.resume_path:
        optimizer.load_state_dict(checkpoint['optimizer'])

    print('===> Loading dataset(s)')

    # TODO: Swap with my dataset
    train_dataset = MSLS(
        opt.dataset_root_dir, 
        mode='train', 
        nNeg=int(config['train']['nNeg']), 
        transform=input_transform(),
        bs=int(config['train']['cachebatchsize']), 
        threads=opt.threads, 
        margin=float(config['train']['margin']),
    )

    # TODO: Swap with my dataset
    validation_dataset = MSLS(
        opt.dataset_root_dir, 
        mode='val', 
        transform=input_transform(),
        bs=int(config['train']['cachebatchsize']), 
        threads=opt.threads,
        margin=float(config['train']['margin']), 
        posDistThr=25
    )

    print(train_dataset)
    print(validation_dataset)

    print('===> Training query set:', len(train_dataset.qIdx))
    print('===> Evaluating on val set, query count:', 
        len(validation_dataset.qIdx))
    print('===> Training model')
    writer = SummaryWriter(log_dir=join(opt.save_path, 
        datetime.now().strftime('%b%d_%H-%M-%S') + '_' + opt.identifier))

    # write checkpoints in logdir
    logdir = writer.file_writer.get_logdir()
    opt.save_file_path = join(logdir, 'checkpoints')
    makedirs(opt.save_file_path)

    # Do training
    do_training(train_dataset, validation_dataset, model, optimizer, 
        criterion, encoder_dim, device, epoch, opt, config, checkpoint, writer)

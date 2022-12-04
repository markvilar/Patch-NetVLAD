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


#import os
import random
import shutil
from os.path import join, isfile
from os import makedirs

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

def prepare_model(opt, config):
    """ """
    encoder_dim, encoder = get_backend()
    
    print("Encoder dim: {0}".format(type(encoder_dim)))
    print("Encoder:     {0}".format(type(encoder)))
    input()

    # if already started training earlier and continuing
    if opt.resume_path: 
        if isfile(opt.resume_path):
            print("=> loading checkpoint '{}'".format(opt.resume_path))
            checkpoint = torch.load(
                opt.resume_path, 
                map_location=lambda storage, 
                loc: storage
            )
            config['global_params']['num_clusters'] = \
                str(checkpoint['state_dict']['pool.centroids'].shape[0])

            model = get_model(encoder, encoder_dim, config['global_params'], 
                append_pca_layer=False)

            model.load_state_dict(checkpoint['state_dict'])
            opt.start_epoch = checkpoint['epoch']

            print("=> loaded checkpoint '{}'".format(opt.resume_path))
        else:
            raise FileNotFoundError("=> no checkpoint found at '{}'".format(
                opt.resume_path))
    # if not, assume fresh training instance and will initially generate 
    # cluster centroids
    else: 
        print('===> Loading model')
        config['global_params']['num_clusters'] = \
            config['train']['num_clusters']

        model = get_model(encoder, encoder_dim, config['global_params'], 
            append_pca_layer=False)

        initcache = join(opt.cache_path, 'centroids', 'vgg16_' + 'mapillary_' 
            + config['train']['num_clusters'] + '_desc_cen.hdf5')

        if opt.cluster_path:
            if isfile(opt.cluster_path):
                if opt.cluster_path != initcache:
                    shutil.copyfile(opt.cluster_path, initcache)
            else:
                raise FileNotFoundError(
                    "=> no cluster data found at '{}'".format(opt.cluster_path))
        else:
            print('===> Finding cluster centroids')

            print('===> Loading dataset(s) for clustering')

            # Load training data to calculate descriptors and clusters
            # TODO: Swap with my dataset.  
            train_dataset = MSLS(
                opt.dataset_root_dir, 
                mode='test', 
                cities='train', 
                transform=input_transform(),
                bs=int(config['train']['cachebatchsize']), 
                threads=opt.threads,
                margin=float(config['train']['margin'])
            )

            model = model.to(device)

            print('===> Calculating descriptors and clusters')
            get_clusters(train_dataset, model, encoder_dim, device, opt, 
                config)

            # a little hacky, but needed to easily run init_params
            model = model.to(device="cpu")

        with h5py.File(initcache, mode='r') as h5:
            clsts = h5.get("centroids")[...]
            traindescs = h5.get("descriptors")[...]
            model.pool.init_params(clsts, traindescs)
            del clsts, traindescs


def perform_training(train_dataset, validation_dataset, model, optimizer, 
    criterion, encoder_dim, device, epoch, opt, config, checkpoint, writer):
    """ Perform training and validation of model."""
    not_improved = 0
    best_score = 0
    if opt.resume_path:
        not_improved = checkpoint['not_improved']
        best_score = checkpoint['best_score']

    for epoch in trange(opt.start_epoch + 1, opt.nEpochs + 1, 
        desc='Epoch number'.rjust(15), position=0):

        train_epoch(train_dataset, model, optimizer, criterion, encoder_dim, 
            device, epoch, opt, config, writer)
        if scheduler is not None:
            scheduler.step(epoch)
        if (epoch % int(config['train']['evalevery'])) == 0:
            recalls = val(validation_dataset, model, encoder_dim, device, opt, 
                config, writer, epoch, write_tboard=True, pbar_position=1)
            is_best = recalls[5] > best_score
            if is_best:
                not_improved = 0
                best_score = recalls[5]
            else:
                not_improved += 1

            save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'recalls': recalls,
                    'best_score': best_score,
                    'not_improved': not_improved,
                    'optimizer': optimizer.state_dict(),
                    'parallel': False,
                }, 
                opt, 
                is_best
            )

            if int(config['train']['patience']) > 0 \
                and not_improved > (int(config['train']['patience']) \
                / int(config['train']['evalevery'])):
                print('Performance did not improve for', 
                    config['train']['patience'], 
                    'epochs. Stopping.')
                break

    print("=> Best Recall@5: {:.4f}".format(best_score), flush=True)
    writer.close()

    # garbage clean GPU memory, a bug can occur when Pytorch doesn't 
    # automatically clear the memory after runs
    torch.cuda.empty_cache()  

    print('Done')

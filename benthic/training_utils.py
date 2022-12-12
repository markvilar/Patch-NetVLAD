#!/usr/bin/env python

import os
import random
import shutil

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter

from patchnetvlad.training_tools.train_epoch import train_epoch
from patchnetvlad.training_tools.val import val
from patchnetvlad.training_tools.get_clusters import get_clusters
from patchnetvlad.training_tools.tools import save_checkpoint
from patchnetvlad.tools.datasets import input_transform
from patchnetvlad.models.models_generic import get_backend, get_model
from patchnetvlad.tools import PATCHNETVLAD_ROOT_DIR

from tqdm.auto import trange

from patchnetvlad.training_tools.msls import MSLS


def create_from_checkpoint(options, config):
    """ Loads an existing model from a checkpoint. """
    assert options.resume_path, "no resume path"
    assert os.path.isfile(options.resume_path), "invalid resume path"

    print("=> loading checkpoint {0}".format(options.resume_path))

    # Load checkpoint
    checkpoint = torch.load(options.resume_path, 
        map_location=lambda storage, loc: storage)

    # Remove PCA layers
    state_dict = checkpoint["state_dict"] # type = collections.OrderedDict
    remove_layers = ["WPCA.0.weight", "WPCA.0.bias"]
    for layer in remove_layers:
        if layer in state_dict:
            del state_dict[layer]
            print("Removed layer: {0}".format(layer))

    # Add checkpoint parameters to config
    config['global_params']['num_clusters'] = \
        str(checkpoint['state_dict']['pool.centroids'].shape[0])

    # Create model
    encoder_dim, encoder = get_backend()
    model = get_model(encoder, encoder_dim, config['global_params'], 
        append_pca_layer=False)

    model.load_state_dict(checkpoint['state_dict'])
    if "epoch" in checkpoint:
        options.start_epoch = checkpoint['epoch']

    print("=> loaded checkpoint {0}".format(options.resume_path))
    return model


def create_from_clusters(options, config):
    """ Creates a new model with"""
    print('===> Loading model')
    config['global_params']['num_clusters'] = config['train']['num_clusters']

    # Create model
    encoder_dim, encoder = get_backend()
    model = get_model(encoder, encoder_dim, config['global_params'], 
        append_pca_layer=False)

    # Create descriptor file
    descriptor_file = "centroids", "vgg16_" + "mapillary_" \
        + config["train"]["num_clusters"] + "_desc_cen.hdf5"
    descriptor_cache = os.path.join(options.cache_path, descriptor_file)

    # Load clusters from file
    assert options.cluster_path
    assert os.path.isfile(options.cluster_path)
    assert options.cluster_path != descriptor_cache
    
    # Copy clusters to cache
    shutil.copyfile(options.cluster_path, descriptor_cache)

    # Return to CPU for initialization
    model = model.to(device="cpu")

    # Initialize model with centroids and descriptors from training data
    with h5py.File(descriptor_cache, mode='r') as h5:
        clusters = h5.get("centroids")[...]
        descriptors = h5.get("descriptors")[...]
        model.pool.init_params(clusters, descriptors)
        del clusters, descriptors

    return model


def create_from_scratch(options, config, dataset, device):
    """ """
    print('===> Finding cluster centroids')
    print('===> Loading dataset(s) for clustering')

    # Create encoder and model
    encoder_dim, encoder = get_backend()
    model = get_model(encoder, encoder_dim, config['global_params'], 
        append_pca_layer=False)

    model = model.to(device)
    
    # Create descriptor file
    descriptor_file = "centroids", "vgg16_" + "mapillary_" \
        + config["train"]["num_clusters"] + "_desc_cen.hdf5"
    descriptor_cache = os.path.join(options.cache_path, descriptor_file)


    print('===> Calculating descriptors and clusters')
    get_clusters(dataset, model, encoder_dim, device, options, config)

    # Return to CPU for initialization
    model = model.to(device="cpu")
    
    # Initialize model with centroids and descriptors from training data
    with h5py.File(descriptor_cache, mode='r') as h5:
        clusters = h5.get("centroids")[...]
        descriptors = h5.get("descriptors")[...]
        model.pool.init_params(clusters, descriptors)
        del clusters, descriptors

    return model


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

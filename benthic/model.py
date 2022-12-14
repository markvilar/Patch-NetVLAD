#!/usr/bin/env python

import os
import shutil

import h5py
import torch

from patchnetvlad.training_tools.get_clusters import get_clusters
from patchnetvlad.models.models_generic import get_backend, get_model


def create_from_checkpoint(options, config):
    """ Loads an existing model from a checkpoint. """
    assert options.resume_path, "no resume path"
    assert os.path.isfile(options.resume_path), "invalid resume path"

    print("=> loading checkpoint {0}".format(options.resume_path))

    # Load checkpoint
    checkpoint = torch.load(options.resume_path, 
        map_location=lambda storage, loc: storage)

    # Remove PCA layers
    state_dict = checkpoint["state_dict"]
    remove_layers = ["WPCA.0.weight", "WPCA.0.bias"]
    for layer in remove_layers:
        if layer in state_dict:
            del state_dict[layer]
            print("Removed layer: {0}".format(layer))

    # Add checkpoint parameters to config
    config["global_params"]["num_clusters"] = \
        str(checkpoint["state_dict"]["pool.centroids"].shape[0])

    # Create model
    encoder_dim, encoder = get_backend()
    model = get_model(encoder, encoder_dim, config["global_params"], 
        append_pca_layer=False)

    model.load_state_dict(checkpoint["state_dict"])
    if "epoch" in checkpoint:
        options.start_epoch = checkpoint["epoch"]

    print("=> loaded checkpoint {0}".format(options.resume_path))
    return model


def create_from_clusters(options, config):
    """ Creates a new model with"""
    print("===> Loading model")
    config["global_params"]["num_clusters"] = config["train"]["num_clusters"]

    # Create model
    encoder_dim, encoder = get_backend()
    model = get_model(encoder, encoder_dim, config["global_params"], 
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
    with h5py.File(descriptor_cache, mode="r") as h5:
        clusters = h5.get("centroids")[...]
        descriptors = h5.get("descriptors")[...]
        model.pool.init_params(clusters, descriptors)
        del clusters, descriptors

    return model


def create_from_scratch(options, config, dataset, device):
    """ """
    print("===> Finding cluster centroids")
    print("===> Loading dataset(s) for clustering")

    # Create encoder and model
    encoder_dim, encoder = get_backend()
    model = get_model(encoder, encoder_dim, config["global_params"], 
        append_pca_layer=False)

    model = model.to(device)
    
    # Create descriptor file
    descriptor_file = "centroids", "vgg16_" + "mapillary_" \
        + config["train"]["num_clusters"] + "_desc_cen.hdf5"
    descriptor_cache = os.path.join(options.cache_path, descriptor_file)


    print("===> Calculating descriptors and clusters")
    get_clusters(dataset, model, encoder_dim, device, options, config)

    # Return to CPU for initialization
    model = model.to(device="cpu")
    
    # Initialize model with centroids and descriptors from training data
    with h5py.File(descriptor_cache, mode="r") as h5:
        clusters = h5.get("centroids")[...]
        descriptors = h5.get("descriptors")[...]
        model.pool.init_params(clusters, descriptors)
        del clusters, descriptors

    return model

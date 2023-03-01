#!/usr/bin/env python

from __future__ import print_function

import argparse
import configparser
import os
import random
import tempfile

from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from tensorboardX import SummaryWriter

from patchnetvlad.models.models_generic import get_backend
from patchnetvlad.tools import PATCHNETVLAD_ROOT_DIR

from benthic.dataset import BenthicDatasetFactory
from benthic.evaluate import evaluate
from benthic.model import (
    create_from_checkpoint, 
    create_from_clusters, 
    create_from_scratch,
)


def prepare_arguments(description: str):
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("--config_path", 
        type=str, 
        default=os.path.join(PATCHNETVLAD_ROOT_DIR, "configs/train.ini"),
        help="File name (with extension) to an ini file with model config.",
    )
    parser.add_argument("--data_path", 
        type=Path,
        required=True,
        help="config file path with image directory, query, and index paths",
    )
    parser.add_argument("--save_path", 
        type=str, 
        default="",
        help="Path to save checkpoints to"
    )
    parser.add_argument("--resume_path", 
        type=str, 
        required=True,
        help="Full path and name (with extension) to load checkpoint from.",
    )
    parser.add_argument("--identifier", 
        type=str, 
        default="",
        help="Description of this model, e.g. mapillary_nopanos_vgg16_netvlad"
    )
    parser.add_argument("--threads", 
        type=int, 
        default=6, 
        help="Number of threads for each data loader to use"
    )
    parser.add_argument("--nocuda", 
        action="store_true", 
        help="If true, use CPU only. Else use GPU."
    )

    return parser


def load_configuration(path: Path):
    assert os.path.isfile(path)
    config = configparser.ConfigParser()
    config.read(path)
    return config


def create_writer(options):
    time_string = datetime.now().strftime("%Y%m%d_%H%M%S")
    if options.identifier:
        filename = time_string + "_" + options.identifier
    else:
        filename = time_string

    writer = SummaryWriter(log_dir=os.path.join(options.save_path, filename))

    # write checkpoints in logdir
    logdir = writer.file_writer.get_logdir()
    options.save_file_path = os.path.join(logdir, "checkpoints")
    os.makedirs(options.save_file_path)
    return writer


def main():
    # Set up command-line arguments
    parser = prepare_arguments("evaluate patchnetvlad benthic")
    options = parser.parse_args()
    
    print(options)

    # Load configuration
    config = load_configuration(options.config_path)

    # Load data
    data = load_configuration(options.data_path)

    # Decide cuda and device
    cuda = not options.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")
    device = torch.device("cuda" if cuda else "cpu")

    # Create dataset factory
    dataset_factory = BenthicDatasetFactory(
            Path(data["paths"]["image"]),
            Path(data["paths"]["query"]),
            Path(data["paths"]["index"]),
            threshold_pos = float(config["data"]["dist_positive"]),
            threshold_neg = float(config["data"]["dist_negative"]),
            altitude_low = float(data["navigation"]["altitude_low"]),
            altitude_high = float(data["navigation"]["altitude_high"])
        )

    # Load / create model
    model = None
    
    model = create_from_checkpoint(options, config)

    assert model, "no model created / loaded"

    # Create datasets
    print("===> Loading dataset(s)")
    _, test_set = dataset_factory.create_training_data(0.99)

    print("===> Test set: {0}".format(len(test_set.query)))
    print("===> Evaluating model")

    # Set up Tensorboard writer
    writer = create_writer(options)

    # Test model
    encoder_dim, _ = get_backend()
    evaluate(test_set, model, encoder_dim, device, options, config, writer)

if __name__ == "__main__":
    main()

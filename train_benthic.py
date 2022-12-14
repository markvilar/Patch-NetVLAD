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
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter

from patchnetvlad.tools import PATCHNETVLAD_ROOT_DIR

from benthic.dataset import BenthicDatasetFactory
from benthic.model import (
    create_from_checkpoint, 
    create_from_clusters, 
    create_from_scratch,
)
from benthic.trainer import ModelTrainer
from benthic.training_utils import train_model


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
    parser.add_argument("--cache_path", 
        type=str, 
        default=tempfile.mkdtemp(),
        help="Path to save cache, centroid data to.",
    )
    parser.add_argument("--save_path", 
        type=str, 
        default="",
        help="Path to save checkpoints to"
    )
    parser.add_argument("--resume_path", 
        type=str, 
        default="",
        help="Full path and name (with extension) to load checkpoint from.",
    )
    parser.add_argument("--cluster_path", 
        type=str, 
        default="",
        help="Full path and name (with extension) to load cluster data from."
    )
    parser.add_argument("--identifier", 
        type=str, 
        default="",
        help="Description of this model, e.g. mapillary_nopanos_vgg16_netvlad"
    )
    parser.add_argument("--epochs", 
        type=int, 
        default=30, 
        help="number of epochs to train for"
    )
    """
    parser.add_argument("--start_epoch", 
        default=0, 
        type=int, 
        metavar="N",
        help="manual epoch number (useful on restarts)"
    )
    """
    parser.add_argument("--save_every_epoch", 
        action="store_true", 
        help="Flag to set a separate checkpoint file for each new epoch"
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


def set_seeds(seed: int, cuda: bool):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


def create_trainer(model, config, device) -> ModelTrainer:
    optimizer, scheduler = None, None
    if config["train"]["optim"] == "ADAM":
        optimizer = optim.Adam(filter(lambda par: par.requires_grad,
            model.parameters()), lr=float(config["train"]["lr"])
        )
    elif config["train"]["optim"] == "SGD":
        optimizer = optim.SGD(filter(lambda par: par.requires_grad,
            model.parameters()), lr=float(config["train"]["lr"]),
            momentum=float(config["train"]["momentum"]),
            weight_decay=float(config["train"]["weightDecay"])
        )

        scheduler = optim.lr_scheduler.StepLR(optimizer, 
            step_size=int(config["train"]["lrstep"]),
            gamma=float(config["train"]["lrgamma"])
        )
    else:
        raise ValueError("Unknown optimizer: " + config["train"]["optim"])

    criterion = nn.TripletMarginLoss(
            margin=float(config["train"]["margin"]) ** 0.5, 
            p=2,
            reduction="sum"
        ).to(device)

    batch_size = int(config["train"]["batchsize"])

    trainer = ModelTrainer(
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            batch_size=batch_size,
        )

    return trainer


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
    parser = prepare_arguments("train patchnetvlad benthic")
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

    # Set seeds
    set_seeds(int(config["train"]["seed"]), cuda)

    # Create dataset factory
    dataset_factory = BenthicDatasetFactory(
            Path(data["paths"]["image"]),
            Path(data["paths"]["query"]),
            Path(data["paths"]["index"]),
            threshold_pos = float(config["train"]["dist_positive"]),
            threshold_neg = float(config["train"]["dist_negative"]),
            altitude_low = float(config["train"]["altitude_low"]),
            altitude_high = float(config["train"]["altitude_high"])
        )

    # Load / create model
    model = None
    if options.resume_path:
        model = create_from_checkpoint(options, config)
    elif options.cluster_path:
        # TODO: Debug.
        model = create_from_clusters(options, config)
    else:
        # TODO: Need to integrate with Benthic dataset
        model = create_from_scratch(options, config, dataset, device)

    assert model, "no model created / loaded"

    # Prepare trainer (optimizer, scheduler, criterion)
    trainer = create_trainer(model, config, device)

    # Create datasets
    print("===> Loading dataset(s)")
    training_set, validation_set = dataset_factory.create_training_data(0.25)

    print("===> Train. query set: {0}".format(len(training_set.query)))
    print("===> Valid. query set: {0}".format(len(validation_set.query)))
    print("===> Training model")

    # Set up Tensorboard writer
    writer = create_writer(options)

    # Train model
    train_model(training_set, validation_set, model, trainer, options, config, 
        writer)

if __name__ == "__main__":
    main()

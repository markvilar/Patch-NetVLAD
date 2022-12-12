#!/usr/bin/env python

from __future__ import print_function

import argparse
import configparser
import os
import random
import tempfile

from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

# NOTE: Might remove?
# from patchnetvlad.training_tools.train_epoch import train_epoch
# from patchnetvlad.training_tools.val import val  
# from patchnetvlad.training_tools.tools import save_checkpoint

from patchnetvlad.tools.datasets import input_transform
from patchnetvlad.models.models_generic import get_backend, get_model
from patchnetvlad.tools import PATCHNETVLAD_ROOT_DIR

from tqdm.auto import trange

from benthic.dataset import BenthicDataset, BenthicDatasetFactory
from benthic.training_utils import (
    create_from_checkpoint,
    create_from_clusters,
    create_from_scratch,
    perform_training
)

def prepare_arguments(description: str):
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--config_path', 
        type=str, 
        default=os.path.join(PATCHNETVLAD_ROOT_DIR, 'configs/train.ini'),
        help='File name (with extension) to an ini file with model config.',
    )
    parser.add_argument("--data_path", 
        type=Path,
        required=True,
        help="config file path with image directory, query, and index paths",
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
    parser.add_argument('--identifier', 
        type=str, 
        default='',
        help='Description of this model, e.g. mapillary_nopanos_vgg16_netvlad'
    )
    parser.add_argument('--epochs', 
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


def prepare_optimizer(model, config):
    optimizer, scheduler = None, None
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

    return optimizer, scheduler


def create_loss_criterion(config, device):
    criterion = nn.TripletMarginLoss(
        margin=float(config["train"]["margin"]) ** 0.5, 
        p=2,
        reduction="sum"
    ).to(device)

    return criterion


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

    dataset_factory = BenthicDatasetFactory(
            Path(data["paths"]["image"]),
            Path(data["paths"]["query"]),
            Path(data["paths"]["index"]),
            threshold_pos = float(config["train"]["dist_positive"]),
            threshold_neg = float(config["train"]["dist_negative"]),
            transform = input_transform,
        )

    # Load / create model
    model = None
    if options.resume_path:
        model = create_from_checkpoint(options, config)
    elif options.cluster_path: # TODO: Figure out what conditions triggers this
        model = create_from_clusters(options, config)
    else:
        # TODO: Need to integrate with Benthic dataset
        model = create_from_scratch(options, config, dataset, device)

    assert model, "no model created / loaded"

    # Prepare optimizer and scheduler   
    optimizer, scheduler = prepare_optimizer(model, config)

    # Prepare criterion
    criterion = create_loss_criterion(config, device)

    # Transfer model to device
    model = model.to(device)

    # TODO: Skip?
    """
    if options.resume_path:
        checkpoint = torch.load(options.resume_path, 
            map_location=lambda storage, loc: storage)
        optimizer.load_state_dict(checkpoint["optimizer"])
    """
    
    print('===> Loading dataset(s)')

    # TODO: Create training and validation dataset
    training, validation = dataset_factory.create_training(0.25)


    """
    train_dataset = MSLS(
        opt.dataset_root_dir, 
        mode='train', 
        nNeg=int(config['train']['nNeg']), 
        transform=input_transform(),
        bs=int(config['train']['cachebatchsize']), 
        threads=opt.threads, 
        margin=float(config['train']['margin']),
    )
    """

    # TODO: Swap with my dataset
    """
    validation_dataset = MSLS(
        opt.dataset_root_dir, 
        mode='val', 
        transform=input_transform(),
        bs=int(config['train']['cachebatchsize']), 
        threads=opt.threads,
        margin=float(config['train']['margin']), 
        posDistThr=25
    )
    """
    
    """
    print(train_dataset)
    print(validation_dataset)

    print('===> Training query set:', len(train_dataset.qIdx))
    print('===> Evaluating on val set, query count:', 
        len(validation_dataset.qIdx))
    print('===> Training model')
    writer = SummaryWriter(log_dir=os.path.join(opt.save_path, 
        datetime.now().strftime('%b%d_%H-%M-%S') + '_' + opt.identifier))

    # write checkpoints in logdir
    logdir = writer.file_writer.get_logdir()
    opt.save_file_path = os.path.join(logdir, 'checkpoints')
    os.makedirs(opt.save_file_path)

    # Do training
    do_training(train_dataset, validation_dataset, model, optimizer, 
        criterion, encoder_dim, device, epoch, opt, config, checkpoint, writer)
    """

if __name__ == "__main__":
    main()

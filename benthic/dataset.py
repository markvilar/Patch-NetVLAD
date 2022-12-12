import os
import sys
import itertools

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.utils.data as data
import pandas as pd

from PIL import Image
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset

from benthic.cache import SubCache
from benthic.filesystem import search_for_files

def is_positive(anchor, sample) -> bool:
    """ Determine whether a sample is a positive or not. """
    return False


def is_negative(anchor, sample) -> bool:
    """ Determine whether a sample is a negative or not. """
    return False


def create_batch(batch: List[Tuple]):
    """Creates mini-batch tensors from the list of tuples 
    (query, positive, negatives).

    Args:
        batch: list of tuple (query, positive, negatives).
            - query: torch tensor of shape (3, h, w).
            - positive: torch tensor of shape (3, h, w).
            - negative: torch tensor of shape (n, 3, h, w).
    Returns:
        query: torch tensor of shape (batch_size, 3, h, w).
        positive: torch tensor of shape (batch_size, 3, h, w).
        negatives: torch tensor of shape (batch_size, n, 3, h, w).
    """
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None, None, None, None, None

    query, positive, negatives, indices = zip(*batch)

    query = data.dataloader.default_collate(query)
    positive = data.dataloader.default_collate(positive)
    negCounts = data.dataloader.default_collate([x.shape[0] 
        for x in negatives])
    negatives = torch.cat(negatives, 0)
    indices = list(itertools.chain(*indices))

    return query, positive, negatives, negCounts, indices



class BenthicDataset(Dataset):
    def __init__(self, root):
        self.triplets = []

        self.query_image_keys = set()
        self.query_image_paths = set()
        self.database_image_keys = set()
        self.database_image_paths = set()

        # Transformation applied to images before input (resize, normalization)
        self.input_transform = None

        self.cache = None
        #TODO: Implement

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        """ Returns a triplet consisting of one query, one positive, and N
        negatives. """
        triplet, target = self.triplets[idx]

        # get query, positive and negative idx
        qidx = triplet[0]
        pidx = triplet[1]
        nidx = triplet[2:]

        # Load query image TODO: Index -> Key -> Path?
        query_image = Image.open(self.query_image_keys[qidx])
        query = self.input_transform(query_image)
        
        # Load positive image TODO: Index -> Key -> Path?
        positive_image = Image.open(self.database_image_paths[pidx])
        positive = self.input_transform(positive_image)

        # Load negative images TODO: Index -> Key -> Path?
        negative_images = [Image.open(self.database_image_paths[idx]) \
            for idx in nidx]
        negatives = [self.input_transform(image) for image in negative_images]
        negatives = torch.stack(negatives, 0)

        return query, positive, negatives, [qidx, pidx] + nidx

    def create_training_set(self):
        pass

    def create_validation_set(self):
        pass

    def create_test_set(self):
        pass

    def prepare_dataloaders(self):
        pass


class BenthicDatasetFactory():
    """ """
    def __init__(self, 
            image_directory: Path, 
            query_path: Path, 
            index_path: Path, 
            threshold_pos: float, 
            threshold_neg: float,
            transform,
        ):
        self.image_directory = image_directory
        self.query_path = query_path
        self.index_path = index_path

        assert os.path.isdir(image_directory)
        assert os.path.isfile(query_path)
        assert os.path.isfile(index_path)

        self.query_data = pd.read_csv(query_path)
        self.index_data = pd.read_csv(index_path)

        self.query_image_keys = set(self.query_data["label"])
        self.index_image_keys = set(self.index_data["label"])

        # TODO: Search for images in image directory based on keys
        self.query_image_paths = search_for_files(self.image_directory, 
            self.query_image_keys)
        self.index_image_paths = search_for_files(self.image_directory, 
            self.index_image_keys)
    
        # TODO: Do more advanced decision making than simple thresholding
        self.threshold_pos = threshold_pos
        self.threshold_neg = threshold_neg

        # TODO: Add
        #self.threads = threads
        #self.batch_size = batch_size

        # other
        self.input_transform = transform
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

    def create_training(self, validation_fraction: float) \
        -> Tuple[BenthicDataset, BenthicDataset]:
        """ Creates a training and validation dataset from benthic imagery. """
        # Sanity check
        assert validation_fraction > 0.0 and validation_fraction < 1.0

        # TODO: Find anchor-positive pairs
        for item in self.query_data:
            print(item)

        # TODO: Distribution queries with positives among training and 
        # validation

        """
        # unique_qSeqIdx = np.unique(qSeqIdxs)
        # unique_dbSeqIdx = np.unique(dbSeqIdxs)

        # TODO: process positions
        # utmQ = qData[['easting', 'northing']].values.reshape(-1, 2)
        # utmDb = dbData[['easting', 'northing']].values.reshape(-1, 2)

        # TODO: find positive images for training
        neigh = NearestNeighbors(algorithm='brute')
        neigh.fit(utmDb)
        pos_distances, pos_indices = neigh.radius_neighbors(utmQ, 
            self.threshold_pos)
        self.all_pos_indices.extend(pos_indices)

        # TODO: Create training data
        nD, nI = neigh.radius_neighbors(utmQ, self.threshold_neg)
        """

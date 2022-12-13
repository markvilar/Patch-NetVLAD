import math
import os
import random
import sys

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
import torch.utils.data as data
import pandas as pd

from PIL import Image
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset

from patchnetvlad.tools.datasets import input_transform

from benthic.cache import SubCache
from benthic.filesystem import search_for_files

@dataclass
class QueryItem():
    key: str
    positives: Set[int]
    negatives: Set[int]
    path: Path


@dataclass
class Query():
    items: List[QueryItem]

    def __len__(self):
        return len(self.items)

    def __get_item__(self, index: int):
        return self.items[index]


@dataclass
class DatabaseItem():
    key: str
    path: Path


@dataclass
class Database():
    items: List[DatabaseItem]

    def __len__(self):
        return len(self.items)

    def __get_item__(self, index: int):
        return self.items[index]


def calculate_positives(anchor: pd.Series, samples: pd.DataFrame, 
    threshold: float) -> Set[int]:
    """ Returns the sample indices which are within the threshold. """
    anchor_pos = anchor[["posx", "posy", "posz"]]
    sample_pos = samples[["posx", "posy", "posz"]]
    distances = np.sqrt(np.square(sample_pos - anchor_pos).sum(axis=1))
    mask = distances < threshold
    positives = set(mask.index[mask == True].tolist())
    return positives


def calculate_negatives(anchor: pd.Series, samples: pd.DataFrame, 
    threshold: float) -> Set[int]:
    anchor_pos = anchor[["posx", "posy", "posz"]]
    sample_pos = samples[["posx", "posy", "posz"]]
    distances = np.sqrt(np.square(sample_pos - anchor_pos).sum(axis=1))
    mask = distances > threshold
    negatives = set(mask.index[mask == True].tolist())
    return negatives


def find_potential_triplets(query, database, threshold_pos, threshold_neg) \
    -> Dict[int, Set[int]]:
    """ Find anchor-positive pairs """
    potential_triplets = dict()
    for index, row in query.iterrows():

        # Check if any distances within threshold
        positives = calculate_positives(row, database, threshold_pos)

        # Skip some work if no positives are present
        if len(positives) == 0:
            continue

        negatives = calculate_negatives(row, database, threshold_neg)

        # Pick a smaller subset of negatives to use
        negatives = set(random.sample(negatives, 100))

        # For each valid anchor, get valid positives
        if len(positives) > 0 and len(negatives) > 0:
            potential_triplets[index] = (positives, negatives)
    return potential_triplets


def split_triplets(triplets: Dict[int, Tuple], fraction: float):
    """ Given a dictionary of triplets with index, positives, and negatives,
    distribute the triplets over two dictionaries. """
    num_triplets = len(triplets)
    num_training = math.ceil((1 - fraction) * num_triplets)
   
    # Create sets of training and validation anchors
    queries = set(triplets.keys())
    training_queries = set(random.sample(queries, num_training))
    validation_queries = queries - training_queries

    training_triplets = {}
    for query in training_queries:
        training_triplets[query] = triplets[query]

    validation_triplets = {}
    for query in validation_queries:
        validation_triplets[query] = triplets[query]

    return training_triplets, validation_triplets


@dataclass
class BenthicDataset(Dataset):
    query: Query
    database: Database
    input_transform: torchvision.transforms.Compose
    caches: List=field(default_factory=list)
    triplets: List=field(default_factory=list)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, index: int):
        """ Returns a triplet consisting of one anchor, one positive, and N
        negatives. """
        # TODO: Figure out what target is!
        triplet, target = self.triplets[index]

        # get query, positive and negative index
        qindex = triplet[0]
        pindex = triplet[1]
        nindex = triplet[2:]

        # Load query image TODO: Index -> Key -> Path?
        query_image = Image.open(self.query_image_keys[qindex])
        query = self.input_transform(query_image)
        
        # Load positive image TODO: Index -> Key -> Path?
        positive_image = Image.open(self.database_image_paths[pindex])
        positive = self.input_transform(positive_image)

        # Load negative images TODO: Index -> Key -> Path?
        negative_images = [Image.open(self.database_image_paths[index]) \
            for index in nindex]
        negatives = [self.input_transform(image) for image in negative_images]
        negatives = torch.stack(negatives, 0)

        return query, positive, negatives, [qindex, pindex] + nindex

    def get_query_count(self) -> int:
        return len(self.query)

    def get_database_count(self) -> int:
        return len(self.database)
   
    def new_epoch(self):
        # TODO: Implement
        # TODO: Calculate number of subsets
        # TODO: Do random sample of queries
        # TODO: Reset iterator index
        pass

    def update_cache(self):
        # TODO: Implement
        pass


class BenthicDatasetFactory():
    """ """
    def __init__(self, 
            image_directory: Path, 
            query_path: Path, 
            database_path: Path, 
            threshold_pos: float, 
            threshold_neg: float,
        ):
        self.image_directory = image_directory
        self.query_path = query_path
        self.database_path = database_path

        assert os.path.isdir(image_directory)
        assert os.path.isfile(query_path)
        assert os.path.isfile(database_path)

        self.query = pd.read_csv(query_path)
        self.database = pd.read_csv(database_path)

        # Filter samples with altitudes out of bounds
        altitude_low, altitude_high = 1.5, 3.5
        query_mask = (self.query["altitude"] < altitude_high) \
            & (self.query["altitude"] > altitude_low)
        index_mask = (self.database["altitude"] < altitude_high) \
            & (self.database["altitude"] > altitude_low)

        self.query = self.query[query_mask]
        self.database = self.database[index_mask]

        self.query = self.query.reset_index()
        self.database = self.database.reset_index()

        self.query_image_keys = set(self.query["label"])
        self.database_image_keys = set(self.database["label"])

        # TODO: Might couple with index?
        # Search for images in image directory based on keys
        self.query_image_paths = search_for_files(self.image_directory, 
            self.query_image_keys)
        self.database_image_paths = search_for_files(self.image_directory, 
            self.database_image_keys)
    
        # Add thresholds
        self.threshold_pos = threshold_pos
        self.threshold_neg = threshold_neg


    def create_query_items(self, triplets: Dict):
        """ 
        Given a set of query indices with positives and negatives, create
        a list of query items with all relevant information for easy lookup.
        """
        query_items = list()
        for index, (positives, negatives) in triplets.items():
            data = self.query.iloc[index]
            label = data["label"]
            path = self.query_image_paths[label]
            item = QueryItem(key=label, positives=positives, 
                negatives=negatives, path=path)
            query_items.append(item)
        return query_items


    def create_database_items(self):
        """ """
        database_items = list()
        for index, row in self.database.iterrows():
            label = row["label"]
            path = self.database_image_paths[label]
            item = DatabaseItem(key=label, path=path)
            database_items.append(item)
        return database_items


    def create_training_data(self, validation_fraction: float) \
        -> Tuple[BenthicDataset, BenthicDataset]:
        """ Creates a training and validation set from benthic imagery. """
        # Sanity check
        assert validation_fraction > 0.0 and validation_fraction < 1.0

        # Get potential triplets
        triplets = find_potential_triplets(self.query, self.database,
            self.threshold_pos, self.threshold_neg)

        assert len(triplets) > 0

        training_triplets, validation_triplets = split_triplets(triplets,
            validation_fraction)
        
        assert len(training_triplets) > 0
        assert len(validation_triplets) > 0
    
        # Get trainin
        training_query_items = self.create_query_items(training_triplets)
        validation_query_items = self.create_query_items(validation_triplets)

        training_database_items = self.create_database_items()
        validation_database_items = self.create_database_items()

        # Create training and validation queries
        training_query = Query(training_query_items)
        validation_query = Query(validation_query_items)

        # Create training and validation database
        training_database = Database(training_database_items)
        validation_database = Database(validation_database_items)

        training_set = BenthicDataset(
            query = training_query,
            database = training_database,
            input_transform = input_transform(),
        )

        validation_set = BenthicDataset(
            query = validation_query,
            database = validation_database,
            input_transform = input_transform(),
        )

        return training_set, validation_set

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch.utils.data as data
import pandas as pd
from os.path import join
from sklearn.neighbors import NearestNeighbors
import math
import torch
import random
import sys
import itertools
from tqdm import tqdm

from .cache import SubCache


class CustomDataset(Dataset):
    def __init__(self, root):
        self.triplets = []
        self.qImages = None
        self.dbImages = None

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

        # load images into triplet list
        query = self.transform(Image.open(self.qImages[qidx]))
        positive = self.transform(Image.open(self.dbImages[pidx]))
        negatives = [
            self.transform(Image.open(self.dbImages[idx])) for idx in nidx
        ]
        negatives = torch.stack(negatives, 0)
        return query, positive, negatives, [qidx, pidx] + nidx

    def create_training_set(self):
        pass

    def create_test_set(self):
        pass

    def prepare_dataloaders(self):
        pass


def create_batch(batch):
    """Creates mini-batch tensors from the list of tuples (query, positive, 
    negatives).

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




class DatasetFactory():
    def __init__(self):
        self.qIdx = []
        self.qImages = []
        self.pIdx = []
        self.nonNegIdx = []
        self.dbImages = []
        
        # hyper-parameters
        self.nNeg = nNeg
        self.margin = margin
        self.posDistThr = posDistThr
        self.negDistThr = negDistThr
        self.cached_queries = cached_queries
        self.cached_negatives = cached_negatives

        # flags
        self.cache = None
        mode = mode

        # other
        self.transform = transform

        # define sequence length based on task
        seq_length_q, seq_length_db = 1, 1

        # load data
        # get len of images from cities so far for indexing
        _lenQ = len(self.qImages)
        _lenDb = len(self.dbImages)


    def create_instance(self, mode):
        if mode in ['train', 'val']:
            # TODO: load query and database data
            # qData = pd.read_csv( path , index_col=0)
            # dbData = pd.read_csv( path , index_col=0)

            # TODO: Create validation dataset
            if mode in ['val']:
                #qIdx = pd.read_csv( 'query/subtask_index.csv', index_col=0)
                #dbIdx = pd.read_csv('database/subtask_index.csv', index_col=0)

            unique_qSeqIdx = np.unique(qSeqIdxs)
            unique_dbSeqIdx = np.unique(dbSeqIdxs)

            self.qImages.extend(qSeqKeys)
            self.dbImages.extend(dbSeqKeys)

            self.qEndPosList.append(len(qSeqKeys))
            self.dbEndPosList.append(len(dbSeqKeys))

            qData = qData.loc[unique_qSeqIdx]
            dbData = dbData.loc[unique_dbSeqIdx]

            # TODO: process positions
            # utmQ = qData[['easting', 'northing']].values.reshape(-1, 2)
            # utmDb = dbData[['easting', 'northing']].values.reshape(-1, 2)

            # TODO: find positive images for training
            """
            neigh = NearestNeighbors(algorithm='brute')
            neigh.fit(utmDb)
            pos_distances, pos_indices = neigh.radius_neighbors(utmQ, 
                self.posDistThr)
            self.all_pos_indices.extend(pos_indices)
            """

            # TODO: Create training data
            if mode == 'train':
                nD, nI = neigh.radius_neighbors(utmQ, self.negDistThr)

        # when GPS / UTM is not available
        elif mode in ['test']:

            # TODO: Load query and database for testing
            #qIdx = pd.read_csv('query/subtask_index.csv'), index_col=0)
            #dbIdx = pd.read_csv('database/subtask_index.csv'), index_col=0)

            # 
            self.qImages.extend(qSeqKeys)
            self.dbImages.extend(dbSeqKeys)

            # add query index
            self.qIdx.extend(list(range(_lenQ, len(qSeqKeys) + _lenQ)))

            # if a combination of cities, task and subtask is chosen, where 
            # there are no query/database images, then exit
        if len(self.qImages) == 0 or len(self.dbImages) == 0:
            print("Exiting...")
            print("A combination of cities, task and subtask have been chosen, 
                where there are no query/database images.")
            print("Try choosing a different subtask or more cities")
            sys.exit() 
        
        # cast to np.arrays for indexing during training
        self.qIdx = np.asarray(self.qIdx)
        self.qImages = np.asarray(self.qImages)
        self.pIdx = np.asarray(self.pIdx)
        self.nonNegIdx = np.asarray(self.nonNegIdx)
        self.dbImages = np.asarray(self.dbImages)
        self.sideways = np.asarray(self.sideways)
        self.night = np.asarray(self.night)

        # decide device type ( important for triplet mining )
        self.device = torch.device("cuda:0" if torch.cuda.is_available() \
            else "cpu")
        self.threads = threads
        self.batch_size = batch_size

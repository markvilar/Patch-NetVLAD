import math
import random

import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


class ImagesFromList(Dataset):
    def __init__(self, images, transform):
        self.images = np.asarray(images)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            img = [Image.open(im) for im in self.images[idx].split(",")]
        except:
            img = [Image.open(self.images[0])]
        img = [self.transform(im) for im in img]

        if len(img) == 1:
            img = img[0]

        return img, idx


class SubCache():
    def __init__(self):
        self.num_cache_subset = 0
        self.triplets = []
        self.subcache_indices = 0
        self.current_subset = 0

        # Hyper
        self.batch_size = 30

        self.threads = 6

    def new_epoch(self):
        # find how many subset we need to do 1 epoch
        self.num_cache_subset = math.ceil(len(self.qIdx) / self.cached_queries)
        # get all indices
        arr = np.arange(len(self.qIdx))
        # apply positive sampling of indices
        arr = random.choices(arr, self.weights, k=len(arr))
        # calculate the subcache indices
        self.subcache_indices = np.array_split(arr, self.num_cache_subset)
        # reset subset counter
        self.current_subset = 0   

    def update(self, net=None, outputdim=None):
        # reset triplets
        self.triplets = []
        
        if self.current_subset >= len(self.subcache_indices):
            tqdm.write('Reset epoch - FIX THIS LATER!')
            self.current_subset = 0
        
        # take n query images
        qidxs = np.asarray(self.subcache_indices[self.current_subset])

        # take their positive in the database
        pidxs = np.unique([i for idx in self.pIdx[qidxs] for i in idx])

        # take m = 5*cached_queries is number of negative images
        nidxs = np.random.choice(len(self.dbImages), self.cached_negatives,
            replace=False)

        # and make sure that there is no positives among them
        nidxs = nidxs[np.in1d(nidxs, 
            np.unique([i for idx in self.nonNegIdx[qidxs] for i in idx]), 
            invert=True)
        ]

        # make dataloaders for query, positive and negative images
        opt = {
            'batch_size': self.bs, 
            'shuffle': False, 
            'num_workers': self.threads, 
            'pin_memory': True
        }

        qloader = torch.utils.data.DataLoader(
            ImagesFromList(self.qImages[qidxs], transform=self.transform), 
            **opt
        )
        ploader = torch.utils.data.DataLoader(
            ImagesFromList(self.dbImages[pidxs], transform=self.transform), 
            **opt
        )
        nloader = torch.utils.data.DataLoader(
            ImagesFromList(self.dbImages[nidxs], transform=self.transform), 
            **opt
        )

        # calculate their descriptors
        net.eval()
        with torch.no_grad():

            # initialize descriptors
            qvecs = torch.zeros(len(qidxs), outputdim).to(self.device)
            pvecs = torch.zeros(len(pidxs), outputdim).to(self.device)
            nvecs = torch.zeros(len(nidxs), outputdim).to(self.device)

            batch_size = opt['batch_size']

            # Calculate descriptors
            self.calculate_descriptors(qloader, qvecs, batch_size)
            self.calculate_descriptors(ploader, pvecs, batch_size)
            self.calculate_descriptors(nloader, nvecs, batch_size)

        tqdm.write('>> Searching for hard negatives...')
        # compute dot product scores and ranks on GPU
        pScores = torch.mm(qvecs, pvecs.t())
        pScores, pRanks = torch.sort(pScores, dim=1, descending=True)

        # calculate distance between query and negatives
        nScores = torch.mm(qvecs, nvecs.t())
        nScores, nRanks = torch.sort(nScores, dim=1, descending=True)

        # convert to cpu and numpy
        pScores, pRanks = pScores.cpu().numpy(), pRanks.cpu().numpy()
        nScores, nRanks = nScores.cpu().numpy(), nRanks.cpu().numpy()

        # selection of hard triplets
        for q in range(len(qidxs)):
            qidx = qidxs[q]

            # find positive idx for this query (cache idx domain)
            cached_pidx = np.where(np.in1d(pidxs, self.pIdx[qidx]))

            # find idx of positive idx in rank matrix 
            # (descending cache idx domain)
            pidx = np.where(np.in1d(pRanks[q, :], cached_pidx))

            # take the closest positve
            dPos = pScores[q, pidx][0][0]

            # get distances to all negatives
            dNeg = nScores[q, :]

            # how much are they violating
            loss = dPos - dNeg + self.margin ** 0.5
            violatingNeg = 0 < loss

            # if less than nNeg are violating then skip this query
            if np.sum(violatingNeg) <= self.nNeg:
                continue

            # select hardest negatives
            hardest_negIdx = np.argsort(loss)[:self.nNeg]

            # select the hardest negatives
            cached_hardestNeg = nRanks[q, hardest_negIdx]

            # select the closest positive (back to cache idx domain)
            cached_pidx = pRanks[q, pidx][0][0]

            # transform back to original index (back to original idx domain)
            qidx = self.qIdx[qidx]
            pidx = pidxs[cached_pidx]
            hardestNeg = nidxs[cached_hardestNeg]

            # package the triplet and target
            triplet = [qidx, pidx, *hardestNeg]
            target = [-1, 1] + [0] * len(hardestNeg)

            self.triplets.append((triplet, target))

        # increment subset counter
        self.current_subset += 1

    def calculate_descriptors(self, dataloader, vecs, batch_size):
        for i, batch in tqdm(enumerate(dataloader), 
            total=len(nidxs) // batch_size,
            position=2, leave=False,
            desc='computing descriptors'):
            X, y = batch
            image_encoding = net.encoder(X.to(self.device))
            vlad_encoding = net.pool(image_encoding)

            low = i * bs
            upper = (i + 1) * bs
            vecs[lower:upper, :] = vlad_encoding

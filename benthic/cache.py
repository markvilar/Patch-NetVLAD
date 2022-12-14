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
            qidxs = np.random.choice(len(self.qIdx), self.cached_queries, 
                replace=False)

            for q in qidxs:
                qidx = self.qIdx[q]
                pidxs = self.pIdx[q]
                pidx = np.random.choice(pidxs, size=1)[0]

                while True:
                    nidxs = np.random.choice(len(self.dbImages), size=self.nNeg)
                    # ensure that non of the choice negative images are within 
                    # the negative range (default 25 m)
                    if sum(np.in1d(nidxs, self.nonNegIdx[q])) == 0:
                        break

                # package the triplet and target
                triplet = [qidx, pidx, *nidxs]
                target = [-1, 1] + [0] * len(nidxs)

                self.triplets.append((triplet, target))

            # increment subset counter
            self.current_subset += 1

            return

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

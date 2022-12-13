import itertools

from typing import List, Tuple

import torch

def tuples_to_tensors(batch: List[Tuple]):
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

    query = torch.utils.data.dataloader.default_collate(query)
    positive = torch.utils.data.dataloader.default_collate(positive)
    neg_counts = torch.utils.data.dataloader.default_collate([x.shape[0] 
        for x in negatives])
    negatives = torch.cat(negatives, 0)
    indices = list(itertools.chain(*indices))

    return query, positive, negatives, neg_counts, indices

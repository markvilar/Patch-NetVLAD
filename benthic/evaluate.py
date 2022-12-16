import faiss
import numpy as np
import torch
from tqdm.auto import tqdm

from torch.utils.data import DataLoader

from patchnetvlad.training_tools.msls import ImagesFromList
from patchnetvlad.tools.datasets import input_transform

def extract_features(dataloader, model, pool_size, pbar_position, device):
    image_count = len(dataloader.dataset)
    features = np.empty((image_count, pool_size), dtype=np.float32)

    for iteration, (input_data, indices) in \
        enumerate(tqdm(dataloader, position=pbar_position, leave=False, 
            desc='Test Iter'.rjust(15)), 1
        ):
        
        input_data = input_data.to(device)

        image_encoding = model.encoder(input_data)
        vlad_encoding = model.pool(image_encoding)

        features[indices.detach().numpy(), :] \
            = vlad_encoding.detach().cpu().numpy()

        del input_data, image_encoding, vlad_encoding
    return features


def evaluate(evaluation_set, model, encoder_dim, device, options, config, 
    writer, epoch_num=0, write_tboard=False, pbar_position=0):
    """ Evaluate a model. """
    if device.type == 'cuda':
        cuda = True
    else:
        cuda = False

    model = model.to(device)

    # Get images from validation set
    query_images = ImagesFromList(evaluation_set.get_query_images(), 
        transform=input_transform())
    database_images = ImagesFromList(evaluation_set.get_database_images(), 
        transform=input_transform())

    # Set up dataloader for query
    query_dataloader = DataLoader(
            dataset=query_images,
            num_workers=options.threads, 
            batch_size=int(config['feature_extract']['cachebatchsize']),
            shuffle=False, 
            pin_memory=cuda,
        )

    # Set up dataloader for database
    database_dataloader = DataLoader(
            dataset=database_images,
            num_workers=options.threads, 
            batch_size=int(config['feature_extract']['cachebatchsize']),
            shuffle=False, 
            pin_memory=cuda,
        )

    model.eval()
    with torch.no_grad():
        tqdm.write('====> Extracting Features')
        pool_size = encoder_dim
        if config['global_params']['pooling'].lower() == 'netvlad':
            pool_size *= int(config['global_params']['num_clusters'])
        query_features = np.empty((len(query_images), pool_size), 
            dtype=np.float32)
        database_features = np.empty((len(database_images), pool_size), 
            dtype=np.float32)
    
        # Extract query features
        query_features = extract_features(query_dataloader, model, 
            pool_size, pbar_position, device)

        # Database query features
        database_features = extract_features(database_dataloader, model, 
            pool_size, pbar_position, device)

    del query_dataloader, database_dataloader

    # Create nearest neighbourhood from database features
    tqdm.write('====> Building faiss index')
    faiss_index = faiss.IndexFlatL2(pool_size)
    faiss_index.add(database_features)

    tqdm.write('====> Calculating recall @ N')
    thresholds = [1, 5, 10, 20, 50, 100]

    # Create index from database features
    faiss_index = faiss.IndexFlatL2(pool_size)
    faiss_index.add(database_features)

    _, predictions = faiss_index.search(query_features, max(thresholds))

    query = evaluation_set.get_query()
    positives = [item.positives for item in query.items]

    hits = np.zeros(len(thresholds))

    # For each query, find correct predictions
    for query_index, prediction in enumerate(predictions):
        for index, threshold in enumerate(thresholds):
            # if in top N then also in top NN, where NN > N
            proposals = prediction[:threshold]
            targets = list(positives[query_index])
            if np.any(np.in1d(proposals, targets)):
                hits[index:] += 1
                break
    recalls = hits / len(evaluation_set.get_query())

    all_recalls = {}  # make dict for output
    for index, threshold in enumerate(thresholds):
        all_recalls[threshold] = recalls[index]
        tqdm.write("====> recall@{}: {:.4f}".format(threshold, recalls[index]))
        if write_tboard:
            writer.add_scalar('eval/recall@' + str(threshold), recalls[index])

    #print("=> Best Recall@5: {:.4f}".format(best_score), flush=True)
    writer.close()

    # Garbage clean GPU memory
    torch.cuda.empty_cache()

    print("Done")

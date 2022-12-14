#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

#from torch.utils.data import DataLoader
from tqdm.auto import trange, tqdm

#from patchnetvlad.training_tools.val import val
from patchnetvlad.training_tools.tools import save_checkpoint, humanbytes
from patchnetvlad.models.models_generic import get_backend

from benthic import batch
from benthic.validation import validate

def train_epoch(training_set, model, trainer, encoder_dim, epoch_num, options, 
    config, writer):
    """ Train the model for an epoch. """

    optimizer = trainer.get_optimizer()
    criterion = trainer.get_criterion()
    device = trainer.get_device()
    
    if device.type == "cuda":
        pin_memory = True
    else:
        pin_memory = False
    
    training_set.new_epoch()

    epoch_loss = 0.0
    start_iter = 1 # keep track of batch iter across subsets for logging

    query_count = training_set.get_query_count()
    batch_size = trainer.get_batch_size()
    batch_count = (query_count + batch_size - 1) // batch_size

    # Iterate over caches - for each cache
    for sub_iter in trange(training_set.get_cache_count(), 
        desc="Cache refresh".rjust(15), position=1):
        
        pool_size = encoder_dim
        if config["global_params"]["pooling"].lower() == "netvlad":
            pool_size *= int(config["global_params"]["num_clusters"])

        tqdm.write("====> Building Cache")
        
        # TODO: Implement cache?
        training_set.update_cache() # update_subcache(model, pool_size)

        # TODO: Implement function to create dataloader
        training_dataloader = torch.utils.data.DataLoader(
                dataset=training_set, 
                num_workers=options.threads,
                batch_size=batch_size, 
                shuffle=True,
                collate_fn=batch.tuples_to_tensors, # TODO: Verify
                pin_memory=pin_memory
            )

        tqdm.write("Allocated: " + humanbytes(torch.cuda.memory_allocated()))
        tqdm.write("Cached:    " + humanbytes(torch.cuda.memory_reserved()))

        model.train()
        
        # Iterate over batches
        for iteration, (query, positives, negatives, negCounts, indices) in \
            enumerate(tqdm(training_dataloader, position=2, leave=False, 
                desc="Train Iter".rjust(15)), start_iter):
            # some reshaping to put query, pos, negs in a single (N, 3, H, W) 
            # tensor where N = batchSize * (nQuery + nPos + nNeg)
            if query is None:
                continue  # in case we get an empty batch

            B, C, H, W = query.shape
            nNeg = torch.sum(negCounts)
            data_input = torch.cat([query, positives, negatives])

            data_input = data_input.to(device)
            image_encoding = model.encoder(data_input)
            vlad_encoding = model.pool(image_encoding)

            vladQ, vladP, vladN = torch.split(vlad_encoding, [B, B, nNeg])

            optimizer.zero_grad()

            # calculate loss for each Query, Positive, Negative triplet
            # due to potential difference in number of negatives have to
            # do it per query, per negative
            loss = 0
            for i, negCount in enumerate(negCounts):
                for n in range(negCount):
                    negIx = (torch.sum(negCounts[:i]) + n).item()
                    loss += criterion(vladQ[i: i + 1], vladP[i: i + 1], 
                        vladN[negIx:negIx + 1])

            # Update model - normalise loss by number of negatives
            loss /= nNeg.float().to(device)  
            loss.backward()
            optimizer.step()
            del data_input, image_encoding, vlad_encoding, vladQ, vladP, vladN
            del query, positives, negatives

            batch_loss = loss.item()
            epoch_loss += batch_loss

            if iteration % 50 == 0 or batch_count <= 10:
                tqdm.write("==> Epoch[{}]({}/{}): Loss: {:.4f}".format(
                    epoch_num, iteration, batch_count, batch_loss))
                writer.add_scalar("Train/Loss", batch_loss,
                    ((epoch_num - 1) * batch_count) + iteration)
                writer.add_scalar("Train/nNeg", nNeg,
                    ((epoch_num - 1) * batch_count) + iteration)
                tqdm.write("Allocated: " 
                    + humanbytes(torch.cuda.memory_allocated()))
                tqdm.write("Cached:    " 
                    + humanbytes(torch.cuda.memory_reserved()))

        start_iter += len(training_dataloader)
        del training_dataloader, loss
        optimizer.zero_grad()
        torch.cuda.empty_cache()
       
        # Update optimizer parameters with scheduler
        if trainer.has_scheduler():
            trainer.get_scheduler().step()


    avg_loss = epoch_loss / batch_count

    tqdm.write("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch_num, 
        avg_loss))
    writer.add_scalar("Train/AvgLoss", avg_loss, epoch_num)


def train_model(training_set, validation_set, model, trainer, options, config, 
    writer):
    """ Perform training and validation of model."""
    not_improved, best_score = 0, 0

    optimizer = trainer.get_optimizer()
    device = trainer.get_device()
    model = model.to(device)
    
    """
    # TODO: Look into checkpoint
    if options.resume_path:
        not_improved = checkpoint["not_improved"]
        best_score = checkpoint["best_score"]
    """
    encoder_dim, _ = get_backend()

    print("Starting training epochs...")
    for epoch in trange(1, options.epochs + 1, desc="Epoch number".rjust(15), 
        position=0):

        # NOTE: Seems to be working.
        train_epoch(training_set, model, trainer, encoder_dim, epoch, options, 
            config, writer)

        if (epoch % int(config["train"]["evalevery"])) == 0:
            
            # NOTE: Seems to be working.
            recalls = validate(validation_set, model, encoder_dim, device, 
                options, config, writer, epoch, write_tboard=True, 
                pbar_position=1)

            is_best = recalls[5] > best_score
            if is_best:
                not_improved = 0
                best_score = recalls[5]
            else:
                not_improved += 1

            save_checkpoint({
                    "epoch"        : epoch,
                    "state_dict"   : model.state_dict(),
                    "recalls"      : recalls,
                    "best_score"   : best_score,
                    "not_improved" : not_improved,
                    "optimizer"    : optimizer.state_dict(),
                    "parallel"     : False,
                }, 
                options, 
                is_best,
            )

            patience = int(config["train"]["patience"])
            eval_step = int(config["train"]["evalevery"])
            horizon = patience / eval_step

            if patience > 0 and not_improved > horizon:
                print("Performance did not improve for "
                    + "{0} epochs. Stopping.".format(patience))
                break

    print("=> Best Recall@5: {:.4f}".format(best_score), flush=True)
    writer.close()

    # Garbage clean GPU memory
    torch.cuda.empty_cache()

    print("Done")

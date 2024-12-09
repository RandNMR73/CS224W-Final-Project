import torch 
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss, L1Loss
from torch.amp import GradScaler, autocast
import contextlib
import inspect 

import copy 
import random
import numpy as np
import config
import math
from baseline_models import BaselineModel
from dataset import create_task_train_dict

from torch_geometric.loader import NeighborLoader

from model import RelTransformer
# from fvcore.nn import FlopCountAnalysis  # need to add to docker container 

# need to add mixed-precision training

# global params 
embed_dim = config.EMBED_DIM
num_heads = config.NUM_HEADS
dropout = config.DROPOUT
num_layers = config.NUM_LAYERS

def train(task, entity_table, model, loader: NeighborLoader, loss_fn, optimizer) -> float:
    model.train()
    scaler = GradScaler()
    loss_accum = count_accum = 0
    for batch in tqdm(loader):
        batch = batch.to(config.DEVICE)
        optimizer.zero_grad()
        with autocast('cuda') if config.DEVICE=='cuda' else contextlib.nullcontext():
            pred = model(
                batch,
                task.entity_table,
            )
            pred = pred.view(-1) if pred.size(1) == 1 else pred
            loss = loss_fn(pred.float(), batch[entity_table].y.float())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_accum += loss.detach().item() * pred.size(0)
        count_accum += pred.size(0)

    return loss_accum / count_accum

def val(task, entity_table, model, loader: NeighborLoader, loss_fn) -> float:
    model.eval()

    pred_list = []
    loss_accum = count_accum = 0
    for batch in loader:
        batch = batch.to(config.DEVICE)
        pred = model(
            batch,
            task.entity_table,
        )
        pred = pred.view(-1) if pred.size(1) == 1 else pred
        pred_list.append(pred.detach().cpu())
        loss = loss_fn(pred.float(), batch[entity_table].y.float())

        loss_accum += loss.detach().item() * pred.size(0)
        count_accum += pred.size(0)
    
    return loss_accum / count_accum, torch.cat(pred_list, dim = 0).numpy()

@torch.no_grad()
def test(model, task, loader: NeighborLoader, loss = True) -> np.ndarray:
    model.eval()

    pred_list = []
    for batch in tqdm(loader):
        batch = batch.to(config.DEVICE)
        pred = model(
            batch,
            task.entity_table,
        )
        pred = pred.view(-1) if pred.size(1) == 1 else pred
        pred_list.append(pred.detach().cpu())
    return torch.cat(pred_list, dim=0).numpy()

def train_val_on_all(task_to_train_info, model, optimizer, loss_fn):
    task_epochs = {task: 0 for task in task_to_train_info.keys()}
    state_dict = None

    total_epochs_to_train = len(task_to_train_info) * config.EPOCHS
    total_epochs_trained = 0

    while total_epochs_trained < total_epochs_to_train:
        valid_keys = [key for key in task_epochs.keys() if task_epochs[key] < config.EPOCHS]
        
        if not valid_keys:
            print("Training finished.")
            break

        random_task = random.choice(valid_keys)
        train_items = task_to_train_info[random_task]

        higher_is_better = config.HIGHER_IS_BETTER[train_items.task_metrics[0]]
        best_val_metric = -math.inf if higher_is_better else math.inf
        tune_metric = train_items.task_metrics[0].__name__
        for epoch in range(config.EPOCHS_TO_SWITCH + 1):
            if task_epochs[random_task] >= config.EPOCHS:
                break

            train_loss = train(train_items.task, train_items.entity_table, model, train_items.loader_dict["train"], loss_fn, optimizer)
            val_loss, val_pred = val(train_items.task, train_items.entity_table, model, train_items.loader_dict["val"], loss_fn)
            val_metrics = train_items.task.evaluate(val_pred, train_items.val_table)
            print(f"Task: {random_task}, Epoch: {epoch:02d}, Train loss: {train_loss}, Val metrics: {val_metrics}")
            
            task_epochs[random_task] += 1
            total_epochs_trained += 1

            if (higher_is_better and val_metrics[tune_metric] > best_val_metric) or (
                not higher_is_better and val_metrics[tune_metric] < best_val_metric
            ):
                best_val_metric = val_metrics[tune_metric]
                state_dict = copy.deepcopy(model.state_dict())
            if config.CHECKPOINT and (epoch + 1) % config.EPOCHS_TO_SAVE == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'training_loss': train_loss,
                    'validation_loss': val_loss,
                }, 'checkpoint.pth')

def main():
    hetero_graph, col_stats_dict, task_to_train_info = create_task_train_dict("rel-f1")

    # model = BaselineModel(
    #     data=hetero_graph,
    #     col_stats_dict=col_stats_dict,
    #     gnn_layer = "HeteroGAT",
    #     num_layers=2,
    #     channels=128,
    #     out_channels=1,
    #     aggr="sum",
    #     norm="batch_norm",
    # ).to(config.DEVICE)
    # model = torch.compile(model)

    # # if you try out different RelBench tasks you will need to change these
    # loss_fn = L1Loss()
    # optimizer = torch.optim.AdamW(model.parameters(), lr = config.LR, weight_decay=config.WEIGHT_DECAY)
    # epochs = config.EPOCHS
    # database_name = "rel-f1"

    # train_val_on_all(task_to_train_info, model, optimizer, loss_fn)

    # dummy example (replace with actual RelBench associated stuff)
    node_embeddings = torch.randn((5,2))
    num_nodes = 5 
    num_edges = 2 
    adj_mat =  (torch.randn((5,5)) > 0.5).int()

    model = RelTransformer(node_embeddings, 
                           config.EMBED_DIM, 
                           config.NUM_LAYERS, 
                           config.NUM_HEADS, 
                           num_nodes, 
                           num_edges, 
                           adj_mat, 
                           config.DROPOUT)
       
    loss_fn = L1Loss()
    # fused AdamW
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and 'cuda' in config.DEVICE
    optimizer = torch.optim.AdamW(model.parameters(), lr = config.LR, weight_decay=config.WEIGHT_DECAY, fused=use_fused)
    
    epochs = config.EPOCHS
    database_name = "rel-f1"

    train_val_on_all(task_to_train_info, model, optimizer, loss_fn)

if __name__ == "__main__":
    main()

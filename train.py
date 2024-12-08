import torch 
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss, L1Loss

import copy 
import numpy as np
import config
import math
from baseline_models import BaselineModel
from dataset import create_task_train_dict

from torch_geometric.loader import NeighborLoader

def train(task, entity_table, model, loader: NeighborLoader, loss_fn, optimizer) -> float:
    model.train()

    loss_accum = count_accum = 0
    for batch in tqdm(loader):
        batch = batch.to(config.DEVICE)
        optimizer.zero_grad()
        pred = model(
            batch,
            task.entity_table,
        )
        pred = pred.view(-1) if pred.size(1) == 1 else pred
        
        loss = loss_fn(pred.float(), batch[entity_table].y.float())
        loss.backward()
        optimizer.step()

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
    for task, train_items in task_to_train_info.items():
        state_dict = None
        higher_is_better = config.HIGHER_IS_BETTER[train_items.task_metrics[0]]
        best_val_metric = -math.inf if higher_is_better else math.inf
        tune_metric = train_items.task_metrics[0].__name__
        for epoch in range(1, config.EPOCHS + 1):
            train_loss = train(train_items.task, train_items.entity_table, model, train_items.loader_dict["train"], loss_fn, optimizer)
            val_loss, val_pred = val(train_items.task, train_items.entity_table, model, train_items.loader_dict["val"], loss_fn)
            val_metrics = train_items.task.evaluate(val_pred, train_items.val_table)
            print(f"--------Task: {task}--------")
            print(f"Epoch: {epoch:02d}, Train loss: {train_loss}, Val metrics: {val_metrics}")
            
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

    model = BaselineModel(
        data=hetero_graph,
        col_stats_dict=col_stats_dict,
        gnn_layer = "HeteroGAT",
        num_layers=2,
        channels=128,
        out_channels=1,
        aggr="sum",
        norm="batch_norm",
    ).to(config.DEVICE)

    # if you try out different RelBench tasks you will need to change these
    loss_fn = L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr = config.LR)
    epochs = config.EPOCHS
    database_name = "rel-f1"

    train_val_on_all(task_to_train_info, model, optimizer, loss_fn)


if __name__ == "__main__":
    main()

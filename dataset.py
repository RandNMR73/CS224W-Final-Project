import numpy as np
import os
from collections import namedtuple

from relbench.datasets import get_dataset
from relbench.tasks import get_task
from relbench.modeling.utils import get_stype_proposal

from torch_frame.config.text_embedder import TextEmbedderConfig
from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.modeling.graph import get_node_train_table_input
from torch_geometric.loader import NeighborLoader

from text_embeddings import GloveTextEmbedding

import config
import torch

# DDP dataloader
class DataLoaderLite:
    def __init__(self, B, process_rank, num_processes):
        self.B = B
        self.process_rank = process_rank 
        self.num_processes = num_processes 
        self.current_position = self.B * self.process_rank

    def next_batch(self):
        pass  

def load_task_dataset(dataset_name, task_name):
    """
    Load the dataset and task for the given dataset and task name.

    Args:
        dataset_name (str): Name of the dataset.
        task_name (str): Name of the task.
    """

    task = get_task(dataset_name, task_name, download=True)

    train_table = task.get_table("train")
    val_table = task.get_table("val")
    test_table = task.get_table("test")

    return task, train_table, val_table, test_table

def get_data_graph(dataset):
    """
    Given dataset, get heterogenous graph for dataset and column statistics of each table
    """
    db = dataset.get_db()
    col_to_stype_dict = get_stype_proposal(db)

    text_embedder_cfg = TextEmbedderConfig(
        text_embedder=GloveTextEmbedding(device=config.DEVICE), batch_size=256
    )

    # Get the heterogeneous graph and column statistics
    hetero_graph, col_stats_dict = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=col_to_stype_dict,  # speficied column types
        text_embedder_cfg=text_embedder_cfg,  # our chosen text encoder
        cache_dir=os.path.join(
            config.ROOT_DIR, f"rel-f1_materialized_cache"
        ),  # store materialized graph for convenience
    )

    return hetero_graph, col_stats_dict

def load_data(hetero_graph, task, train_table, val_table, test_table):
    """
    Return a single dictionary containing the dataset, task, train_table, val_table, 
    test_table, hetero_graph, and col_stats_dict

    Args:
        hetero_graph (HeteroData): HeteroData object
        dataset (Dataset): Dataset object
        task (Task): Task object
        train_table (Table): Train Table object
        val_table (Table): Validation Table object
        test_table (Table): Test Table object
    """
    
    loader_dict = {}

    for split, table in [
        ("train", train_table),
        ("val", val_table),
        ("test", test_table),
    ]:
        table_input = get_node_train_table_input(
            table=table,
            task=task,
        )
        entity_table = table_input.nodes[0]
        loader_dict[split] = NeighborLoader(
            hetero_graph,
            num_neighbors=[
                config.NEIGHBORS_PER_NODE for i in range(config.DEPTH)
            ],  # we sample subgraphs of depth 2, 128 neighbors per node.
            time_attr="time",
            input_nodes=table_input.nodes,
            input_time=table_input.time,
            transform=table_input.transform,
            batch_size=config.BATCH_SIZE,
            temporal_strategy="uniform",
            shuffle = split == "train",
            num_workers=config.NUM_WORKERS,
            persistent_workers=False,
        )
    
    return loader_dict, entity_table

TaskTrainValInfo = namedtuple("TaskTrainValInfo", ["task", "task_metrics", "loader_dict", 
                                                        "col_stats_dict", "entity_table", "val_table"])
def create_task_train_dict(dataset_name):
    """
    Create a dictionary mapping task names to their training information.

    This function loads the specified dataset, retrieves the heterogeneous graph and column statistics,
    and then iterates through the tasks associated with the dataset. For each task, it loads the task dataset,
    prepares the data loaders, and stores the task information in a structured format.

    Args:
        dataset_name (str): The name of the dataset for which to create the task training dictionary.

    Returns:
        tuple: A tuple containing:
            - hetero_graph (HeteroData): The heterogeneous graph representation of the dataset.
            - col_stats_dict (dict): A dictionary containing column statistics for each table in the dataset.
            - task_to_train_info (dict): A dictionary mapping task names to their corresponding training information,
              which includes the task object, metrics, data loaders, column statistics, entity table, and validation table.
    """
    task_to_train_info = {}
    dataset = get_dataset(dataset_name, download=True)
    hetero_graph, col_stats_dict = get_data_graph(dataset)
    for task_name in config.RELBENCH_DATASETS[dataset_name]:
        task, train_table, val_table, test_table = load_task_dataset(dataset_name, task_name)
        loader_dict, entity_table = load_data(hetero_graph, task, train_table, val_table, test_table)
        task_to_train_info[task_name] = TaskTrainValInfo(task, task.metrics, loader_dict, col_stats_dict, entity_table, val_table)

    return hetero_graph, col_stats_dict, task_to_train_info

def main():
    hetero_graph, col_stats_dict, task_to_train_info = create_task_train_dict("rel-f1")
    print(task_to_train_info)

if __name__ == "__main__":
    main()
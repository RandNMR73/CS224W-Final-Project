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

from src.embeddings.text_embeddings import GloveTextEmbedding

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

    Returns:
        tuple: A tuple containing the task and its associated train, validation, and test tables.
    """

    task = get_task(dataset_name, task_name, download=True)

    train_table = task.get_table("train")
    val_table = task.get_table("val")
    test_table = task.get_table("test")

    return task, train_table, val_table, test_table

def get_data_graph(dataset, dataset_name):
    """
    Create a heterogeneous graph and retrieve column statistics from the dataset.

    Args:
        dataset (Dataset): The dataset object.
        dataset_name (str): The name of the dataset.

    Returns:
        tuple: A tuple containing the heterogeneous graph and column statistics dictionary.
    """
    db = dataset.get_db()
    col_to_stype_dict = get_stype_proposal(db)

    # Configure the text embedder
    text_embedder_cfg = TextEmbedderConfig(
        text_embedder=GloveTextEmbedding(device=config.DEVICE), batch_size=256
    )

    # Create the heterogeneous graph and retrieve column statistics
    hetero_graph, col_stats_dict = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=col_to_stype_dict,  # speficied column types
        text_embedder_cfg=text_embedder_cfg,  # our chosen text encoder
        cache_dir=os.path.join(
            config.ROOT_DIR, f"{dataset_name}_materialized_cache"
        ),  # Store materialized graph for convenience
    )

    return hetero_graph, col_stats_dict

def load_data(hetero_graph, task, train_table, val_table, test_table):
    """
    Load data into a dictionary containing the dataset, task, and data loaders.

    Args:
        hetero_graph (HeteroData): HeteroData object.
        task (Task): Task object.
        train_table (Table): Train Table object.
        val_table (Table): Validation Table object.
        test_table (Table): Test Table object.

    Returns:
        tuple: A tuple containing the loader dictionary and the entity table.
    """
    loader_dict = {}

    # Iterate through train, validation, and test splits
    for split, table in [
        ("train", train_table),
        ("val", val_table),
        ("test", test_table),
    ]:
        # Prepare the input for the node training table
        table_input = get_node_train_table_input(
            table=table,
            task=task,
        )
        entity_table = table_input.nodes[0]  # Get the first node as the entity table
        loader_dict[split] = NeighborLoader(
            hetero_graph,
            num_neighbors=[
                config.NEIGHBORS_PER_NODE for i in range(config.DEPTH)
            ],  # Sample subgraphs of specified depth and neighbors
            time_attr="time",
            input_nodes=table_input.nodes,
            input_time=table_input.time,
            transform=table_input.transform,
            batch_size=config.BATCH_SIZE,
            temporal_strategy="uniform",
            shuffle=split == "train",  # Shuffle only for training
            num_workers=config.NUM_WORKERS,
            persistent_workers=False,
        )
    
    return loader_dict, entity_table

# Named tuple to hold task training information
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
    hetero_graph, col_stats_dict = get_data_graph(dataset, dataset_name)
    for task_name in config.RELBENCH_DATASETS[dataset_name]:
        task, train_table, val_table, test_table = load_task_dataset(dataset_name, task_name)
        loader_dict, entity_table = load_data(hetero_graph, task, train_table, val_table, test_table)
        task_to_train_info[task_name] = TaskTrainValInfo(task, task.metrics, loader_dict, col_stats_dict, entity_table, val_table)

    return hetero_graph, col_stats_dict, task_to_train_info

def main():
    """
    Main function to execute the task training dictionary creation and print the results.
    """
    hetero_graph, col_stats_dict, task_to_train_info = create_task_train_dict("rel-f1")
    print(task_to_train_info)

if __name__ == "__main__":
    main()
import numpy as np
import os

from relbench.datasets import get_dataset
from relbench.tasks import get_task
from relbench.modeling.utils import get_stype_proposal

from torch_frame.config.text_embedder import TextEmbedderConfig
from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.modeling.graph import get_node_train_table_input
from torch_geometric.loader import NeighborLoader

from text_embeddings import GloveTextEmbedding

import config

def load_dataset(dataset_name, task_name):
    """
    Load the dataset and task for the given dataset and task name.

    Args:
        dataset_name (str): Name of the dataset.
        task_name (str): Name of the task.
    """

    dataset = get_dataset("rel-f1", download=True)
    task = get_task("rel-f1", "driver-position", download=True)

    train_table = task.get_table("train")
    val_table = task.get_table("val")
    test_table = task.get_table("test")

    return dataset, task, train_table, val_table, test_table

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

def load_data(dataset, task, train_table, val_table, test_table):
    """
    Return a single dictionary containing the dataset, task, train_table, val_table, 
    test_table, hetero_graph, and col_stats_dict

    Args:
        dataset (Dataset): Dataset object
        task (Task): Task object
        train_table (Table): Train Table object
        val_table (Table): Validation Table object
        test_table (Table): Test Table object
    """
    
    hetero_graph, col_stats_dict = get_data_graph(dataset)
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
    
    return loader_dict, col_stats_dict, entity_table

def create_loader_dicts():
    task_to_train_info = {}
    for database_name, tasks in config.RELBENCH_DATASETS.items():
        for task in tasks:
            dataset, task, train_table, val_table, test_table = load_dataset(database_name, task)
            loader_dict, col_stats_dict, entity_table = load_data(dataset, task, train_table, val_table, test_table)
            task_to_train_info[task] = (task.entity_table, loader_dict, col_stats_dict, entity_table)
        
    return task_to_train_info

def main():
    task_to_train_info = create_loader_dicts()
    print(task_to_train_info)

if __name__ == "__main__":
    main()
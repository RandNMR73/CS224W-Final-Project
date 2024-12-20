from typing import Any, Dict, List, Optional
import copy

import torch
from torch import Tensor
from torch.nn import BCEWithLogitsLoss
from torch.nn import Embedding, ModuleDict

from torch_geometric.nn import HeteroConv, LayerNorm, GCNConv, GATConv, SAGEConv
from torch_geometric.typing import EdgeType, NodeType
import numpy as np

from typing import Any, Dict, List

from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP

from relbench.modeling.nn import HeteroEncoder, HeteroTemporalEncoder

import config
import time

from model_PyG import HeteroRelTransformer

class RGCN(torch.nn.Module):
    """
    Implementation of heterogeneous GCN.
    """

    def __init__(
        self,
        node_types: List[NodeType],
        edge_types: List[EdgeType],
        channels: int,
        aggr: str = "mean",
        num_layers: int = 2,
    ):
        """
        Initialize the RGCN model.

        Args:
            node_types (List[NodeType]): List of node types in the graph.
            edge_types (List[EdgeType]): List of edge types in the graph.
            channels (int): Number of channels for the convolution layers.
            aggr (str): Aggregation method to use (default is "mean").
            num_layers (int): Number of layers in the model (default is 2).
        """
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            # Create a HeteroConv layer for each layer in the model
            conv = HeteroConv(
                {
                    edge_type: SAGEConv((channels, channels), channels, aggr=aggr)
                    for edge_type in edge_types
                },
                aggr="sum",
            )
            self.convs.append(conv)

        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            norm_dict = torch.nn.ModuleDict()
            for node_type in node_types:
                # Layer normalization for each node type
                norm_dict[node_type] = LayerNorm(channels, mode="node")
            self.norms.append(norm_dict)

    def reset_parameters(self):
        """
        Reset the parameters of the model layers.
        """
        for conv in self.convs:
            conv.reset_parameters()
        for norm_dict in self.norms:
            for norm in norm_dict.values():
                norm.reset_parameters()

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[NodeType, Tensor],
        num_sampled_nodes_dict: Optional[Dict[NodeType, List[int]]] = None,
        num_sampled_edges_dict: Optional[Dict[EdgeType, List[int]]] = None,
    ) -> Dict[NodeType, Tensor]:
        """
        Forward pass through the model.

        Args:
            x_dict (Dict[NodeType, Tensor]): Node features for each node type.
            edge_index_dict (Dict[NodeType, Tensor]): Edge indices for each node type.
            num_sampled_nodes_dict (Optional[Dict[NodeType, List[int]]]): Sampled nodes per type.
            num_sampled_edges_dict (Optional[Dict[EdgeType, List[int]]]): Sampled edges per type.

        Returns:
            Dict[NodeType, Tensor]: Updated node features after the forward pass.
        """
        for _, (conv, norm_dict) in enumerate(zip(self.convs, self.norms)):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: norm_dict[key](x) for key, x in x_dict.items()}
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        return x_dict

class HeteroGAT(torch.nn.Module):
    """
    Implementation of heterogeneous GAT.
    """
    def __init__(
        self,
        node_types: List[NodeType],
        edge_types: List[EdgeType],
        channels: int,
        aggr: str = "mean",
        heads: int = 1, 
        num_layers: int = 2,
    ):
        """
        Initialize the HeteroGAT model.

        Args:
            node_types (List[NodeType]): List of node types in the graph.
            edge_types (List[EdgeType]): List of edge types in the graph.
            channels (int): Number of channels for the convolution layers.
            aggr (str): Aggregation method to use (default is "mean").
            heads (int): Number of attention heads (default is 1).
            num_layers (int): Number of layers in the model (default is 2).
        """
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            # Create a HeteroConv layer for GAT with specified number of attention heads
            conv = HeteroConv(
                {
                    edge_type: GATConv(
                        (channels, channels), channels, heads=heads, add_self_loops=False
                    )
                    for edge_type in edge_types
                },
                aggr="sum",
            ) 
            self.convs.append(conv)

        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            norm_dict = torch.nn.ModuleDict()
            for node_type in node_types:
                # Layer normalization for each node type
                norm_dict[node_type] = LayerNorm(channels, mode="node")
            self.norms.append(norm_dict)

    def reset_parameters(self):
        """
        Reset the parameters of the model layers.
        """
        for conv in self.convs:
            conv.reset_parameters()
        for norm_dict in self.norms:
            for norm in norm_dict.values():
                norm.reset_parameters()

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[NodeType, Tensor],
        num_sampled_nodes_dict: Optional[Dict[NodeType, List[int]]] = None,
        num_sampled_edges_dict: Optional[Dict[EdgeType, List[int]]] = None,
    ) -> Dict[NodeType, Tensor]:
        """
        Forward pass through the model.

        Args:
            x_dict (Dict[NodeType, Tensor]): Node features for each node type.
            edge_index_dict (Dict[NodeType, Tensor]): Edge indices for each node type.
            num_sampled_nodes_dict (Optional[Dict[NodeType, List[int]]]): Sampled nodes per type.
            num_sampled_edges_dict (Optional[Dict[EdgeType, List[int]]]): Sampled edges per type.

        Returns:
            Dict[NodeType, Tensor]: Updated node features after the forward pass.
        """
        for _, (conv, norm_dict) in enumerate(zip(self.convs, self.norms)):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: norm_dict[key](x) for key, x in x_dict.items()}
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        return x_dict

def process_hetero_batch_nodes(x_dict, batch):
    """
    Process the nodes in a heterogeneous batch.

    Args:
        x_dict (Dict[NodeType, Tensor]): A dictionary mapping node types to their embeddings.
        batch (HeteroData): The batch containing sampled nodes information.

    Returns:
        Tuple[List[int], Dict[NodeType, Dict[int, Tensor]], List[int]]:
            - A list of total node counts for each node type. 
            - A dictionary mapping node types to their features for each hop.
            - A list of counts of nodes per hop.
    """
    node_type_counts = {node_type: 0 for node_type in x_dict.keys()}  # Dictionary that maps node type to number of nodes for that node type
    node_features_dict = {}  # Dictionary to store node features for each node type and hop {0: {A: [node features for node type A]}, 1: []}
    hop_node_counts = [0] * len(list(batch.num_sampled_nodes_dict.values())[0])   # List to track number of nodes per hop
    print(hop_node_counts)
    # Iterate over all node types and their sampled node counts
    for node_type, hop_node_counts_for_node_type in batch.num_sampled_nodes_dict.items():
        if sum(hop_node_counts_for_node_type):  # Process only if there are sampled nodes      
            # Calculate ranges of nodes between hops
            ends = [sum(hop_node_counts_for_node_type[:i+2]) for i in range(len(hop_node_counts_for_node_type) - 1)]
            tmp = hop_node_counts_for_node_type[0]
            hop_node_counts_for_node_type[0] = 0
            starts = [0] * (len(hop_node_counts_for_node_type) - 1)
            starts = [sum(hop_node_counts_for_node_type[:i+1]) for i in range(len(hop_node_counts_for_node_type) - 1)]
            ranges = []
            for start, end in zip(starts, ends):
                ranges.append((start, end))
            
            if ranges[0] == ranges[1]: ranges = [ranges[0]]
            hop_node_counts_for_node_type[0] = tmp
            
            node_features_hop = {}  # Dictionary to store features for each hop for the current node type
            
            # Extract node features for each hop
            for i, hop_range in enumerate(ranges):
                start, end = hop_range
                node_features_hop[i] = x_dict[node_type][start:end]
            for hop in range(len(hop_node_counts_for_node_type)):
                if hop_node_counts_for_node_type[hop]: hop_node_counts[hop] += hop_node_counts_for_node_type[hop]
            node_features_dict[node_type] = node_features_hop  # Store features by hop for this node type
    
        node_type_counts[node_type] = x_dict[node_type].size(0)  # Total nodes for this node type

    return node_type_counts, node_features_dict, hop_node_counts

def compute_node_offsets(batch, node_type_counts):
    """
    Compute global offsets for node indexing based on the counts of each node type.

    Args:
        batch (HeteroData): The batch containing sampled nodes information.
        node_type_counts (List[int]): A list of counts of nodes for each node type.

    Returns:
        Dict[NodeType, int]: A dictionary mapping node types to their global offsets.
    """
    # Compute offsets for global node indexing
    offsets = {}  # Dictionary to map node types to their global offset
    current_offset = 0  # Start offset

    for node_type, count in node_type_counts.items():
        if count > 0:  # Only process node types with non-zero nodes
            offsets[node_type] = current_offset  # Assign current offset to the node type
            current_offset += count  # Update offset for the next node type

    return offsets

def process_hetero_edges(batch):
    """
    Process the edges in a heterogeneous batch.

    Args:
        batch (HeteroData): The batch containing sampled edges information.

    Returns:
        Dict[EdgeType, Dict[int, Tensor]]: A dictionary mapping relation types to edge indices for each edge type and hop.
    """
    edge_index_dict = {}  # Dictionary to that maps relation types to edge indices for each edge type and hop

    # Extract edge indices for each edge type and hop
    for relation_type, relation_type_hop_counts in batch.num_sampled_edges_dict.items():
        cumsum_rel_types = torch.cumsum(torch.tensor(relation_type_hop_counts), dim=0)  # Cumulative edge counts
        if sum(cumsum_rel_types):  # Process only if there are edges
            rel_type_hop_edge_index = {}  # Store edge indices for each hop
            
            # Extract edges for each hop
            for hop in range(len(relation_type_hop_counts)):
                if relation_type_hop_counts[hop]:  # Skip hops with zero edges
                    start = cumsum_rel_types[hop - 1] if hop > 0 else 0  # Start index for this hop
                    end = cumsum_rel_types[hop]  # End index for this hop
                    
                    # Slice the edge index tensor for this hop
                    rel_type_hop_edge_index[hop] = batch[relation_type].edge_index[:, start:end].clone()  # Clone to avoid in-place modification
            
            edge_index_dict[relation_type] = rel_type_hop_edge_index  # Store edges by hop for this relation type
    
    return edge_index_dict

def process_hetero_batch(x_dict, batch: HeteroData, emb_dim):
    """
    Process a heterogeneous batch of nodes and edges.

    Args:
        x_dict (Dict[NodeType, Tensor]): A dictionary mapping node types to their embeddings.
        batch (HeteroData): The batch containing sampled nodes and edges information.
        emb_dim (int): The dimensionality of the embeddings.

    Returns:
        None: This function modifies the edge_index in place and does not return a value.
    """
    node_type_counts, node_features_dict, hop_node_counts = process_hetero_batch_nodes(x_dict, batch)
    offsets = compute_node_offsets(batch, node_type_counts)
    edge_index_dict = process_hetero_edges(batch)

    # Determine the number of hops (assume all edge types have the same number of hops)
    num_hops = len(list(batch.num_sampled_edges_dict.values())[0])
    edge_index = [torch.empty((2, 0), dtype=torch.long, device=config.DEVICE) for _ in range(num_hops)]  # Empty edge index for each hop

    # Calculate number of nodes per edge hop
    nodes_per_hop_edge, node_features = [0] * len(edge_index), [0] * len(edge_index)

    for i in range(len(nodes_per_hop_edge)):
        nodes_per_hop_edge[i] = hop_node_counts[i] + hop_node_counts[i + 1]
        node_features[i] = torch.zeros((nodes_per_hop_edge[i], emb_dim))

    # Combine edge indices across all relation types and hops
    for relation_type, rel_type_hop_edge_index in edge_index_dict.items():
        h_type = relation_type[0]  # Source node type
        t_type = relation_type[2]  # Target node type

        for hop, hop_edge_index in rel_type_hop_edge_index.items():
            h_offset = offsets[h_type]  # Global offset for target nodes

            # Add offsets to source and target node indices to make them global
            hop_edge_index = hop_edge_index.clone()  # Ensure no in-place modification
            hop_edge_index[0] += h_offset  # Offset for source nodes
            t_offset = offsets[t_type]  # Global offset for target nodes
            hop_edge_index[1] += t_offset 

            # Concatenate the new edges to the existing edge_index for the current hop
            edge_index[hop] = torch.cat((edge_index[hop], hop_edge_index), dim=1)

            # Populate node features for the current hop
            for h_node_idx, t_node_idx in zip(hop_edge_index[0], hop_edge_index[1]):
                if node_features_dict[h_type][hop].size(0):
                    if h_node_idx >= node_features[hop].shape[0]:
                        # Handle case where index is out of bounds
                        pass
                    else:
                        try:
                            # Attempt to access the element
                            node_features[hop][h_node_idx] = node_features_dict[h_type][hop][h_node_idx - h_offset]
                        except IndexError as e:
                            # Print debugging information
                            print("IndexError encountered!")
                            print(f"hop: {hop}")
                            print(f"h_node_idx: {h_node_idx}")
                            print(f"h_offset: {h_offset}")
                            print(f"node_features: {node_features}")
                            print(f"Calculated index: {h_node_idx - h_offset}")
                            print(f"node_features_dict[h_type][hop] shape: {node_features_dict[h_type][hop].shape}")
                            print(f"node_features shape: {node_features[hop].shape}")
                            print(f"hop_edge_index[0]: {hop_edge_index[0]}")
                            
                            # Terminate the program
                            raise  # Re-raise the exception after logging information

                if node_features_dict[t_type][hop].size(0):
                    node_features[hop][t_node_idx] = node_features_dict[t_type][hop][t_node_idx - t_offset]

    return node_features, edge_index

def num_node_features_get(self):
    # Return the number of node features for each node type
    return {
        key: store.num_node_features
        for key, store in self._node_store_dict.items()
    }

def num_node_features_set(self, new_features_dict):
    # Adjust each store.x according to new feature counts
    for key, new_num_feats in new_features_dict.items():
        store = self._node_store_dict[key]
        
        if store.x is None:
            # Create a new feature matrix if it doesn't exist
            num_nodes = store.num_nodes
            store.x = torch.zeros((num_nodes, new_num_feats))
        else:
            # Reshape or recreate the existing feature matrix
            old_num_feats = store.x.size(1)
            if new_num_feats > old_num_feats:
                # Increase the number of features
                num_nodes = store.x.size(0)
                new_x = torch.zeros((num_nodes, new_num_feats), dtype=store.x.dtype, device=store.x.device)
                new_x[:, :old_num_feats] = store.x
                store.x = new_x
            elif new_num_feats < old_num_feats:
                # Decrease the number of features by truncation
                store.x = store.x[:, :new_num_feats]

# Monkey patch the property on HeteroData
HeteroData.num_node_features = property(num_node_features_get, num_node_features_set)

def process_hetero_batch_vectorized(x_dict, batch: HeteroData, emb_dim):
    # print(f"hetero edge_index: {batch.edge_index_dict}")
    # for relation_type, edge_index in batch.edge_index_dict.items():
    #     if edge_index.size(1) != 0:
    #         print(relation_type)
    # og_num_node_features = batch.num_node_features
    # new_num_node_features = {'constructor_results': 9, 'drivers': 1, 'races': 2, 'standings': 3, 'results': 4, 'constructors': 5, 'qualifying': 6, 'constructor_standings': 7, 'circuits': 8}
    # print(f"new_num_node_features: {new_num_node_features}")
    # batch.num_node_features = new_num_node_features  # NOT SETTING??
    # print(f"batch.num_node_features before to_homogeneous: {batch.num_node_features}")
    # print(f"batch.n_id: {batch.n_id}")
    # print(type(batch))
    # print(f"batch.n_id: {batch.n_id}")
    # homo = to_homogeneous(batch)
    # print(f"homo.node_type: {homo.node_type}")

    # For heterogeneous batch
    hetero_node_counts = {
        node_type: batch[node_type].num_nodes
        for node_type in batch.node_types
    }

    homo = batch.to_homogeneous()  # Convert to homogeneous representation

    # For homogeneous graph
    homo_node_types, homo_node_counts = torch.unique(homo.node_type, return_counts=True)
    homo_node_counts_dict = {
        batch.node_types[type_idx.item()]: count.item()
        for type_idx, count in zip(homo_node_types, homo_node_counts)
    }

    # Map batch_id to node_type string
    orig_id_to_node_type = {orig_id: node_type for orig_id, node_type in enumerate(homo.node_type)}  
    orig_id_to_str = {orig_id: batch.node_types[type_idx.item()] for orig_id, type_idx in enumerate(homo.node_type)}
    
    node_features = torch.zeros((len(homo.n_id), emb_dim))  # Initialize node features tensor
    for orig_id, (new_id, node_type) in enumerate(zip(homo.n_id, homo.node_type)):
        string_node_type = orig_id_to_str[orig_id]
        emb_vector = x_dict[string_node_type][new_id]  # Get embedding vector for the node
        node_features[orig_id] = emb_vector

    return node_features, homo.edge_index 

class BaselineModel(torch.nn.Module):
    def __init__(
        self,
        data: HeteroData,
        col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
        gnn_layer: str, 
        num_layers: int,
        channels: int,
        out_channels: int,
        aggr: str,
        norm: str,
        # List of node types to add shallow embeddings to input
        shallow_list: List[NodeType] = [],
        # ID awareness
        id_awareness: bool = False,
    ):
        """
        Initialize the BaselineModel.

        Args:
            data (HeteroData): The heterogeneous data for the model.
            col_stats_dict (Dict[str, Dict[str, Dict[StatType, Any]]]): Column statistics for the model.
            gnn_layer (str): The type of GNN layer to use ("RGCN" or "HeteroGAT").
            num_layers (int): Number of layers in the GNN.
            channels (int): Number of channels for the GNN.
            out_channels (int): Number of output channels.
            aggr (str): Aggregation method for the GNN.
            norm (str): Normalization method for the GNN.
            shallow_list (List[NodeType], optional): List of node types for shallow embeddings (default is empty).
            id_awareness (bool, optional): Whether to use ID awareness (default is False).
        """
        super().__init__()
        self.channels = channels
        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                node_type: data[node_type].tf.col_names_dict
                for node_type in data.node_types
            },
            node_to_col_stats=col_stats_dict,
        )
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[
                node_type for node_type in data.node_types if "time" in data[node_type]
            ],
            channels=channels,
        )
        if gnn_layer == "RGCN":
            gnn = RGCN
        elif gnn_layer == "HeteroGAT":
            gnn = HeteroGAT
        else:
            raise ValueError(f"Unknown GNN layer: {gnn_layer}")

        self.gnn = gnn(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=channels,
            aggr=aggr,
            num_layers=num_layers,
        )
        self.head = MLP(
            channels,
            out_channels=out_channels,
            norm=norm,
            num_layers=1,
        )
        self.embedding_dict = ModuleDict(
            {
                node: Embedding(data.num_nodes_dict[node], channels)
                for node in shallow_list
            }
        )

        self.id_awareness_emb = None
        if id_awareness:
            self.id_awareness_emb = torch.nn.Embedding(1, channels)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset parameters for all components of the model.
        """
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.head.reset_parameters()
        for embedding in self.embedding_dict.values():
            torch.nn.init.normal_(embedding.weight, std=0.1)
        if self.id_awareness_emb is not None:
            self.id_awareness_emb.reset_parameters()

    def forward(self, batch: HeteroData, entity_table: NodeType) -> Tensor:
        """
        Forward pass through the BaselineModel.

        Args:
            batch (HeteroData): The batch of data for the model.
            entity_table (NodeType): The entity table to process.

        Returns:
            Tensor: The output tensor after processing the batch.
        """
        seed_time = batch[entity_table].seed_time

        x_dict = self.encoder(batch.tf_dict)

        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )

        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)

        # process_hetero_batch(x_dict, batch, self.channels)

        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
        )
        
        return self.head(x_dict[entity_table][: seed_time.size(0)])

    def forward_dst_readout(self, batch: HeteroData, entity_table: NodeType, dst_table: NodeType) -> Tensor:
        """
        Forward pass for destination readout.

        Args:
            batch (HeteroData): The batch of data for the model.
            entity_table (NodeType): The entity table to process.
            dst_table (NodeType): The destination table to read from.

        Returns:
            Tensor: The output tensor after processing the destination readout.
        """
        if self.id_awareness_emb is None:
            raise RuntimeError("id_awareness must be set True to use forward_dst_readout")
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)
        # Add ID-awareness to the root node
        x_dict[entity_table][: seed_time.size(0)] += self.id_awareness_emb.weight

        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )

        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)
        
        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
        )

        return self.head(x_dict[dst_table])

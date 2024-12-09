from typing import Any, Dict, List, Optional
import copy

import torch
from torch import Tensor
from torch.nn import BCEWithLogitsLoss
from torch.nn import Embedding, ModuleDict

from torch_geometric.nn import HeteroConv, LayerNorm, GCNConv, GATConv
from torch_geometric.typing import EdgeType, NodeType
import numpy as np

from typing import Any, Dict, List

from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP

from relbench.modeling.nn import HeteroEncoder, HeteroTemporalEncoder

import config
import time

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
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    edge_type: GCNConv(
                        channels, channels 
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
                norm_dict[node_type] = LayerNorm(channels, mode="node")
            self.norms.append(norm_dict)

    def reset_parameters(self):
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
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
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
                norm_dict[node_type] = LayerNorm(channels, mode="node")
            self.norms.append(norm_dict)

    def reset_parameters(self):
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
        for _, (conv, norm_dict) in enumerate(zip(self.convs, self.norms)):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: norm_dict[key](x) for key, x in x_dict.items()}
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        return x_dict

def process_hetero_batch_nodes(x_dict, batch):
    """
    {node_type: {hop_number : torch.tensor[node embeddings, ...]}, ...}
    """
    node_type_counts = []  # List to store the total number of nodes for each node type
    node_features_dict = {}  # Dictionary to store node features for each node type and hop
    hop_node_counts = [0] * len(list(batch.num_sampled_nodes_dict.values())[0])   #Dictionary to track number of nodes per hop

    # Iterate over all node types and their sampled node counts
    for node_type, node_counts in batch.num_sampled_nodes_dict.items():
        print(node_type, node_counts)
        if sum(node_counts):  # Process only if there are sampled nodes
            cumsum_nodes = torch.cumsum(torch.tensor(node_counts), dim=0)  # Compute cumulative sum of node counts
            node_features_hop = {}  # Dictionary to store features for each hop for the current node type
            
            # Extract node features for each hop
            for hop in range(len(node_counts)):
                if node_counts[hop]:  # Skip hops with zero nodes
                    start = cumsum_nodes[hop - 1].item() if hop > 0 else 0  # Start index for this hop
                    end = cumsum_nodes[hop].item()  # End index for this hop
                    node_features_hop[hop] = x_dict[node_type][start:end]  # Slice node features for this hop
                    hop_node_counts[hop] += node_counts[hop]
            node_features_dict[node_type] = node_features_hop  # Store features by hop for this node type
        
        node_type_counts.append(sum(node_counts))  # Total nodes for this node type
    print(f"hop_node_counts: {hop_node_counts}")
    return node_type_counts, node_features_dict, hop_node_counts

def compute_node_offsets(batch, node_type_counts):
    # Compute offsets for global node indexing
    offsets = {}  # Dictionary to map node types to their global offset
    current_offset = 0  # Start offset

    for node_type, count in zip(batch.num_sampled_nodes_dict.keys(), node_type_counts):
        if count > 0:  # Only process node types with non-zero nodes
            offsets[node_type] = current_offset  # Assign current offset to the node type
            current_offset += count  # Update offset for the next node type

    return offsets

def process_hetero_edges(batch):
    edge_index_dict = {}  # Dictionary to that maps relation types to edge indices for each edge type and hop

    # Extract edge indices for each edge type and hop
    for relation_type, relation_type_hop_counts in batch.num_sampled_edges_dict.items():
        cumsum_rel_types = torch.cumsum(torch.tensor(relation_type_hop_counts), dim=0)  # Cumulative edge counts
        if sum(cumsum_rel_types):  # Process only if there are edges
            rel_type_hop_edge_index = {}  # Dictionary to store edge indices for each hop
            
            # Extract edges for each hop
            for hop in range(len(relation_type_hop_counts)):
                if relation_type_hop_counts[hop]:  # Skip hops with zero edges
                    start = cumsum_rel_types[hop - 1] if hop > 0 else 0  # Start index for this hop
                    end = cumsum_rel_types[hop]  # End index for this hop
                    
                    # Slice the edge index tensor for this hop
                    indices = batch[relation_type].edge_index[:, start:end].clone()  # Clone to avoid in-place modification
                    rel_type_hop_edge_index[hop] = indices  # Store edge indices for this hop
            
            edge_index_dict[relation_type] = rel_type_hop_edge_index  # Store edges by hop for this relation type
    
    return edge_index_dict

def process_hetero_batch(x_dict, batch: HeteroData, emb_dim):
    node_type_counts, node_features_dict, hop_node_counts = process_hetero_batch_nodes(x_dict, batch)
    
    offsets = compute_node_offsets(batch, node_type_counts)

    edge_index_dict = process_hetero_edges(batch)

    # Determine the number of hops (assume all edge types have the same number of hops)
    num_hops = len(list(batch.num_sampled_edges_dict.values())[0])

    edge_index = [torch.empty((2, 0), dtype=torch.long, device=config.DEVICE) for _ in range(num_hops)]  # Empty edge index for each hop
    print(node_type_counts)
    print(edge_index)
    #Calculate number of nodes per edge hop (so number of nodes for hops 0 and 1, 1 and 2, etc. since edges are between hops)
    #List of n by d node features for each hop subgrpah
    nodes_per_hop_edge, node_features = [0] * len(edge_index), [0] * len(edge_index)

    for i in range(len(nodes_per_hop_edge)):
        nodes_per_hop_edge[i] = hop_node_counts[i] + hop_node_counts[i + 1]
        node_features[i] = torch.zeros((nodes_per_hop_edge[i], emb_dim))
        print(node_features[i].shape)
    
    # Combine edge indices across all relation types and hops
    for relation_type, rel_type_hop_edge_index in edge_index_dict.items():
        print(relation_type, rel_type_hop_edge_index)  # Debug: Print edge indices for this relation type
        
        for hop, hop_edge_index in rel_type_hop_edge_index.items():
            h_type = relation_type[0]  # Source node type
            t_type = relation_type[2]  # Target node type
            h_offset = offsets[h_type]  # Global offset for source nodes
            t_offset = offsets[t_type]  # Global offset for target nodes

            # Add offsets to source and target node indices to make them global
            hop_edge_index = hop_edge_index.clone()  # Ensure no in-place modification
            hop_edge_index[0] += h_offset
            hop_edge_index[1] += t_offset

            # Concatenate the new edges to the existing edge_index for the current hop
            edge_index[hop] = torch.cat((edge_index[hop], hop_edge_index), dim=1)
    
    raise ValueError()  # Placeholder: Raise an exception to stop execution

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
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.head.reset_parameters()
        for embedding in self.embedding_dict.values():
            torch.nn.init.normal_(embedding.weight, std=0.1)
        if self.id_awareness_emb is not None:
            self.id_awareness_emb.reset_parameters()

    def forward(
        self,
        batch: HeteroData,
        entity_table: NodeType,
    ) -> Tensor:
        seed_time = batch[entity_table].seed_time
        print(batch.tf_dict)
        x_dict = self.encoder(batch.tf_dict)

        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )

        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)
        process_hetero_batch(x_dict, batch, self.channels)
        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
        )
        
        return self.head(x_dict[entity_table][: seed_time.size(0)])

    def forward_dst_readout(
        self,
        batch: HeteroData,
        entity_table: NodeType,
        dst_table: NodeType,
    ) -> Tensor:
        if self.id_awareness_emb is None:
            raise RuntimeError(
                "id_awareness must be set True to use forward_dst_readout"
            )
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

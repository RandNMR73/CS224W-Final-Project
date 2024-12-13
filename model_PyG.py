from typing import Any, Dict, List, Optional

import torch
from torch import Tensor
from torch_geometric.nn import HeteroConv, LayerNorm, GATConv, MessagePassing
from torch_geometric.typing import EdgeType, NodeType

import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from typing import Any, Dict, List, Optional
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData

from relbench.modeling.nn import HeteroEncoder, HeteroTemporalEncoder
from torch.nn import Embedding, ModuleDict
from torch_geometric.nn import MLP

# General Architecture Stuff
# add regularization if needed (we can see based on the training dynamics of model) (using AdamW by default)
# sharing vs not sharing params for queries and keys 
# lr warmup and decay 
# check optimality of torch methods being called 

class MultiHeadAttention(nn.Module):
    def __init__(self, 
                 embed_dim, # config
                 num_heads, # config
                 num_nodes, # pass in directly
                 num_edges, # pass in directly 
                 dropout, # config
                 is_hh_att=False, 
                 is_he_att=False, 
                 is_eh_att=False, 
                 is_ee_att=False,
                 is_e_out=False):
        """
        Initializes the MultiHeadAttention module.

        Args:
            embed_dim (int): Dimension of the input embeddings.
            num_heads (int): Number of attention heads.
            num_nodes (int): Number of nodes in the graph.
            num_edges (int): Number of edges in the graph.
            dropout (float): Dropout rate for attention.
            is_hh_att (bool): Flag for using head-to-head attention.
            is_he_att (bool): Flag for using head-to-edge attention.
            is_eh_att (bool): Flag for using edge-to-head attention.
            is_ee_att (bool): Flag for using edge-to-edge attention.
        """
        super().__init__()
        assert embed_dim % num_heads == 0

        # We will initialize these layers for you, since swapping the ordering
        # would affect the random number generation (and therefore your exact
        # outputs relative to the autograder). Note that the layers use a bias
        # term, but this isn't strictly necessary (and varies by
        # implementation).
        
        self.key_h = nn.Linear(embed_dim, embed_dim, bias=False)
        self.query_h = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_h = nn.Linear(embed_dim, embed_dim, bias=False)

        self.key_e = nn.Linear(embed_dim, embed_dim, bias=False)
        self.query_e = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_e = nn.Linear(embed_dim, embed_dim, bias=False)
        
        self.attn_drop = nn.Dropout(dropout)

        self.n_head = num_heads
        self.emd_dim = embed_dim
        self.head_dim = self.emd_dim // self.n_head
        
        # additional params 
        self.is_hh_att = is_hh_att  # determines if the adjacency calculation is used 
        self.is_he_att = is_he_att
        self.is_eh_att = is_eh_att 
        self.is_ee_att = is_ee_att 
        self.is_e_out = is_e_out  # if the output of the computation gives new edge embeddings

        # find where to get the number of edges and the number of nodes in the graph from relbench (used for making sure that the shapes of the proj matrices match up)
        self.shape_proj_mat_1 = nn.Parameter(torch.empty(num_nodes, num_edges))  # (num_nodes, num_edges)
        self.shape_proj_mat_2 = nn.Parameter(torch.empty(num_nodes, num_edges))  # (num_nodes, num_edges)
        nn.init.xavier_uniform_(self.shape_proj_mat_1)
        nn.init.xavier_uniform_(self.shape_proj_mat_2)

    def forward(self, q, k, v, edge_index):
        """
        Forward pass for the MultiHeadAttention module.

        Args:
            q (Tensor): Query tensor of shape (N, E).
            k (Tensor): Key tensor of shape (N, E).
            v (Tensor): Value tensor of shape (N, E).

        Returns:
            Tensor: Output tensor after applying attention.
        """
        N, E = q.shape  # query.shape
        N, E = v.shape  # value.shape 
        H = self.n_head

        # new modifications 
        if self.is_hh_att:
            self.query = self.query_h 
            self.key = self.key_h
        
        if self.is_he_att: 
            self.query = self.query_h 
            self.key = self.key_e 
        
        if self.is_eh_att:
            self.query = self.query_e 
            self.key = self.key_h 
        
        if self.is_ee_att:
            self.query = self.query_e 
            self.key = self.key_e 
        
        if self.is_e_out:
            self.value = self.value_e
        else:
            self.value = self.value_h 

        query = self.query(q) # (N, E)
        key = self.key(k) # (N, E)
        value = self.value(v) # (N, E)

        # new modifications
        if self.is_hh_att:  # equivalent of a graph convolution 
            query = torch.sparse.mm(edge_index, query)  # might need to do conversion (pass in placeholder tensor torch.ones())
        
        if self.is_he_att:
            query = self.shape_proj_mat_1 @ query 
        
        if self.is_eh_att:
            query = self.shape_proj_mat_2 @ query 
        
        query, key, value = query.view(N, H, E//H), key.view(N, H, E//H), value.view(N, H, E//H) 
        
        # naive attn 
        # qk = torch.matmul(query, key.transpose(2,3))     
        # output = self.attn_drop(torch.softmax((qk) / math.sqrt(self.head_dim), dim=-1)) # (N, H, S, T)          
        # output = torch.matmul(output, value) # (N, H, E/H)

        # flash attn
        output = F.scaled_dot_product_attention(query, key, value)

        output = output.view(N, E)
        return output

# if we want to use later
class GeGLU(nn.Module):
    def __init__(self):
        """
        Initializes the GeGLU module.
        """
        super(GeGLU, self).__init__()

    def forward(self, x):
        """
        Forward pass for the GeGLU module.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying GeGLU activation.
        """
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)

class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout):
        """
        Initializes the FeedForward module.

        Args:
            n_embed (int): Dimension of the input embeddings.
            dropout (float): Dropout rate for the feedforward network.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed * 4),
            nn.GELU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """
        Forward pass for the FeedForward module.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying the feedforward network.
        """
        return self.net(x)

class Block(nn.Module):
    def __init__(self,
                 embed_dim, # config
                 num_heads, # config
                 num_nodes, # pass in directly
                 num_edges, # pass in directly
                 dropout_att, # config
                 dropout_ffwd,  # add to config 
                 ):
        """
        Initializes the Block module, which consists of multiple attention
        heads and a feedforward network.

        Args:
            embed_dim (int): Dimension of the input embeddings.
            num_heads (int): Number of attention heads.
            num_nodes (int): Number of nodes in the graph.
            num_edges (int): Number of edges in the graph.
            dropout_att (float): Dropout rate for attention.
            dropout_ffwd (float): Dropout rate for the feedforward network.
        """
        # n_embed, n_heads, head_size, dropout,
        super().__init__()
        # head_size = embed_dim // num_heads 

        self.sa_hh_h = MultiHeadAttention(embed_dim, num_heads, num_nodes, num_edges, dropout_att, is_hh_att=True)  # (N, H, E/H)
        self.sa_eh_h = MultiHeadAttention(embed_dim, num_heads, num_nodes, num_edges, dropout_att, is_eh_att=True)  # (N, H, E/H)

        self.sa_ee_e = MultiHeadAttention(embed_dim, num_heads, num_nodes, num_edges, dropout_att, is_ee_att=True, is_e_out=True)  # (N, H, E/H)
        self.sa_he_e = MultiHeadAttention(embed_dim, num_heads, num_nodes, num_edges, dropout_att, is_he_att=True, is_e_out=True)  # (N, H, E/H)

        self.ffwd_h = FeedForward(embed_dim, dropout_ffwd)
        self.ffwd_e = FeedForward(embed_dim, dropout_ffwd)
        
        self.rmsn1_h = nn.RMSNorm(embed_dim)
        self.rmsn1_e = nn.RMSNorm(embed_dim)
        self.rmsn2_h = nn.RMSNorm(embed_dim)
        self.rmsn2_e = nn.RMSNorm(embed_dim)
    
    def forward(self, x_node, x_edge, edge_index):
        """
        Forward pass for the Block module.

        Args:
            x_node (Tensor): Input tensor for node embeddings.
            x_edge (Tensor): Input tensor for edge embeddings.

        Returns:
            Tensor: Output tensor after applying attention and feedforward layers.
        """
        h = self.rmsn1_h(x_node) + x_node 
        e = self.rmsn1_e(x_edge) + x_edge 
        x_h = self.sa_hh_h(h, h, h, edge_index) + self.sa_eh_h(e, h, h, edge_index) + h # can add gating here 
        x_e = self.sa_ee_e(e, e, e, edge_index) + self.sa_he_e(h, e, e, edge_index) + e # can add gating here  # shouldn't be a problem in terms of efficiency but we can check       

        x_h = self.ffwd_h(self.rmsn2_h(x_h)) + x_h 
        x_e = self.ffwd_e(self.rmsn2_e(x_e)) + x_e
        return x_h, x_e 

class RelTransformer(MessagePassing):  # needed to call HeteroConv wrapper 
    def __init__(self, 
                n_embed, # config
                num_blocks, # config              
                num_heads, # config
                num_nodes, # pass in x.shape[0]
                num_edges, # pass in edge_index.shape[1]
                dropout):  # config 
        """
        Initializes the RelTransformer module, which consists of multiple
        blocks of attention and feedforward layers.

        Args:
            node_embeddings (Tensor): Initial node embeddings.
            n_embed (int): Dimension of the input embeddings.
            num_blocks (int): Number of blocks in the transformer.
            num_heads (int): Number of attention heads.
            num_nodes (int): Number of nodes in the graph.
            num_edges (int): Number of edges in the graph.
            dropout (float): Dropout rate for the transformer.
        """
        super().__init__()
        # process embeddings 
        self.num_nodes = num_nodes
        self.num_edges = num_edges 
        self.n_embed = n_embed 
        self.edge_embeddings = nn.Parameter(torch.empty(num_edges, n_embed))
        nn.init.kaiming_uniform_(self.edge_embeddings, a=1.7715, nonlinearity='leaky_relu')
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList()
        for _ in range(self.num_blocks):
            self.blocks.append(Block(n_embed, num_heads, num_nodes, num_edges, dropout_att=dropout, dropout_ffwd=dropout)) 
    
    # dummy function
    def message(self, x):
        pass 
    
    def forward(self, x, edge_index):  
        """
        Forward pass for the RelTransformer module.

        Returns:
            Tensor: Output tensor after passing through all transformer blocks.
        """
        out_h = x
        out_e = self.edge_embeddings 
        for block in self.blocks:
            out_h, out_e = block(out_h, out_e, edge_index)  
        return out_h

class HeteroRelTransformer(torch.nn.Module):
    """
    PyG-ified RelTransformer.
    """
    def __init__(
        self,
        edge_types: List[EdgeType],
        channels: int,
        num_blocks,
        num_heads,
        num_nodes,  # pass in x.shape[0]
        num_edges,  # pass in edge_index.shape[1]
        dropout,
        aggr="mean",
        num_layers: int = 4,
    ):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    edge_type: RelTransformer( 
                                              n_embed=channels, 
                                              num_blocks=num_blocks, 
                                              num_heads=num_heads,
                                              num_nodes=num_nodes,
                                              num_edges=num_edges,
                                              dropout=dropout)
                    for edge_type in edge_types
                },
                aggr=aggr,
            )
            self.convs.append(conv)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[NodeType, Tensor],
        num_sampled_nodes_dict: Optional[Dict[NodeType, List[int]]] = None,
        num_sampled_edges_dict: Optional[Dict[EdgeType, List[int]]] = None,
    ) -> Dict[NodeType, Tensor]:
        for conv in self.convs: 
            x_dict = conv(x_dict, edge_index_dict)

        return x_dict

class Model(torch.nn.Module):
    def __init__(
        self,
        data: HeteroData,
        col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
        num_layers: int,
        channels: int,
        out_channels: int,
        aggr: str,
        norm: str,
        num_blocks,
        num_heads,
        num_nodes, 
        num_edges, 
        dropout,
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

        self.gnn = HeteroRelTransformer( 
                                        edge_types=data.edge_types, 
                                        channels=channels, 
                                        num_blocks=num_blocks,
                                        num_heads=num_heads,
                                        num_nodes=num_nodes, 
                                        num_edges=num_edges, 
                                        dropout=dropout,
                                        aggr=aggr,
                                        num_layers=num_layers)
        
        # self.gnn = gnn(
        #     node_types=data.node_types,
        #     edge_types=data.edge_types,
        #     channels=channels,
        #     aggr=aggr,
        #     num_layers=num_layers,
        # )

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

    # make sure to adjust batch before HeteroData 
    def forward(
        self,
        batch: HeteroData,
        # batch_scuffed, 
        entity_table: NodeType,
    ) -> Tensor:
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

    def forward_dst_readout(
        self,
        batch: HeteroData,
        entity_table: NodeType,
        dst_table: NodeType,
    ) -> Tensor:
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
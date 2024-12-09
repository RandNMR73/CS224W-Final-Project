import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from typing import Any, Dict, List, Optional
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData

from relbench.modeling.nn import HeteroEncoder, HeteroTemporalEncoder
from baseline_models import process_hetero_batch

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
                 adj_mat, # pass in directly 
                 dropout, # config
                 is_hh_att=False, 
                 is_he_att=False, 
                 is_eh_att=False, 
                 is_ee_att=False):
        """
        Initializes the MultiHeadAttention module.

        Args:
            embed_dim (int): Dimension of the input embeddings.
            num_heads (int): Number of attention heads.
            num_nodes (int): Number of nodes in the graph.
            num_edges (int): Number of edges in the graph.
            adj_mat (Tensor): Adjacency matrix for the graph.
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
        
        self.attn_drop = nn.Dropout(dropout)

        self.n_head = num_heads
        self.emd_dim = embed_dim
        self.head_dim = self.emd_dim // self.n_head
        
        # additional params 
        self.is_hh_att = is_hh_att  # determines if the adjacency calculation is used 
        self.is_he_att = is_he_att
        self.is_eh_att = is_eh_att 
        self.is_ee_att = is_ee_att 

        self.adj_mat = adj_mat # find where to get adjacency matrix from relbench and pass in as param to constructor (function to convert coo tensor to adj matrix) (pass in the coo tensor and do sparse_mm instead of passing adj mat since we don't want to bloat memory either)
        self.shape_proj_1 = nn.Linear(num_edges, num_nodes, bias=False) # find where to get the number of edges and the number of nodes in the graph from relbench (used for making sure that the shapes of the proj matrices match up)
        self.shape_proj_2 = nn.Linear(num_nodes, num_edges, bias=False)

    def forward(self, q, k, v):
        """
        Forward pass for the MultiHeadAttention module.

        Args:
            q (Tensor): Query tensor of shape (N, E).
            k (Tensor): Key tensor of shape (N, E).
            v (Tensor): Value tensor of shape (N, E).

        Returns:
            Tensor: Output tensor after applying attention.
        """
        N, E = query.shape
        N, E = value.shape
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

        query = self.query(q).view(N, H, E//H) # (N, E) -> (N, H, E/H) -> (N, H, E/H)
        key = self.key(k).view(N, H, E//H) # (N, E) -> (N, H, E/H) -> (N, H, E/H)
        value = self.value(v).view(N, H, E//H) # (N, E) -> (N, H, E/H)
        
        # new modifications
        if self.is_hh_att:
            query = torch.sparse.mm(query, self.adj_mat)
        
        if self.is_he_att:
            key = self.proj_1(key)
        
        if self.is_eh_att:
            key = self.proj_2(key)    
        
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
                 adj_mat, # pass in directly 
                 dropout_att, # config
                 dropout_ffwd,
                 ):
        """
        Initializes the Block module, which consists of multiple attention
        heads and a feedforward network.

        Args:
            embed_dim (int): Dimension of the input embeddings.
            num_heads (int): Number of attention heads.
            num_nodes (int): Number of nodes in the graph.
            num_edges (int): Number of edges in the graph.
            adj_mat (Tensor): Adjacency matrix for the graph.
            dropout_att (float): Dropout rate for attention.
            dropout_ffwd (float): Dropout rate for the feedforward network.
        """
        # n_embed, n_heads, head_size, dropout,
        super().__init__()
        head_size = embed_dim // num_heads 
        self.sa_hh = MultiHeadAttention(embed_dim, num_heads, num_nodes, num_edges, adj_mat, dropout_att, is_hh_att=True)  # (N, H, E/H)
        self.sa_ee = MultiHeadAttention(embed_dim, num_heads, num_nodes, num_edges, adj_mat, dropout_att, is_ee_att=True)  # (N, H, E/H)
        self.sa_he = MultiHeadAttention(embed_dim, num_heads, num_nodes, num_edges, adj_mat, dropout_att, is_he_att=True)  # (N, H, E/H)
        self.sa_eh = MultiHeadAttention(embed_dim, num_heads, num_nodes, num_edges, adj_mat, dropout_att, is_eh_att=True)  # (N, H, E/H)
        self.ffwd = FeedForward(embed_dim, dropout_ffwd)
        self.rmsn1_n = nn.RMSNorm(embed_dim)
        self.rmsn1_e = nn.RMSNorm(embed_dim)
        self.rmsn2 = nn.RMSNorm(embed_dim)
    
    def forward(self, x_node, x_edge):
        """
        Forward pass for the Block module.

        Args:
            x_node (Tensor): Input tensor for node embeddings.
            x_edge (Tensor): Input tensor for edge embeddings.

        Returns:
            Tensor: Output tensor after applying attention and feedforward layers.
        """
        h = self.rmsn1_n(x_node)
        e = self.rmsn1_e(x_edge)
        x = self.sa_hh(h, h, h) + self.sa_ee(e, e, h) + self.sa_eh(e, h, h) + self.sa_he(h, e, h)  # can add gating here 
        x = self.ffwd(self.rmsn2(x))
        return x

class RelTransformer(nn.Module):
    def __init__(self, 
                data: HeteroData,
                col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
                node_embeddings, 
                n_embed, # config
                num_blocks, # config              
                num_heads, # config
                num_nodes, # pass in directly
                num_edges, # pass in directly
                adj_mat, # pass in directly 
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
            adj_mat (Tensor): Adjacency matrix for the graph.
            dropout (float): Dropout rate for the transformer.
        """
        super().__init__()
        # process RDB
        self.encoder = HeteroEncoder(
            channels=n_embed,
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
            channels=n_embed,
        )

        # process embeddings 
        self.node_embeddings = nn.Parameter(node_embeddings)
        self.num_nodes = node_embeddings.shape[0]  # N
        self.n_embed = n_embed 
        self.edge_embeddings = nn.Embedding(num_edges, n_embed)
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList()
        for _ in range(self.num_blocks):
            self.blocks.append(Block(n_embed, num_heads, num_nodes, num_edges, adj_mat, dropout)) 
    
    def forward(self, x_dict, batch):  # pass in node embeddings from subgraph sampling function (make sure that n_embed)
        """
        Forward pass for the RelTransformer module.

        Returns:
            Tensor: Output tensor after passing through all transformer blocks.
        """
        # pass in x_dict into Gabe's function to get the node embeddings and batch (HeteroData object) into fwd method (look in baseline_models.py)
        self.node_embeddings = process_hetero_batch(x_dict, batch, self.n_embed)
        out = self.node_embeddings
        for block in self.blocks:
            out = block(out, self.edge_embeddings)  # is there something weird about this? yes! what if we try learning an (n+m) x d tensor instead
        return out

# TODOS:
# DDP, add stuff to train.py file, figure out how 
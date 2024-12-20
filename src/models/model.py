import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from typing import Any, Dict, List, Optional
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData

from relbench.modeling.nn import HeteroEncoder, HeteroTemporalEncoder
from src.models.baseline_models import process_hetero_batch

# General Architecture Stuff
# Consider adding regularization if needed based on training dynamics (using AdamW by default)
# Evaluate sharing vs not sharing parameters for queries and keys 
# Implement learning rate warmup and decay 
# Check optimality of torch methods being called 

class MultiHeadAttention(nn.Module):
    def __init__(self, 
                 embed_dim,  # config
                 num_heads,  # config
                 num_nodes,  # pass in directly
                 num_edges,  # pass in directly
                 adj_mat,  # pass in directly 
                 dropout,  # config
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
        self.value_e = nn.Linear(embed_dim, embed_dim, bias=False)
        
        self.attn_drop = nn.Dropout(dropout)

        self.n_head = num_heads
        self.emd_dim = embed_dim
        self.head_dim = self.emd_dim // self.n_head
        
        # Additional parameters to control attention types
        self.is_hh_att = is_hh_att  # Determines if the adjacency calculation is used 
        self.is_he_att = is_he_att
        self.is_eh_att = is_eh_att 
        self.is_ee_att = is_ee_att 
        self.is_e_out = is_e_out  # If the output of the computation gives new edge embeddings

        # Adjacency matrix and projection matrices for node and edge embeddings
        self.adj_mat = adj_mat
        self.shape_proj_mat_1 = nn.Parameter(torch.empty(num_nodes, num_edges))  # (num_nodes, num_edges)
        self.shape_proj_mat_2 = nn.Parameter(torch.empty(num_nodes, num_edges))  # (num_nodes, num_edges)
        nn.init.xavier_uniform_(self.shape_proj_mat_1)
        nn.init.xavier_uniform_(self.shape_proj_mat_2)

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

        # Select appropriate query and key based on attention flags
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

        query = self.query(q)  # (N, E)
        key = self.key(k)  # (N, E)
        value = self.value(v)  # (N, E)

        # Apply modifications based on attention flags
        if self.is_hh_att:  # Equivalent of a graph convolution 
            query = torch.sparse.mm(query, self.adj_mat)  # Might need to do conversion (pass in placeholder tensor torch.ones())
        
        if self.is_he_att:
            key = self.shape_proj_mat_1 @ key 
        
        if self.is_eh_att:
            key = self.shape_proj_mat_2 @ key  
        
        query, key, value = query.view(N, H, E // H), key.view(N, H, E // H), value.view(N, H, E // H) 
        
        # naive attn 
        # qk = torch.matmul(query, key.transpose(2,3))     
        # output = self.attn_drop(torch.softmax((qk) / math.sqrt(self.head_dim), dim=-1)) # (N, H, S, T)          
        # output = torch.matmul(output, value) # (N, H, E/H)

        # flash attn
        output = F.scaled_dot_product_attention(query, key, value)

        output = output.view(N, E)
        return output

# If we want to use later
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
                 embed_dim,  # config
                 num_heads,  # config
                 num_nodes,  # pass in directly
                 num_edges,  # pass in directly
                 adj_mat,  # pass in directly 
                 dropout_att,  # config
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
            adj_mat (Tensor): Adjacency matrix for the graph.
            dropout_att (float): Dropout rate for attention.
            dropout_ffwd (float): Dropout rate for the feedforward network.
        """
        super().__init__()
        # Initialize multiple attention heads
        self.sa_hh_h = MultiHeadAttention(embed_dim, num_heads, num_nodes, num_edges, adj_mat, dropout_att, is_hh_att=True)  # (N, H, E/H)
        self.sa_ee_h = MultiHeadAttention(embed_dim, num_heads, num_nodes, num_edges, adj_mat, dropout_att, is_ee_att=True)  # (N, H, E/H)
        self.sa_he_h = MultiHeadAttention(embed_dim, num_heads, num_nodes, num_edges, adj_mat, dropout_att, is_he_att=True)  # (N, H, E/H)
        self.sa_eh_h = MultiHeadAttention(embed_dim, num_heads, num_nodes, num_edges, adj_mat, dropout_att, is_eh_att=True)  # (N, H, E/H)

        self.sa_hh_e = MultiHeadAttention(embed_dim, num_heads, num_nodes, num_edges, adj_mat, dropout_att, is_hh_att=True, is_e_out=True)  # (N, H, E/H)
        self.sa_ee_e = MultiHeadAttention(embed_dim, num_heads, num_nodes, num_edges, adj_mat, dropout_att, is_ee_att=True, is_e_out=True)  # (N, H, E/H)
        self.sa_he_e = MultiHeadAttention(embed_dim, num_heads, num_nodes, num_edges, adj_mat, dropout_att, is_he_att=True, is_e_out=True)  # (N, H, E/H)
        self.sa_eh_e = MultiHeadAttention(embed_dim, num_heads, num_nodes, num_edges, adj_mat, dropout_att, is_eh_att=True, is_e_out=True)  # (N, H, E/H)

        self.ffwd_h = FeedForward(embed_dim, dropout_ffwd)
        self.ffwd_e = FeedForward(embed_dim, dropout_ffwd)
        
        self.rmsn1_h = nn.RMSNorm(embed_dim)
        self.rmsn1_e = nn.RMSNorm(embed_dim)
        self.rmsn2_h = nn.RMSNorm(embed_dim)
        self.rmsn2_e = nn.RMSNorm(embed_dim)
    
    def forward(self, x_node, x_edge):
        """
        Forward pass for the Block module.

        Args:
            x_node (Tensor): Input tensor for node embeddings.
            x_edge (Tensor): Input tensor for edge embeddings.

        Returns:
            Tensor: Output tensor after applying attention and feedforward layers.
        """
        h = self.rmsn1_h(x_node)
        e = self.rmsn1_e(x_edge)
        # Compute attention outputs for nodes and edges
        x_h = self.sa_hh_h(h, h, h) + self.sa_ee_h(e, e, h) + self.sa_eh_h(e, h, h) + self.sa_he_h(h, e, h)  # can add gating here 
        x_e = self.sa_hh_e(h, h, h) + self.sa_ee_e(e, e, h) + self.sa_eh_e(e, h, h) + self.sa_he_e(h, e, h)  # can add gating here  
        
        # Apply feedforward networks
        x_h = self.ffwd_h(self.rmsn2_h(x_h))
        x_e = self.ffwd_e(self.rmsn2_h(x_e))
        return x_h, x_e 

class RelTransformer(nn.Module):
    def __init__(self, 
                data: HeteroData,
                col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
                node_embeddings, 
                n_embed,  # config
                num_blocks,  # config              
                num_heads,  # config
                num_nodes,  # pass in directly
                num_edges,  # pass in directly
                adj_mat,  # pass in directly 
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
        # Initialize encoders for heterogeneous data
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

        # Process embeddings 
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
        # Update node embeddings based on the input batch
        self.node_embeddings = process_hetero_batch(x_dict, batch, self.n_embed)
        out_h = self.node_embeddings
        out_e = self.edge_embeddings 
        for block in self.blocks:
            out_h, out_e = block(out_h, out_e)  # Process through each block
        return out_h

# TODOS:
# DDP, add stuff to train.py file, figure out how 
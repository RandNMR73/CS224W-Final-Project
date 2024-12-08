import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import config

# global params 
# embed_dim = config[]
dropout = config['DROPOUT']

# General Architecture Stuff
# Hyperparams in config file 
# Include GeLU / GeGLU
# Add FFN
# add regularization if needed (we can see based on the training dynamics of model)

# mixed precision

class MultiHeadAttention(nn.Module):
    """
    A model layer which implements a simplified version of masked attention, as
    introduced by "Attention Is All You Need" (https://arxiv.org/abs/1706.03762).

    Usage:
      attn = MultiHeadAttention(embed_dim, num_heads=2)

      # self-attention
      data = torch.randn(batch_size, sequence_length, embed_dim)
      self_attn_output = attn(query=data, key=data, value=data)

      # attention using two inputs
      other_data = torch.randn(batch_size, sequence_length, embed_dim)
      attn_output = attn(query=data, key=other_data, value=other_data)
    """

    def __init__(self, 
                 embed_dim, 
                 num_heads, 
                 num_nodes, 
                 num_edges, 
                 adj_mat, 
                 dropout=dropout, 
                 is_hh_att=False, 
                 is_he_att=False, 
                 is_eh_att=False, 
                 is_ee_att=False):
        """
        Construct a new MultiHeadAttention layer.

        Inputs:
         - embed_dim: Dimension of the token embedding
         - num_heads: Number of attention heads
         - dropout: Dropout probability
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

        self.adj_mat = adj_mat # find where to get adjacency matrix from relbench and pass in as param to constructor (function to convert coo tensor to adj matrix)
        self.shape_proj_1 = nn.Linear(num_edges, num_nodes, bias=False) # find where to get the number of edges and the number of nodes in the graph from relbench (used for making sure that the shapes of the proj matrices match up)
        self.shape_proj_2 = nn.Linear(num_nodes, num_edges, bias=False)

    def forward(self, q, k, v):
        """
        Calculate the masked attention output for the provided data, computing
        all attention heads in parallel.

        In the shape definitions below, N is the batch size, S is the source
        sequence length, T is the target sequence length, and E is the embedding
        dimension.

        Inputs:
        - query: Input data to be used as the query, of shape (N, S, E)
        - key: Input data to be used as the key, of shape (N, T, E)
        - value: Input data to be used as the value, of shape (N, T, E)
        - attn_mask: Array of shape (S, T) where mask[i,j] == 0 indicates token
          i in the source should not influence token j in the target.

        Returns:
        - output: Tensor of shape (N, S, E) giving the weighted combination of
          data in value according to the attention weights calculated using key
          and query.
        """
        N, E = query.shape
        N, E = value.shape
        H = self.n_head
        # Create a placeholder, to be overwritten by your code below.
        output = torch.empty((N, E))

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
            query = query @ self.adj_mat 
        
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

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed * 4),
            nn.GELU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embed, n_heads):
        super().__init__()
        head_size = n_embed // n_heads 
        self.sa_hh = MultiHeadAttention(n_heads, head_size, dropout=dropout, is_hh_att=True)  # (N, H, E/H)
        self.sa_ee = MultiHeadAttention(n_heads, head_size, dropout=dropout, is_ee_att=True)  # (N, H, E/H)
        self.sa_he = MultiHeadAttention(n_heads, head_size, dropout=dropout, is_he_att=True)  # (N, H, E/H)
        self.sa_eh = MultiHeadAttention(n_heads, head_size, dropout=dropout, is_eh_att=True)  # (N, H, E/H)
        self.ffwd = FeedForward(n_embed)
        self.rmsn1 = nn.RMSNorm(n_embed)
        self.rmsn2 = nn.RMSNorm(n_embed)
    
    def forward(self, x):
        x = self.sa_hh(self.rmsn1(x))+self.sa_ee(self.rmsn1(x))+self.sa_eh(self.rmsn1(x))+self.sa_he(self.rmsn1(x))
        x = self.ffwd(self.rmsn2(x))
        return x

class RelTransformer(nn.Module):
    def __init__(self, num_nodes, num_edges, n_embed):
        super().__init__()
        self.node_embeddings = nn.Embedding(num_nodes, n_embed)
        self.edge_embeddings = nn.Embedding(num_edges, n_embed)
        self.blocks = nn.ModuleList(
            [Block(n_embed, n_heads=4), # add in config file 
            Block(n_embed, n_heads=4),
            Block(n_embed, n_heads=4)]
        )
    
    def forward(self, x):
        


        

# check shapes 
# stuff to move to train.py:
# 

# TODOS:
# Implement Cross Attention Correctly In The Forward Method, Mixed precision, DDP, add stuff to train.py file, figure out how 
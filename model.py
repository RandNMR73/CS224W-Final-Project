import torch
import torch.nn as nn
from torch.nn import functional as F
import math

# General Architecture Stuff
# Include GeLU / GeGLU
# add regularization if needed (we can see based on the training dynamics of model)
# sharing vs not sharing params for queries and keys 
# sharing vs not sharing RMSNorm layer

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

        self.adj_mat = adj_mat # find where to get adjacency matrix from relbench and pass in as param to constructor (function to convert coo tensor to adj matrix) (pass in the coo tensor and do sparse_mm instead of passing adj mat since we don't want to bloat memory either)
        self.shape_proj_1 = nn.Linear(num_edges, num_nodes, bias=False) # find where to get the number of edges and the number of nodes in the graph from relbench (used for making sure that the shapes of the proj matrices match up)
        self.shape_proj_2 = nn.Linear(num_nodes, num_edges, bias=False)

    def forward(self, q, k, v):
        """
        q = node embeddings or edge embeddings
        k = node embeddings or edge embeddings
        v = node embeddings 

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
        super(GeGLU, self).__init__()

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)

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
    def __init__(self,
                 embed_dim, # config
                 num_heads, # config
                 num_nodes, # pass in directly
                 num_edges, # pass in directly
                 adj_mat, # pass in directly 
                 dropout, # config
                 ):
        # n_embed, n_heads, head_size, dropout,
        super().__init__()
        head_size = embed_dim // num_heads 
        self.sa_hh = MultiHeadAttention(embed_dim, num_heads, num_nodes, num_edges, adj_mat, dropout, is_hh_att=True)  # (N, H, E/H)
        self.sa_ee = MultiHeadAttention(embed_dim, num_heads, num_nodes, num_edges, adj_mat, dropout, is_ee_att=True)  # (N, H, E/H)
        self.sa_he = MultiHeadAttention(embed_dim, num_heads, num_nodes, num_edges, adj_mat, dropout, is_he_att=True)  # (N, H, E/H)
        self.sa_eh = MultiHeadAttention(embed_dim, num_heads, num_nodes, num_edges, adj_mat, dropout, is_eh_att=True)  # (N, H, E/H)
        self.ffwd = FeedForward(embed_dim)
        self.rmsn1_n = nn.RMSNorm(embed_dim)
        self.rmsn1_e = nn.RMSNorm(embed_dim)
        self.rmsn2 = nn.RMSNorm(embed_dim)
    
    def forward(self, x_node, x_edge):
        h = self.rmsn1_n(x_node)
        e = self.rmsn1_e(x_edge)
        x = self.sa_hh(h, h, h) + self.sa_ee(e, e, h) + self.sa_eh(e, h, h) + self.sa_he(h, e, h)  # can add gating here 
        x = self.ffwd(self.rmsn2(x))
        return x

class RelTransformer(nn.Module):
    def __init__(self, 
                 node_embeddings, 
                 n_embed, # config
                 num_blocks, # config              
                 num_heads, # config
                 num_nodes, # pass in directly
                 num_edges, # pass in directly
                 adj_mat, # pass in directly 
                 dropout):  # config 
        super().__init__()
        self.node_embeddings = nn.Parameter(node_embeddings)
        self.num_nodes = node_embeddings.shape[0]  # N
        self.n_embed = n_embed 
        self.edge_embeddings = nn.Embedding(num_edges, n_embed)
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList()
        for _ in range(self.num_blocks):
            self.blocks.append(Block(n_embed, num_heads, num_nodes, num_edges, adj_mat, dropout)) 
    
    def forward(self):  # pass in node embeddings from subgraph sampling function (make sure that n_embed)
        out = self.node_embeddings
        for block in self.blocks:
            out = block(out, self.edge_embeddings)  # is there something weird about this? potentially, what if we try learning an (n+m) x d tensor instead
        return out


# check shapes 
# stuff to move to train.py:
# 

# TODOS:
# DDP, add stuff to train.py file, figure out how 
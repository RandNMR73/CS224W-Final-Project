import copy
import re
import warnings
from collections import defaultdict, namedtuple
from collections.abc import Mapping
from itertools import chain
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import torch
from torch import Tensor
from typing_extensions import Self

from torch_geometric.data import EdgeAttr, FeatureStore, GraphStore, TensorAttr, HeteroData
from torch_geometric.data.data import BaseData, Data, size_repr, warn_or_raise
from torch_geometric.data.graph_store import EdgeLayout
from torch_geometric.data.storage import BaseStorage, EdgeStorage, NodeStorage
from torch_geometric.typing import (
    DEFAULT_REL,
    EdgeTensorType,
    EdgeType,
    FeatureTensorType,
    NodeOrEdgeType,
    NodeType,
    QueryType,
    SparseTensor,
    TensorFrame,
    torch_frame,
)
from torch_geometric.utils import (
    bipartite_subgraph,
    contains_isolated_nodes,
    is_sparse,
    is_undirected,
    mask_select,
)


class HeteroDataBrian(HeteroData):
    def set_num_node_features(self, num_node_features):
        self.num_node_features = num_node_features
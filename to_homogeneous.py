import copy
import re
import warnings
from collections import defaultdict, namedtuple
from collections.abc import Mapping
from itertools import chain
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.data import EdgeAttr, FeatureStore, GraphStore, TensorAttr
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
    is_undirected,
    mask_select,
)

NodeOrEdgeStorage = Union[NodeStorage, EdgeStorage]

def to_homogeneous_edge_index(
    data: HeteroData,
) -> Tuple[Optional[Tensor], Dict[NodeType, Any], Dict[EdgeType, Any]]:
    r"""Converts a heterogeneous graph into a homogeneous typed graph."""
    # Record slice information per node type:
    node_slices = get_node_slices(data.num_nodes_dict)

    # Record edge indices and slice information per edge type:
    cumsum = 0
    edge_indices: List[Tensor] = []
    edge_slices: Dict[EdgeType, Tuple[int, int]] = {}
    for edge_type, edge_index in data.collect('edge_index', True).items():
        edge_index = offset_edge_index(node_slices, edge_type, edge_index)
        edge_indices.append(edge_index)
        edge_slices[edge_type] = (cumsum, cumsum + edge_index.size(1))
        cumsum += edge_index.size(1)

    edge_index: Optional[Tensor] = None
    if len(edge_indices) == 1:  # Memory-efficient `torch.cat`:
        edge_index = edge_indices[0]
    elif len(edge_indices) > 1:
        edge_index = torch.cat(edge_indices, dim=-1)

    return edge_index, node_slices, edge_slices


def to_homogeneous(
    batch,
    node_attrs: Optional[List[str]] = None,
    edge_attrs: Optional[List[str]] = None,
    add_node_type: bool = True,
    add_edge_type: bool = True,
    dummy_values: bool = True,
) -> Data:
    """Converts a :class:`~torch_geometric.data.HeteroData` object to a
    homogeneous :class:`~torch_geometric.data.Data` object.
    By default, all features with same feature dimensionality across
    different types will be merged into a single representation, unless
    otherwise specified via the :obj:`node_attrs` and :obj:`edge_attrs`
    arguments.
    Furthermore, attributes named :obj:`node_type` and :obj:`edge_type`
    will be added to the returned :class:`~torch_geometric.data.Data`
    object, denoting node-level and edge-level vectors holding the
    node and edge type as integers, respectively.

    Args:
        node_attrs (List[str], optional): The node features to combine
            across all node types. These node features need to be of the
            same feature dimensionality. If set to :obj:`None`, will
            automatically determine which node features to combine.
            (default: :obj:`None`)
        edge_attrs (List[str], optional): The edge features to combine
            across all edge types. These edge features need to be of the
            same feature dimensionality. If set to :obj:`None`, will
            automatically determine which edge features to combine.
            (default: :obj:`None`)
        add_node_type (bool, optional): If set to :obj:`False`, will not
            add the node-level vector :obj:`node_type` to the returned
            :class:`~torch_geometric.data.Data` object.
            (default: :obj:`True`)
        add_edge_type (bool, optional): If set to :obj:`False`, will not
            add the edge-level vector :obj:`edge_type` to the returned
            :class:`~torch_geometric.data.Data` object.
            (default: :obj:`True`)
        dummy_values (bool, optional): If set to :obj:`True`, will fill
            attributes of remaining types with dummy values.
            Dummy values are :obj:`NaN` for floating point attributes,
            and :obj:`-1` for integers. (default: :obj:`True`)
    """
    def get_sizes(stores: List[BaseStorage]) -> Dict[str, List[Tuple]]:
        sizes_dict = defaultdict(list)
        for store in stores:
            for key, value in store.items():
                if key in [
                        'edge_index', 'edge_label_index', 'adj', 'adj_t'
                ]:
                    continue
                if isinstance(value, Tensor):
                    dim = batch.__cat_dim__(key, value, store)
                    size = value.size()[:dim] + value.size()[dim + 1:]
                    sizes_dict[key].append(tuple(size))
        return sizes_dict

    def fill_dummy_(stores: List[BaseStorage],
                    keys: Optional[List[str]] = None):
        sizes_dict = get_sizes(stores)

        if keys is not None:
            sizes_dict = {
                key: sizes
                for key, sizes in sizes_dict.items() if key in keys
            }

        sizes_dict = {
            key: sizes
            for key, sizes in sizes_dict.items() if len(set(sizes)) == 1
        }

        for store in stores:  # Fill stores with dummy features:
            for key, sizes in sizes_dict.items():
                if key not in store:
                    ref = list(batch.collect(key).values())[0]
                    dim = batch.__cat_dim__(key, ref, store)
                    dummy = float('NaN') if ref.is_floating_point() else -1
                    if isinstance(store, NodeStorage):
                        dim_size = store.num_nodes
                    else:
                        dim_size = store.num_edges
                    shape = sizes[0][:dim] + (dim_size, ) + sizes[0][dim:]
                    store[key] = torch.full(shape, dummy, dtype=ref.dtype,
                                            device=ref.device)

    def _consistent_size(stores: List[BaseStorage]) -> List[str]:
        sizes_dict = get_sizes(stores)
        keys = []
        for key, sizes in sizes_dict.items():
            # The attribute needs to exist in all types:
            if len(sizes) != len(stores):
                continue
            # The attributes needs to have the same number of dimensions:
            lengths = set([len(size) for size in sizes])
            if len(lengths) != 1:
                continue
            # The attributes needs to have the same size in all dimensions:
            if len(sizes[0]) != 1 and len(set(sizes)) != 1:
                continue
            keys.append(key)

        # Check for consistent column names in `TensorFrame`:
        tf_cols = defaultdict(list)
        for store in stores:
            for key, value in store.items():
                if isinstance(value, TensorFrame):
                    cols = tuple(chain(*value.col_names_dict.values()))
                    tf_cols[key].append(cols)

        for key, cols in tf_cols.items():
            # The attribute needs to exist in all types:
            if len(cols) != len(stores):
                continue
            # The attributes needs to have the same column names:
            lengths = set(cols)
            if len(lengths) != 1:
                continue
            keys.append(key)

        return keys

    if dummy_values:
        batch = copy.copy(batch)
        fill_dummy_(batch.node_stores, node_attrs)
        fill_dummy_(batch.edge_stores, edge_attrs)

    edge_index, node_slices, edge_slices = to_homogeneous_edge_index(batch)
    device = edge_index.device if edge_index is not None else None

    data = Data(**batch._global_store.to_dict())
    if edge_index is not None:
        data.edge_index = edge_index
    data._node_type_names = list(node_slices.keys())
    data._edge_type_names = list(edge_slices.keys())

    # Combine node attributes into a single tensor:
    if node_attrs is None:
        node_attrs = _consistent_size(batch.node_stores)
    for key in node_attrs:
        if key in {'ptr'}:
            continue
        # Group values by node type to prevent merging across types
        node_type_values = defaultdict(list)
        for store, node_type in zip(batch.node_stores, data._node_type_names):
            node_type_values[node_type].append(store[key])
        
        for node_type, values in node_type_values.items():
            if isinstance(values[0], TensorFrame):
                value = torch_frame.cat(values, along='row')
            else:
                dim = batch.__cat_dim__(key, values[0], batch.node_stores[0])
                dim = values[0].dim() + dim if dim < 0 else dim
                if values[0].dim() == 2 and dim == 0:
                    _max = max([value.size(-1) for value in values])
                    for i, v in enumerate(values):
                        if v.size(-1) < _max:
                            pad = v.new_zeros(v.size(0), _max - v.size(-1))
                            values[i] = torch.cat([v, pad], dim=-1)
                value = torch.cat(values, dim)
            data[f"{node_type}_{key}"] = value

    if not data.can_infer_num_nodes:
        data.num_nodes = list(node_slices.values())[-1][1]

    # Combine edge attributes into a single tensor:
    if edge_attrs is None:
        edge_attrs = _consistent_size(batch.edge_stores)
    for key in edge_attrs:
        values = [store[key] for store in batch.edge_stores]
        dim = batch.__cat_dim__(key, values[0], batch.edge_stores[0])
        value = torch.cat(values, dim) if len(values) > 1 else values[0]
        data[key] = value

    if 'edge_label_index' in batch:
        edge_label_index_dict = batch.edge_label_index_dict
        for edge_type, edge_label_index in edge_label_index_dict.items():
            edge_label_index = edge_label_index.clone()
            edge_label_index[0] += node_slices[edge_type[0]][0]
            edge_label_index[1] += node_slices[edge_type[-1]][0]
            edge_label_index_dict[edge_type] = edge_label_index
        data.edge_label_index = torch.cat(
            list(edge_label_index_dict.values()), dim=-1)

    if add_node_type:
        sizes = [offset[1] - offset[0] for offset in node_slices.values()]
        sizes = torch.tensor(sizes, dtype=torch.long, device=device)
        node_type = torch.arange(len(sizes), device=device)
        data.node_type = node_type.repeat_interleave(sizes)

    if add_edge_type and edge_index is not None:
        sizes = [offset[1] - offset[0] for offset in edge_slices.values()]
        sizes = torch.tensor(sizes, dtype=torch.long, device=device)
        edge_type = torch.arange(len(sizes), device=device)
        data.edge_type = edge_type.repeat_interleave(sizes)

    return data

"""
PyTorch Geometric utility functions for the Hospital Readmission Prediction model.
This file contains functions for encoding sequences, genres, and loading CSV data.
"""

import torch
import pandas as pd
import numpy as np

class SequenceEncoder:
    """Encode text sequences using a pretrained model."""
    def __init__(self, model_name):
        self.model_name = model_name
        # In a full implementation, we would load the pretrained model here
        # For now, we'll just create a placeholder that returns zeros
    
    def __call__(self, df):
        # Placeholder: in a real implementation, this would encode the text
        # using the pretrained model
        return torch.zeros(len(df), 768)  # BERT hidden size is typically 768

class GenresEncoder:
    """One-hot encode categorical values based on a predefined list."""
    def __init__(self, types_list):
        self.types_list = types_list
    
    def __call__(self, df):
        result = torch.zeros(len(df), len(self.types_list))
        for i, val in enumerate(df):
            if val in self.types_list:
                idx = self.types_list.index(val)
                result[i, idx] = 1
        return result

def node_range(idx_mapping):
    """Encode token ranges in the BERT tokenization."""
    def encode(df):
        result = torch.zeros(len(df), 2, dtype=torch.long)
        for i, val in enumerate(df):
            try:
                if val[0] in idx_mapping and val[1]-1 in idx_mapping:
                    result[i, 0] = idx_mapping[val[0]][0]
                    result[i, 1] = idx_mapping[val[1]-1][1]
            except:
                # Handle edge cases where the mapping doesn't exist
                continue
        return result
    return encode

def load_node_csv(df, index_col, map_start=0, encoders=None):
    """
    Load node features from a dataframe.
    
    Args:
        df: pandas DataFrame containing node information
        index_col: column name to use as node ID
        map_start: starting index for node mapping
        encoders: dict of column name -> encoder function
        
    Returns:
        x: node feature tensor
        mapping: dict mapping node IDs to indices
    """
    mapping = {idx: i + map_start for i, idx in enumerate(df[index_col].unique())}
    x = None
    
    if encoders is None:
        encoders = {}
    
    for column, encoder in encoders.items():
        if column not in df.columns:
            continue
        x_new = encoder(df[column])
        x = x_new if x is None else torch.cat([x, x_new], dim=-1)
    
    return x, mapping

def load_edge_csv(df, src_index_col, src_mapping, dst_index_col, dst_mapping, encoders=None):
    """
    Load edge features from a dataframe.
    
    Args:
        df: pandas DataFrame containing edge information
        src_index_col: column name for source node IDs
        src_mapping: mapping from source node IDs to indices
        dst_index_col: column name for destination node IDs
        dst_mapping: mapping from destination node IDs to indices
        encoders: dict of column name -> encoder function
        
    Returns:
        edge_index: edge index tensor
        edge_attr: edge attribute tensor
    """
    if len(df) == 0:
        return torch.empty(2, 0, dtype=torch.long), torch.empty(0, 0)
    
    # Map source and destination node IDs to indices
    src = []
    dst = []
    for s, d in zip(df[src_index_col], df[dst_index_col]):
        if str(s) in src_mapping and str(d) in dst_mapping:
            src.append(src_mapping[str(s)])
            dst.append(dst_mapping[str(d)])
    
    if not src or not dst:
        return torch.empty(2, 0, dtype=torch.long), torch.empty(0, 0)
    
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    edge_attr = None
    if encoders is not None:
        for column, encoder in encoders.items():
            if column not in df.columns:
                continue
            edge_attr_new = encoder(df[column])
            edge_attr = edge_attr_new if edge_attr is None else torch.cat([edge_attr, edge_attr_new], dim=-1)
    
    if edge_attr is None:
        # Return empty tensor with the correct shape
        edge_attr = torch.empty(len(src), 0)
    
    return edge_index, edge_attr
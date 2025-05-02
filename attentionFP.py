import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

class AttentiveFP(nn.Module):
    """
    Implementation of AttentiveFP model for graph-level predictions
    Based on "Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph Attention Mechanism"
    Modified for the hospital readmission prediction task.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim=None, 
                 num_layers=3, num_timesteps=3, dropout=0.0):
        super(AttentiveFP, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.dropout = dropout
        
        # Initial node embedding
        self.node_embedding = nn.Linear(in_channels, hidden_channels)
        
        # Gated Graph Neural Network layers
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gnn_layers.append(
                GATELayer(hidden_channels, hidden_channels, edge_dim, dropout)
            )
        
        # Graph readout with attention
        self.graph_attention = GraphAttention(hidden_channels, dropout)
        
        # Final MLP for prediction
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        Forward pass of AttentiveFP
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            batch: Batch assignment [num_nodes]
            
        Returns:
            output: Graph-level prediction [batch_size, out_channels]
        """
        # Initial node embedding
        h = self.node_embedding(x)
        
        # Apply GNN layers
        for layer in self.gnn_layers:
            h = layer(h, edge_index, edge_attr)
        
        # Apply graph-level attention for multiple timesteps
        for _ in range(self.num_timesteps):
            h_graph = self.graph_attention(h, batch)
            # Use global context to update node features
            h = F.relu(h + h_graph[batch])
        
        # Final graph-level readout with attention
        h_graph = self.graph_attention(h, batch)
        
        # MLP for prediction
        output = self.mlp(h_graph)
        
        return output


class GATELayer(MessagePassing):
    """
    Graph Attention with Edge Features
    """
    def __init__(self, in_channels, out_channels, edge_dim=None, dropout=0.0):
        super(GATELayer, self).__init__(aggr='add', node_dim=0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.dropout = dropout
        
        # Learnable weights
        self.W = nn.Linear(in_channels, out_channels, bias=False)
        self.a = nn.Linear(2 * out_channels + (edge_dim or 0), 1, bias=False)
        
        # GRU for updating node states
        self.gru = nn.GRUCell(out_channels, out_channels)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize learnable parameters"""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.W.weight, gain=gain)
        nn.init.xavier_normal_(self.a.weight, gain=gain)
        self.gru.reset_parameters()
    
    def forward(self, x, edge_index, edge_attr=None):
        """Forward pass through the layer"""
        x_transformed = self.W(x)
        
        # Propagate messages
        m = self.propagate(edge_index, x=x_transformed, edge_attr=edge_attr)
        
        # GRU update
        x_new = self.gru(m, x_transformed)
        
        return x_new
    
    def message(self, x_i, x_j, edge_attr=None, index=None, ptr=None, size_i=None):
        """Message computation for GATE"""
        # Compute attention coefficients
        if edge_attr is not None:
            attention_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        else:
            attention_input = torch.cat([x_i, x_j], dim=-1)
        
        alpha = self.a(attention_input)
        alpha = F.leaky_relu(alpha, 0.2)
        
        # Normalize attention coefficients
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Return weighted message
        return x_j * alpha


class GraphAttention(nn.Module):
    """
    Graph-level attention for readout
    """
    def __init__(self, hidden_channels, dropout=0.0):
        super(GraphAttention, self).__init__()
        
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        
        # Attention mechanism for graph-level readout
        self.attention = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, 1, bias=False)
        )
    
    def forward(self, x, batch):
        """
        Compute graph-level representation with attention
        
        Args:
            x: Node features [num_nodes, hidden_channels]
            batch: Batch assignment [num_nodes]
            
        Returns:
            h_graph: Graph-level representations [batch_size, hidden_channels]
        """
        # Compute attention weights
        attention_weights = self.attention(x)
        attention_weights = F.dropout(attention_weights, p=self.dropout, training=self.training)
        
        # Apply softmax within each graph
        attention_weights = softmax(attention_weights, batch, None)
        
        # Weighted sum of node features
        h_graph = global_add_pool(x * attention_weights, batch)
        
        return h_graph
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    """
    Self-attention module for sequence modeling in the CHSLM architecture.
    This implements the attention mechanism used in the paper.
    """
    def __init__(self, hidden_size, num_attention_heads=1, dropout_prob=0.1):
        super(SelfAttention, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Create the query, key, and value projections
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        # Dropout for attention probabilities
        self.attn_dropout = nn.Dropout(dropout_prob)
        self.output_dropout = nn.Dropout(dropout_prob)
        
        # Output projection
        self.output = nn.Linear(hidden_size, hidden_size)
        
    def transpose_for_scores(self, x):
        """Reshape to separate attention heads"""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        # Transpose to [batch_size, num_heads, seq_length, head_size]
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, attention_mask=None):
        """
        Apply self-attention over the hidden states
        
        Args:
            hidden_states: input tensor of shape [batch_size, seq_length, hidden_size]
            attention_mask: optional mask of shape [batch_size, seq_length, seq_length]
        
        Returns:
            context_layer: attended output of same shape as hidden_states
            attention_probs: attention distributions
        """
        # Project inputs to queries, keys, and values
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        
        # Reshape for multi-head attention
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # Take the dot product between query and key to get attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        # Apply softmax to get attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)
        
        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        # Reshape back to original dimensions
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # Apply output projection and dropout
        output_layer = self.output(context_layer)
        output_layer = self.output_dropout(output_layer)
        
        return output_layer, attention_probs
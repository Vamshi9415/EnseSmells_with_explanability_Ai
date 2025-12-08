import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    '''
    Multi-Head Self-Attention Mechanism
    Replaces simple linear attention with standard Transformer-style attention.
    '''
    
    def __init__(self, embed_size, num_heads=4):
        super().__init__()
        self.embed_size = embed_size
        
        # Multi-Head Attention
        # batch_first=True ensures input/output are (batch, seq, feature)
        self.mha = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads, batch_first=True)
        
        # Layer Normalization for stability
        self.norm = nn.LayerNorm(embed_size)
        
    def forward(self, x):
        '''
        Args:
            x: (batch_size, seq_len, embed_size) - Output from BiLSTM
            
        Returns:
            pooled_output: (batch, embed_size) - Aggregated vector for the sequence
            attn_weights: (batch, seq_len) - Average attention weights across heads
        '''
        # Self-attention: Query, Key, and Value are all 'x'
        # attn_output shape: (batch, seq_len, embed_size)
        # attn_weights shape: (batch, seq_len, seq_len) -> We only need weights relative to the sequence
        attn_output, attn_weights_matrix = self.mha(x, x, x)
        
        # Residual Connection + Normalization (Standard Transformer block practice)
        x = self.norm(x + attn_output)
        
        # POOLING:
        # Instead of just taking the last state, we take the mean of the attention-enriched sequence
        # This creates a single vector representation for the whole code snippet
        pooled_output = x.mean(dim=1) 
        
        # For visualization/explainability:
        # We average the attention weights across all heads and all target tokens
        # to get a single "importance score" per token in the sequence.
        # attn_weights_matrix is (batch, num_heads, target_len, src_len) or (batch, target_len, src_len)
        # We simplify to (batch, seq_len) for visualization
        avg_attn_weights = attn_weights_matrix.mean(dim=1) 
        
        return pooled_output, avg_attn_weights
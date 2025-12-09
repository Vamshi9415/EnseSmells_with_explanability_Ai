import torch 
import torch.nn as nn
import torch.nn.functional as F 
import os
import sys

# Add path for imports if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
complete_network_dir = os.path.dirname(current_dir)
if complete_network_dir not in sys.path:
    sys.path.insert(0, complete_network_dir)

from semantic_module.AttentionLayer import AttentionLayer

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class SemanticModule(nn.Module):
    '''
    CNN-Transformer Architecture
    Replaces LSTM with Transformer Encoder for better long-range dependency capture.
    '''
    def __init__(self, embedding_dim, output_features=100):
        super().__init__()
        
        # --- CNN Block (Local Feature Extraction) ---
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.se1 = SEBlock(channel=32)
        self.pool1 = nn.MaxPool1d(kernel_size=3)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.se2 = SEBlock(channel=64)
        self.pool2 = nn.MaxPool1d(kernel_size=3)
        
        # --- Transformer Block (Global Context) ---
        # d_model must match conv2 out_channels (64)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, 
            nhead=4, 
            dim_feedforward=256, 
            dropout=0.2, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # --- Attention Pooling ---
        self.attention = AttentionLayer(embed_size=64, num_heads=4)
        
        # Final Projection
        self.fc = nn.Linear(64, output_features)
        
        self.attention_weights = None
        
    def forward(self, x):
        # x: (batch, embedding_dim)
        x = x.unsqueeze(1) 
        
        # CNN Layers
        x = self.pool1(self.se1(self.bn1(F.relu(self.conv1(x)))))
        x = self.pool2(self.se2(self.bn2(F.relu(self.conv2(x)))))
        
        # Prepare for Transformer: (batch, channels, seq) -> (batch, seq, channels)
        x = x.transpose(1, 2)
        
        # Transformer Encoder
        # No positional encoding needed as CNN preserves relative order locally
        # and we want translation invariance for code patterns.
        trans_out = self.transformer(x)
        
        # Attention Pooling
        attended_output, attention_weights = self.attention(trans_out)
        self.attention_weights = attention_weights
        
        # Final Features
        features = self.fc(attended_output)
        
        # Return both the pooled features AND the full sequence (for Cross-Attention Fusion)
        return features, attention_weights, trans_out
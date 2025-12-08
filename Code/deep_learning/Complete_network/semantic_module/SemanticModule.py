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
    """
    Squeeze-and-Excitation Block
    Adaptive feature recalibration: explicitly models interdependencies between channels.
    """
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
    CNN-BiLSTM Architecture with SE-Blocks and Multi-Head Attention
    '''
    def __init__(self, embedding_dim, output_features=100):
        super().__init__()
        
        # Block 1
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.se1 = SEBlock(channel=16) # NEW: SE Block
        self.pool1 = nn.MaxPool1d(kernel_size=3)
        
        # Block 2
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.se2 = SEBlock(channel=32) # NEW: SE Block
        self.pool2 = nn.MaxPool1d(kernel_size=3)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=32, 
            hidden_size=32, 
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # NEW: Multi-Head Attention Layer
        # Input size is 64 (32 hidden * 2 directions)
        self.attention = AttentionLayer(embed_size=64, num_heads=4)
        
        # Final projection
        self.fc = nn.Linear(64, output_features)
        
        self.attention_weights = None
        
    def forward(self, x):
        # x shape: (batch, embedding_dim) -> unsqueeze for conv1d: (batch, 1, embedding_dim)
        x = x.unsqueeze(1) 
        
        # Conv Block 1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.se1(x) # Apply SE Attention
        x = self.pool1(x)
        
        # Conv Block 2
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.se2(x) # Apply SE Attention
        x = self.pool2(x)
        
        # Prepare for LSTM: (batch, channels, seq_len) -> (batch, seq_len, channels)
        x = x.transpose(1, 2)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Multi-Head Attention
        attended_output, attention_weights = self.attention(lstm_out)
        self.attention_weights = attention_weights
        
        # Final Projection
        features = self.fc(attended_output)
        
        return features, attention_weights
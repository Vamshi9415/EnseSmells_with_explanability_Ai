import torch 
import torch.nn as nn
import torch.nn.functional as F 

class SemanticModule(nn.Module):
    '''
    Architecture goes like this
    - conv block 1 : conv1d 16 filters -> relu -> batch norm -> max pool
    - conv block 2 : conv1d 32 filters -> relu -> batch norm -> max pool
    - bi-lstm : 32 hidden units
    - output : 100 dimensional features
    '''
    def __init__(self, embedding_dim, output_features = 100):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, #number of input features
                               out_channels, # number of output filters
                               cnn_kernel_size)
        
        self.batch_norm - nn.BatchNorm1d(batch_norm_input_channels)
        
        self.pooling_layer = nn.MaxPool1d(pooling_layer_kernelsize)
        
        self.conv2 = nn.Conv2d(out_channels = 32)
        
        self.attention = nn.MultiheadAttention()
        
        self.lstm = nn.LSTM(lstm_input, lstm_hidden_size, lstm_num_layers, bidirectional = True)
        
        self.attention_weights = None
        
    def forward(self,x):
        
        features = self.conv1(x)
        
        features = self.batch_norm(features)
        
        features = self.pooling_layer(features)
        
        features = self.attention(features)
        
        features. self.lstm(features)
        

        
        
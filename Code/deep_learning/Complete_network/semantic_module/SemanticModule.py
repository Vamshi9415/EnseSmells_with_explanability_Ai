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
        
        self.conv1 = nn.Conv1d(in_channels = 1, # single channel input
                               out_channels = 16, # number of  filters
                               kernel_size = 5, #kernel size need to experiment with 2,4,5,7
                               padding = 2 
                               # padding  = 0 valid convolution no padding output shrinks
                               # padding = (kernel_size -1)/2 -> to keep output sequence length as input 
                               # here stride = 1
                               )
        
        self.batch_norm1 = nn.BatchNorm1d(16) # normalizes the output layer , basically keeps activations(features) within stable range during training
        
        self.pool1 = nn.MaxPool1d(kernel_size = 3) # generally if kernel size = 3 then stride is also 3
        
        self.conv2 = nn.Conv2d(
            in_channels = 16, #from first block number of output_features
            out_channels = 32,
            kernel_size = 5,
            padding = 2
            )
        
        self.batch_norm2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size = 3)
        
        # birectional lstm
        
        self.lstm = nn.LSTM(
            input_size = 32, # from conv2 ouput channels
            hidden_size = 32, # 32 hidden_units
            num_layers =1,
            batch_first = True,
            bidirectional = True # bilstm outputs 64 features
        )
        
        #ading attention layer for explanability
        self.attention = nn.AttentionLayer(hidden_size = 64)
        
        self.fc = nn.Linear(64, output_features)
        
        self.attention_weights = None
        
    def forward(self,x):
        
        
        batch_size = x.size(0)
        
        x = x.unsqueeze(1) #reshape for conv1d
        
        # 1st convoultional layer
        x = self.conv1(x)
        x = F.relu(x)
        x = self.batch_norm1(x)
        x = self.pool1(x)
        
        # 2nd convoultional layer 
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batch_norm2(x)
        x = self.poool2(x)
        
        # lstm : (batch,seq_len, features)
        
        
        
        
        
        
        

        
        
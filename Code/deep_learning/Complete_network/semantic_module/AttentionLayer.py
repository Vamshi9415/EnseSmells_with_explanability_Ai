import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    '''
    self attention mechanism for bilstm outputs
    
    '''
    
    
    
    def __init__(self, embed_size):
        super().__init__()
        
        
        self.embed_size = embed_size
        #define linear transformations for Q,K,V
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size,embed_size)
        self.value = nn.Linear(embed_size,embed_size)
        
    def forward(self, x):
        # here x is the ouput from the lstm layer 
        # which is in the form of (batch_size, time_Steps, units) when return_Sewunces are true
        # batch_size - number of sequences processed at once
        # time_Steps - number of embeddings in each sequence like how many embedding vectors 
        # units - number of lstm units where each unit represents one dimension of the hidden state at each time step
        # (batch_size, time_steps, embedding_dim) -> input shape
        #                ||
        #                ||
        #          -- LSTM layer --
        #                ||
        #                ||
        #   (batch_size, tim_steps, units)  -> output shape
        # we need to generate the Q,K,V matrices
        
        
        
        
        
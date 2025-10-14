import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    '''
    self attention mechanism for bilstm outputs
    
    '''
    
    
    
    def __init__(self, embed_size):
        super().__init__()
        
        
        self.embed_size = 
        #define linear transformations for Q,K,V
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size,embed_size)
        self.value = nn.Linear(embed_size,embed_size)
        
        self.scale = torch.sqrt(torch.FloatTensor([embed_size]))
        
    def forward(self, x):
        '''
        here x is the ouput from the lstm layer 
        which is in the form of (batch_size, time_Steps, units) when return_Sewunces are true
        batch_size - number of sequences processed at once
        time_Steps - number of embeddings in each sequence like how many embedding vectors 
        units - number of lstm units where each unit represents one dimension of the hidden state at each time step
        (batch_size, time_steps, embedding_dim) -> input shape
                       ||
                       ||
                 -- LSTM layer --
                       ||
                       ||
          (batch_size, tim_steps, units)  -> output shape
          
        we need to generate the Q,K,V matrices
         Returns:
            attended_output: (batch, hidden_size) weighted representation
            attention_weights: (batch, seq_len) importance scores
        '''
        
        
        
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # calculating attetnion_Scores Q x K^T
        
        attention_scores = torch.matmul(Q, K.transpose(-2,-1))/ (self.scale.to(x.device)) 

        # (batch_size, seq_len, embed_dim) K.transpose(-2, -1) → shape (batch_size, embed_dim, seq_len)
        #Without scaling, dot products can be large, making softmax very peaky and gradients unstable.
        #.to(x.device) ensures self.scale is on the same device as the input (CPU or GPU), so no device mismatch occurs.

        #applying softmax function
        
        attention_weights = F.softmax(attention_scores.mean(dim = 1), dim = -1)
        
        # Apply attention to values
        attention_weights_expanded = attention_weights.unsqueeze(1)  # (batch, 1, seq_len)
        attended_output = torch.matmul(attention_weights_expanded, V).squeeze(1)  # (batch, hidden_size)
        
        return attended_output, attention_weights
        
        
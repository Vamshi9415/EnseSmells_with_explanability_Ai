import torch 
import torch.nn as nn
import torch.nn.functional as F 
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split

class StructuralModule(nn.Module):
    '''
    Structural module with feature attention
    
    converting software metrics to 24 feature vectors
    '''
    
    def __init__(self, input_metrics , output_features , attention_dimension,hidden_layer_dim):
        
        super().__init__()
        
        #attetnion mechanism 
        
        self.attention_network = nn.Sequential(
            nn.Linear(input_metrics, attention_dimension),
            nn.Tanh(),
            nn.Linear(attention_dimension, input_metrics),
            nn.Softmax(dim = 1)
        )
        
        self.fc1 = nn.Linear(input_metrics,hidden_layer_dim)
        self.out = nn.Linear(hidden_layer_dim, output_features)
        
        self.attention_weights = None
        
    def forward(self, x):
        attention_weights = self.attention_network(x)
        self.attention_weights = attention_weights
        
        weighted_attention  = x*attention_weights
        
        features = F.relu(self.fc1(weighted_attention))
        features = self.out(features)
        
        return features, attention_weights
        
    
    def get_metric_importance(self):
        if self.attention_weights is not None:
            return self.attention_weights.mean(dim = 0).detach()
        return None

if __name__ == "__main__":
    # Load dataset
    LONG_METHOD_dataset = pd.read_csv("../../../dataset-source/embedding-dataset/software-metrics/LongMethod_code_metrics_values.csv")
    X_data = LONG_METHOD_dataset.drop(['sample_id', 'label'], axis=1).values
    y_data = LONG_METHOD_dataset['label'].values
    
    # Automatically detect input features
    input_features = X_data.shape[1]
    print(f"Detected {input_features} input features from dataset")
    
    # Create model
    model = StructuralModule(
        input_metrics=input_features,
        output_features=24,
        attention_dimension=32,
        hidden_layer_dim=64  # ✓ FIXED: Added missing parameter
    )
    
    print(f"\nModel: {input_features} metrics → 24 features")
    print(f"Architecture: {input_features} → Attention → 64 → 24")

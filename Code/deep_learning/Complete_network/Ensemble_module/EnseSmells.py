import torch
import torch.nn as nn
import sys
import os

# Add Complete_network to path if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
complete_network_dir = os.path.dirname(current_dir)
if complete_network_dir not in sys.path:
    sys.path.insert(0, complete_network_dir)

# Use absolute imports from Complete_network
from semantic_module.SemanticModule import SemanticModule
from structural_module.StructuralModule import StructuralModule

class EnseSmells(nn.Module):
    '''
    Combining both the modules
    '''
    
    def __init__(self, embedding_dim, input_metrics):
        super().__init__()
        
        self.semantic = SemanticModule(
            embedding_dim=embedding_dim,
            output_features=100
        )
        
        self.structural = StructuralModule(
            input_metrics=input_metrics,
            output_features=24,
            attention_dimension=32,
            hidden_layer_dim=64
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(124, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
    
    def forward(self, embeddings, metrics):
        semantic_features, semantic_attention_weights = self.semantic.forward(embeddings)
        structural_features, structural_attention_weights = self.structural.forward(metrics)
        
        combined_features = torch.cat([semantic_features, structural_features], dim=1)
        combined_prediction = self.classifier(combined_features)
        
        return combined_prediction, semantic_attention_weights, structural_attention_weights
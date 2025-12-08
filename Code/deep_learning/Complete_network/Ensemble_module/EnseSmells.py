import torch
import torch.nn as nn
import sys
import os
import torch.nn.functional as F
current_dir = os.path.dirname(os.path.abspath(__file__))
complete_network_dir = os.path.dirname(current_dir)
if complete_network_dir not in sys.path:
    sys.path.insert(0, complete_network_dir)

from semantic_module.SemanticModule import SemanticModule
from structural_module.StructuralModule import StructuralModule

class GatedFusion(nn.Module):
    """
    Intelligent fusion of two modalities (Code + Metrics).
    Learns to weigh each branch dynamically per sample.
    """
    def __init__(self, dim_a, dim_b, hidden_dim):
        super().__init__()
        # Project both inputs to the same hidden dimension
        self.project_a = nn.Linear(dim_a, hidden_dim)
        self.project_b = nn.Linear(dim_b, hidden_dim)
        
        # The Gate Network: Decides 'z' (weight 0 to 1) based on concatenated inputs
        self.gate_net = nn.Sequential(
            nn.Linear(dim_a + dim_b, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, a, b):
        # a: Semantic features (100)
        # b: Structural features (24)
        
        h_a = F.relu(self.project_a(a)) # shape: (batch, hidden_dim)
        h_b = F.relu(self.project_b(b)) # shape: (batch, hidden_dim)
        
        # Calculate Gate z
        # Concatenate raw inputs to decide weights
        cat_inputs = torch.cat([a, b], dim=1)
        z = self.gate_net(cat_inputs) # shape: (batch, 1)
        
        # Weighted combination: z * A + (1-z) * B
        fused = z * h_a + (1 - z) * h_b
        
        return fused

class EnseSmells(nn.Module):
    '''
    Ensemble Model with Gated Fusion
    '''
    def __init__(self, embedding_dim, input_metrics):
        super().__init__()
        
        # 1. Semantic Branch (Code)
        self.semantic = SemanticModule(
            embedding_dim=embedding_dim,
            output_features=100
        )
        
        # 2. Structural Branch (Metrics)
        self.structural = StructuralModule(
            input_metrics=input_metrics,
            output_features=24,
            attention_dimension=32,
            hidden_layer_dim=64
        )
        
        # 3. Gated Fusion
        # Fuses 100 (Semantic) + 24 (Structural) -> 64 fused features
        self.fusion = GatedFusion(dim_a=100, dim_b=24, hidden_dim=64)
        
        # 4. Final Classifier
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
            # No Sigmoid here if using BCEWithLogitsLoss
        )
    
    def forward(self, embeddings, metrics):
        # Get features from both branches
        sem_feat, sem_attn = self.semantic(embeddings)
        str_feat, str_attn = self.structural(metrics)
        
        # Intelligent Fusion
        fused_features = self.fusion(sem_feat, str_feat)
        
        # Classification
        prediction = self.classifier(fused_features)
        
        return prediction, sem_attn, str_attn
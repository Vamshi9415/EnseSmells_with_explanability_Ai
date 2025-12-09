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

class CrossModalAttentionFusion(nn.Module):
    """
    Advanced Fusion: Uses Metrics to 'attend' to specific parts of the Code.
    Query = Metrics Vector
    Key/Value = Code Sequence
    """
    def __init__(self, semantic_dim, structural_dim, fused_dim):
        super().__init__()
        
        # Project metrics to match semantic dimension for attention
        self.metric_proj = nn.Linear(structural_dim, semantic_dim)
        
        # Cross Attention
        self.cross_attn = nn.MultiheadAttention(embed_dim=semantic_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(semantic_dim)
        
        # Final fusion layer
        self.fusion_fc = nn.Linear(semantic_dim + structural_dim, fused_dim)
        
    def forward(self, code_seq, metrics_vec):
        # code_seq: (batch, seq_len, semantic_dim) -> From Transformer
        # metrics_vec: (batch, structural_dim)
        
        # 1. Prepare Query (Metrics)
        # Reshape metrics to (batch, 1, semantic_dim) to act as a sequence of length 1
        query = self.metric_proj(metrics_vec).unsqueeze(1)
        
        # 2. Prepare Key/Value (Code)
        key = code_seq
        value = code_seq
        
        # 3. Perform Attention
        # "Which parts of the code are relevant given these metrics?"
        attn_output, _ = self.cross_attn(query, key, value)
        
        # attn_output is (batch, 1, semantic_dim). Squeeze to (batch, semantic_dim)
        context_code = attn_output.squeeze(1)
        context_code = self.norm(context_code)
        
        # 4. Concatenate Original Metrics with Context-Aware Code
        combined = torch.cat([context_code, metrics_vec], dim=1)
        
        # 5. Fuse
        fused = F.relu(self.fusion_fc(combined))
        return fused

class EnseSmells(nn.Module):
    '''
    Ensemble Model with Transformer Backbone and Cross-Modal Attention
    '''
    def __init__(self, embedding_dim, input_metrics):
        super().__init__()
        
        # 1. Semantic Branch (Outputting 64 dim sequence)
        self.semantic = SemanticModule(
            embedding_dim=embedding_dim,
            output_features=100 # Kept for compatibility, but we use the sequence mostly
        )
        
        # 2. Structural Branch
        self.structural = StructuralModule(
            input_metrics=input_metrics,
            output_features=24,
            attention_dimension=32,
            hidden_layer_dim=64
        )
        
        # 3. Cross-Modal Fusion
        # Semantic Dim = 64 (from Transformer in SemanticModule)
        # Structural Dim = 24
        self.fusion = CrossModalAttentionFusion(semantic_dim=64, structural_dim=24, fused_dim=64)
        
        # 4. Final Classifier
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )
    
    def forward(self, embeddings, metrics):
        # Semantic: Get the full sequence (trans_out) in addition to pooled features
        # trans_out shape: (batch, seq_len, 64)
        sem_feat, sem_attn, trans_out = self.semantic(embeddings)
        
        # Structural: Get metric features
        str_feat, str_attn = self.structural(metrics)
        
        # Cross-Modal Fusion: Metrics attend to Code Sequence
        fused_features = self.fusion(trans_out, str_feat)
        
        # Classification
        prediction = self.classifier(fused_features)
        
        return prediction, sem_attn, str_attn
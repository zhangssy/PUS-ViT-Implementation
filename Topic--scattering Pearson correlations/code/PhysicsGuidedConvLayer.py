"""
Physics-guided convolutional layer (PGCL)
Fuses CLS features with a topic vector for land-cover classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsGuidedConvLayer(nn.Module):
    """
    Physics-guided layer
    Combines Vision Transformer CLS features with the topic representation
    Uses the topic vector as a gate on the fused features
    """
    
    def __init__(self, embed_dim=64, num_topics=175, num_classes=9, hidden_dim=128):
        """
        Initialize PGCL
        
        Args:
            embed_dim: CLS feature dimension
            num_topics: number of topics
            num_classes: number of classes
            hidden_dim: hidden width
        """
        super(PhysicsGuidedConvLayer, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_topics = num_topics
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # Fuse the CLS feature with the topic vector
        # Input: embed_dim (CLS) + num_topics (topic)
        fusion_input_dim = embed_dim + num_topics
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Topic-conditioned gate
        # The topic vector supplies the gate input
        self.physics_guidance = nn.Sequential(
            nn.Linear(num_topics, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid()  # gate
        )
        
        # Refine the gated features
        self.feature_enhancement = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
    
    def forward(self, original_features, topic_representation):
        """
        Forward pass
        
        Args:
            original_features: CLS features (B, embed_dim)
            topic_representation: topic representation (B, num_topics)
            
        Returns:
            enhanced_features: enhanced features (B, hidden_dim)
        """
        # 1. Concatenate CLS features and the topic vector
        fused_features = torch.cat([original_features, topic_representation], dim=1)  # (B, embed_dim + num_topics)
        
        # 2. Fusion MLP
        fused_output = self.fusion_layer(fused_features)  # (B, hidden_dim)
        
        # 3. Topic-conditioned gate
        physics_guidance_signal = self.physics_guidance(topic_representation)  # (B, hidden_dim)
        
        # 4. Apply the gate
        guided_features = fused_output * physics_guidance_signal  # (B, hidden_dim)
        
        # 5. Feature refinement
        enhanced_features = self.feature_enhancement(guided_features)  # (B, hidden_dim)
        
        # 6. Return the enhanced features
        return enhanced_features
    
    def get_enhanced_features(self, original_features, topic_representation):
        """
        Return enhanced features for analysis
        
        Args:
            original_features: original features
            topic_representation: topic representation
            
        Returns:
            enhanced_features: enhanced features
        """
        self.eval()
        with torch.no_grad():
            enhanced_features = self.forward(original_features, topic_representation)
        return enhanced_features


class DualBranchFusionPGCL(nn.Module):
    """
    Dual-branch fusion PGCL
    During RSD training, two branches process the patch and are fused
    During T-matrix training, the layer takes CLS features and returns enhanced features
    """
    
    def __init__(self, embed_dim=64, num_topics=175, num_classes=9, hidden_dim=128, 
                 dropout=0.1, in_channels=12, img_size=7):
        """
        Initialize the dual-branch fusion PGCL
        
        Args:
            embed_dim: embedding dimension (CLS dimension)
            num_topics: number of topics
            num_classes: number of classes
            hidden_dim: hidden width
            dropout: dropout rate
            in_channels: input channels (branch-1 projection during RSD training)
            img_size: image size (branch-1 projection during RSD training)
        """
        super(DualBranchFusionPGCL, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_topics = num_topics
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.in_channels = in_channels
        self.img_size = img_size
        
        # Lazy import to avoid a circular dependency
        from SupervisedTopicModelForPGCL import SupervisedTopicModelForPGCL
        
        # ========== Branch 1: direct topic path ==========
        # Branch 1 projects the raw RSD patch to embed_dim during RSD training
        self.branch1_projection = nn.Sequential(
            nn.Flatten(),  # (B, C, H, W) -> (B, C*H*W)
            nn.Linear(in_channels * img_size * img_size, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Branch-1 topic model on the projected RSD patch
        self.branch1_topic_model = SupervisedTopicModelForPGCL(
            embed_dim=embed_dim,
            num_topics=num_topics,
            num_classes=num_classes
        )
        
        # Branch-1 PGCL
        self.branch1_pgcl = PhysicsGuidedConvLayer(
            embed_dim=embed_dim,
            num_topics=num_topics,
            num_classes=num_classes,
            hidden_dim=hidden_dim
        )
        
        # ========== Branch 2: CLS feature path ==========
        # Branch-2 topic model on the ViT CLS feature
        self.branch2_topic_model = SupervisedTopicModelForPGCL(
            embed_dim=embed_dim,
            num_topics=num_topics,
            num_classes=num_classes
        )
        
        # Branch-2 PGCL
        self.branch2_pgcl = PhysicsGuidedConvLayer(
            embed_dim=embed_dim,
            num_topics=num_topics,
            num_classes=num_classes,
            hidden_dim=hidden_dim
        )
        
        # ========== Fuse the two enhanced branch features ==========
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
    
    def forward(self, cls_features, raw_data=None):
        """
        Forward pass
        
        Args:
            cls_features: CLS features (B, embed_dim)
                         During RSD training this is the ViT CLS feature
                         During T-matrix training this is the ViT CLS feature
            raw_data: optional raw RSD patch (B, C, H, W)
                      If given, branch 1 uses the raw patch (RSD training)
                      If omitted, branch 1 also uses the CLS feature (T-matrix training)
        
        Returns:
            enhanced_features: fused enhanced features (B, hidden_dim)
            branch1_topic: branch-1 topic vector (B, num_topics)
            branch2_topic: branch-2 topic vector (B, num_topics)
        """
        # ========== Branch 1: direct topic path ==========
        if raw_data is not None:
            # RSD training: use the raw RSD patch
            branch1_features = self.branch1_projection(raw_data)  # (B, embed_dim)
        else:
            # T-matrix training: use the CLS feature
            branch1_features = cls_features  # (B, embed_dim)
        
        branch1_topic, _ = self.branch1_topic_model(branch1_features)  # (B, num_topics)
        branch1_enhanced = self.branch1_pgcl(branch1_features, branch1_topic)  # (B, hidden_dim)
        
        # ========== Branch 2: CLS feature path ==========
        # Pass the CLS feature through the branch-2 topic model
        branch2_topic, _ = self.branch2_topic_model(cls_features)  # (B, num_topics)
        branch2_enhanced = self.branch2_pgcl(cls_features, branch2_topic)  # (B, hidden_dim)
        
        # ========== Fuse the enhanced features of both branches ==========
        fused_features = torch.cat([branch1_enhanced, branch2_enhanced], dim=1)  # (B, hidden_dim * 2)
        enhanced_features = self.fusion_layer(fused_features)  # (B, hidden_dim)
        
        return enhanced_features, branch1_topic, branch2_topic


# if __name__ == "__main__":
#     # Smoke test; runs only when this file is executed directly
#     batch_size = 32
#     embed_dim = 64
#     num_topics = 175
#     num_classes = 9

#     model = PhysicsGuidedConvLayer(
#         embed_dim=embed_dim,
#         num_topics=num_topics,
#         num_classes=num_classes
#     )

#     # Synthetic input
#     original_features = torch.randn(batch_size, embed_dim)
#     topic_representation = torch.randn(batch_size, num_topics)

#     # Forward pass
#     enhanced_features = model(original_features, topic_representation)

#     print(f"Original feature shape: {original_features.shape}")
#     print(f"Topic shape: {topic_representation.shape}")
#     print(f"Enhanced feature shape: {enhanced_features.shape}")
#     print(f"Enhanced feature shape: {enhanced_features.shape}")

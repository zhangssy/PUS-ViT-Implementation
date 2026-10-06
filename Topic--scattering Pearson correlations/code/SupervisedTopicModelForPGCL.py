"""
Supervised topic model used by the physics-guided layer
Builds a supervised topic model from Vision Transformer features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SupervisedTopicModelForPGCL(nn.Module):
    """
    Supervised topic model used by the physics-guided layer
    Decomposes CLS features into a topic representation for PGCL
    """
    
    def __init__(self, embed_dim=64, num_topics=175, num_classes=9):
        """
        Initialize the supervised topic model
        
        Args:
            embed_dim: input feature dimension (CLS dimension)
            num_topics: number of topics
            num_classes: number of classes (used by the supervised loss)
        """
        super(SupervisedTopicModelForPGCL, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_topics = num_topics
        self.num_classes = num_classes
        
        # Topic basis (num_topics, embed_dim), constrained to be nonnegative
        self.topic_basis = nn.Parameter(torch.randn(num_topics, embed_dim))
        nn.init.xavier_uniform_(self.topic_basis)
        # Keep the initial values positive
        with torch.no_grad():
            self.topic_basis.data = torch.abs(self.topic_basis.data) + 0.1
        
        # Map CLS features into the topic space
        self.feature_to_topic = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_topics),
            nn.ReLU()  # Keep the output nonnegative
        )
        
        # Class prototypes in topic space, used by the supervised loss
        self.class_prototypes = nn.Parameter(torch.randn(num_classes, num_topics))
        nn.init.xavier_uniform_(self.class_prototypes)
        with torch.no_grad():
            self.class_prototypes.data = torch.abs(self.class_prototypes.data) + 0.1
    
    def forward(self, cls_features):
        """
        Forward pass
        
        Args:
            cls_features: CLS features, shape (B, embed_dim)
            
        Returns:
            topic_representation: topic representation (B, num_topics)
            reconstructed_features: reconstructed features (B, embed_dim)
        """
        # 1. Map features into the topic space
        doc_topic = self.feature_to_topic(cls_features)  # (B, num_topics)
        # Keep the output nonnegative
        doc_topic = F.relu(doc_topic)
        # Normalize so the topic weights form a distribution
        doc_topic = doc_topic / (doc_topic.sum(dim=1, keepdim=True) + 1e-10)
        
        # 2. Reconstruct features from the topic basis
        reconstructed_features = torch.matmul(doc_topic, self.topic_basis)  # (B, embed_dim)
        
        return doc_topic, reconstructed_features
    
    def compute_supervised_loss(self, cls_features, labels):
        """
        Supervised topic loss used during training
        
        Args:
            cls_features: CLS features (B, embed_dim)
            labels: ground-truth labels (B,)
            
        Returns:
            loss: supervised loss
        """
        doc_topic, reconstructed_features = self.forward(cls_features)
        
        # 1. Reconstruction loss
        recon_loss = F.mse_loss(reconstructed_features, cls_features)
        
        # 2. Class-prototype loss: topics of one class stay near its prototype
        proto_loss = 0.0
        for class_idx in range(self.num_classes):
            class_mask = (labels == class_idx)
            if class_mask.sum() > 0:
                class_topics = doc_topic[class_mask]  # (n_class_samples, num_topics)
                class_proto = self.class_prototypes[class_idx:class_idx+1]  # (1, num_topics)
                # Distance between the class-mean topic and the class prototype
                proto_loss += F.mse_loss(class_topics.mean(dim=0, keepdim=True), class_proto)
        
        proto_loss = proto_loss / self.num_classes
        
        # 3. Diversity loss: discourage identical topic bases
        topic_similarity = torch.matmul(self.topic_basis, self.topic_basis.t())  # (num_topics, num_topics)
        mask = torch.eye(self.num_topics, device=topic_similarity.device).bool()
        topic_similarity = topic_similarity.masked_fill(mask, 0)
        diversity_loss = torch.mean(torch.abs(topic_similarity))
        
        # Combined loss
        total_loss = recon_loss + 0.1 * proto_loss + 0.01 * diversity_loss
        
        return total_loss
    
    def get_topic_representation(self, cls_features):
        """
        Return the topic representation used by PGCL
        
        Args:
            cls_features: CLS features
            
        Returns:
            doc_topic: document-topic distribution (B, num_topics)
        """
        self.eval()
        with torch.no_grad():
            doc_topic, _ = self.forward(cls_features)
        return doc_topic


if __name__ == "__main__":
    # Smoke test
    batch_size = 32
    embed_dim = 64
    num_topics = 175
    num_classes = 9
    
    model = SupervisedTopicModelForPGCL(
        embed_dim=embed_dim,
        num_topics=num_topics,
        num_classes=num_classes
    )
    
    # Synthetic input
    cls_features = torch.randn(batch_size, embed_dim)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # Forward pass
    doc_topic, recon_features = model(cls_features)
    
    print(f"Input feature shape: {cls_features.shape}")
    print(f"Topic shape: {doc_topic.shape}")
    print(f"Reconstructed feature shape: {recon_features.shape}")
    
    # Compute the loss
    loss = model.compute_supervised_loss(cls_features, labels)
    print(f"\nSupervised loss: {loss.item():.4f}")


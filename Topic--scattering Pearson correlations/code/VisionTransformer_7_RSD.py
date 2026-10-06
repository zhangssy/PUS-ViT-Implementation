import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=12, embed_dim=64, img_size=7, patch_size=1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Patch embedding implemented with Conv2d
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H/ps, W/ps)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, mlp_dim=128, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Self-attention
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        # Feed-forward network
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=7, patch_size=1, in_channels=12, num_classes=9,
                 embed_dim=64, depth=3, num_heads=4, mlp_dim=128, dropout=0.1,
                 num_topics=175, use_pgcl=True, use_dual_branch=True):
        """
        Initialize the dual-branch Vision Transformer
        
        Args:
            img_size: image size
            patch_size: patch size
            in_channels: number of input channels
            num_classes: number of classes
            embed_dim: embedding dimension
            depth: Transformer depth (ViT encoder on branch 2)
            num_heads: number of attention heads
            mlp_dim: MLP width
            dropout: dropout rate
            num_topics: number of topics for the supervised topic model
            use_pgcl: whether to use PGCL
            use_dual_branch: whether to use the dual-branch architecture
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.use_pgcl = use_pgcl
        self.use_dual_branch = use_dual_branch
        self.img_size = img_size
        self.in_channels = in_channels
        
        # Lazy import to avoid a circular dependency
        from SupervisedTopicModelForPGCL import SupervisedTopicModelForPGCL
        from PhysicsGuidedConvLayer import PhysicsGuidedConvLayer
        
        self.pgcl_hidden_dim = 128
        
        if use_dual_branch and use_pgcl:
            # ========== Branch 2: ViT encoder on the RSD patch ==========
            self.patch_embed = PatchEmbedding(in_channels, embed_dim, img_size, patch_size)
            num_patches = self.patch_embed.num_patches

            # Classification token
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            # Positional embedding
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
            self.pos_drop = nn.Dropout(p=dropout)

            # Stacked ViT encoder blocks
            self.blocks = nn.ModuleList([
                TransformerEncoder(embed_dim, num_heads, mlp_dim, dropout) for _ in range(depth)
            ])
            self.norm = nn.LayerNorm(embed_dim)
            
            # ========== Dual-branch fusion PGCL ==========
            # Import the fusion PGCL
            from PhysicsGuidedConvLayer import DualBranchFusionPGCL
            
            # Fusion PGCL containing branch 1 and branch 2
            self.fusion_pgcl = DualBranchFusionPGCL(
                embed_dim=embed_dim,
                num_topics=num_topics,
                num_classes=num_classes,
                hidden_dim=self.pgcl_hidden_dim,
                dropout=dropout,
                in_channels=in_channels,
                img_size=img_size
            )
            
            # Final classifier
            self.pgcl_classifier = nn.Sequential(
                nn.LayerNorm(self.pgcl_hidden_dim),
                nn.GELU(),
                nn.Linear(self.pgcl_hidden_dim, num_classes)
            )
            
            # Keep topic_model and pgcl as aliases of the fusion PGCL for compatibility
            # These aliases are used when loading weights for T-matrix training; fusion_pgcl is the module that runs
            self.topic_model = self.fusion_pgcl.branch2_topic_model
            self.pgcl = self.fusion_pgcl.branch2_pgcl
            
        elif use_pgcl:
            # Single-branch mode, kept for compatibility
            self.patch_embed = PatchEmbedding(in_channels, embed_dim, img_size, patch_size)
            num_patches = self.patch_embed.num_patches

            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
            self.pos_drop = nn.Dropout(p=dropout)

            self.blocks = nn.ModuleList([
                TransformerEncoder(embed_dim, num_heads, mlp_dim, dropout) for _ in range(depth)
            ])
            self.norm = nn.LayerNorm(embed_dim)
            
            self.topic_model = SupervisedTopicModelForPGCL(
                embed_dim=embed_dim,
                num_topics=num_topics,
                num_classes=num_classes
            )
            
            self.pgcl = PhysicsGuidedConvLayer(
                embed_dim=embed_dim,
                num_topics=num_topics,
                num_classes=num_classes,
                hidden_dim=self.pgcl_hidden_dim
            )
            
            self.pgcl_classifier = nn.Sequential(
                nn.LayerNorm(self.pgcl_hidden_dim),
                nn.GELU(),
                nn.Linear(self.pgcl_hidden_dim, num_classes)
            )
        else:
            # Original path without PGCL
            self.patch_embed = PatchEmbedding(in_channels, embed_dim, img_size, patch_size)
            num_patches = self.patch_embed.num_patches

            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
            self.pos_drop = nn.Dropout(p=dropout)

            self.blocks = nn.ModuleList([
                TransformerEncoder(embed_dim, num_heads, mlp_dim, dropout) for _ in range(depth)
            ])
            self.norm = nn.LayerNorm(embed_dim)
            self.head = nn.Linear(embed_dim, num_classes)

        # Initialize parameters
        if hasattr(self, 'pos_embed'):
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if hasattr(self, 'cls_token'):
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        if hasattr(self, 'head'):
            nn.init.trunc_normal_(self.head.weight, std=0.02)

    def forward(self, x):
        B = x.size(0)
        
        # Dual-branch forward
        if self.use_dual_branch and self.use_pgcl:
            # ========== Branch 2: ViT encoder on the RSD patch ==========
            # Extract features with the ViT encoder
            x_vit = self.patch_embed(x)  # (B, num_patches, embed_dim)
            
            cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
            x_vit = torch.cat((cls_tokens, x_vit), dim=1)  # (B, 1+num_patches, embed_dim)
            x_vit = x_vit + self.pos_embed
            x_vit = self.pos_drop(x_vit)
            
            # Run the stacked ViT blocks
            for blk in self.blocks:
                x_vit = blk(x_vit)
            
            x_vit = self.norm(x_vit)
            cls_features = x_vit[:, 0]  # CLS token (B, embed_dim)
            
            # ========== Fusion PGCL ==========
            # The fusion PGCL contains:
            # - branch 1: raw RSD patch into the topic model
            # - branch 2: CLS feature into the topic model
            # - fusion of the two enhanced features
            enhanced_features, branch1_topic, branch2_topic = self.fusion_pgcl(
                cls_features, raw_data=x  # raw RSD patch for branch 1
            )  # (B, hidden_dim)
            
            # Final classification
            logits = self.pgcl_classifier(enhanced_features)  # (B, num_classes)
            
            # Return branch-2 CLS and topic for compatibility
            return logits, cls_features, branch2_topic
            
        elif self.use_pgcl:
            # Single-branch mode, kept for compatibility
            x = self.patch_embed(x)  # (B, num_patches, embed_dim)

            cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
            x = torch.cat((cls_tokens, x), dim=1)  # (B, 1+num_patches, embed_dim)
            x = x + self.pos_embed
            x = self.pos_drop(x)

            for blk in self.blocks:
                x = blk(x)

            x = self.norm(x)
            cls_out = x[:, 0]  # CLS token
            
            topic_representation, _ = self.topic_model(cls_out)  # (B, num_topics)
            enhanced_features = self.pgcl(cls_out, topic_representation)  # (B, hidden_dim)
            logits = self.pgcl_classifier(enhanced_features)
            return logits, cls_out, topic_representation
        else:
            # Original path without PGCL
            x = self.patch_embed(x)  # (B, num_patches, embed_dim)

            cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
            x = torch.cat((cls_tokens, x), dim=1)  # (B, 1+num_patches, embed_dim)
            x = x + self.pos_embed
            x = self.pos_drop(x)

            for blk in self.blocks:
                x = blk(x)

            x = self.norm(x)
            cls_out = x[:, 0]  # CLS token
            
            out = self.head(cls_out)
            return out, cls_out
    
    def extract_features(self, x):
        """
        Extract the CLS feature
        
        Args:
            x: input tensor, shape (B, C, H, W)
            
        Returns:
            features: CLS feature, shape (B, embed_dim)
        """
        B = x.size(0)
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 1+num_patches, embed_dim)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_out = x[:, 0]  # CLS token
        return cls_out
    
    def get_topic_representation(self, x):
        """
        Return the topic representation for analysis
        
        Args:
            x: input tensor, shape (B, C, H, W)
            
        Returns:
            topic_representation: topic representation, shape (B, num_topics)
        """
        if not self.use_pgcl:
            raise ValueError("PGCL is disabled, so no topic representation is available")
        
        cls_features = self.extract_features(x)
        topic_representation = self.topic_model.get_topic_representation(cls_features)
        return topic_representation


# if __name__ == "__main__":
#     batch_size = 32
#     in_channels = 12
#     H, W = 7, 7
#     num_classes = 7
# 
#     model = VisionTransformer(img_size=H, in_channels=in_channels, num_classes=num_classes)
#     dummy_input = torch.randn(batch_size, in_channels, H, W)
#     output = model(dummy_input)
# 
#     print("Input shape:", dummy_input.shape)   # (32,12,7,7)
#     print("Output shape:", output.shape)      # (32,7)

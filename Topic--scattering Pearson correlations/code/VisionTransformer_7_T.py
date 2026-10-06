import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=10, embed_dim=64, img_size=7, patch_size=1):
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
    def __init__(self, embed_dim=64, num_heads=4, mlp_dim=108, dropout=0.1):
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
    def __init__(self, img_size=7, patch_size=1, in_channels=10, num_classes=9,
                 embed_dim=64, depth=4, num_heads=4, mlp_dim=108, dropout=0.1,
                 num_topics=175, use_pgcl=False, rsd_model_path=None):
        """
        Initialize the Vision Transformer
        
        Args:
            img_size: image size
            patch_size: patch size
            in_channels: number of input channels
            num_classes: number of classes
            embed_dim: embedding dimension
            depth: Transformer depth
            num_heads: number of attention heads
            mlp_dim: MLP width
            dropout: dropout rate
            num_topics: number of topics for the supervised topic model
            use_pgcl: whether to use PGCL
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.use_pgcl = use_pgcl
        self.depth = depth
        
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, img_size, patch_size)
        num_patches = self.patch_embed.num_patches

        # Classification token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads, mlp_dim, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
        # PGCL modules loaded from the RSD checkpoint and applied after encoders 1-4
        self.rsd_pgcl_layers = nn.ModuleList()
        if rsd_model_path is not None:
            self._load_rsd_pgcl_layers(rsd_model_path, embed_dim, num_topics, num_classes)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize parameters
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
    
    def _load_rsd_pgcl_layers(self, rsd_model_path, embed_dim, num_topics, num_classes):
        """
        Load PGCL weights from an RSD checkpoint
        
        Args:
            rsd_model_path: path to the RSD checkpoint
            embed_dim: embedding dimension
            num_topics: number of topics
            num_classes: number of classes
        """
        try:
            # Load the RSD checkpoint
            rsd_state_dict = torch.load(rsd_model_path, map_location='cpu')
            
            # Import the topic and PGCL modules
            from SupervisedTopicModelForPGCL import SupervisedTopicModelForPGCL
            from PhysicsGuidedConvLayer import PhysicsGuidedConvLayer
            
            # Try to load the dual-branch fusion PGCL
            from PhysicsGuidedConvLayer import DualBranchFusionPGCL
            fusion_pgcl_keys = {k.replace('fusion_pgcl.', ''): v 
                                for k, v in rsd_state_dict.items() 
                                if 'fusion_pgcl' in k}
            
            if fusion_pgcl_keys:
                # Use the dual-branch fusion PGCL
                fusion_pgcl = DualBranchFusionPGCL(
                    embed_dim=embed_dim,
                    num_topics=num_topics,
                    num_classes=num_classes,
                    hidden_dim=128,
                    dropout=0.1,
                    in_channels=10,  # T-matrix channel count
                    img_size=7
                )
                
                # Drop keys whose shapes differ; branch1_projection differs because RSD and T have different channel counts
                fusion_pgcl_dict = fusion_pgcl.state_dict()
                filtered_fusion_keys = {}
                skipped_keys = []
                
                for k, v in fusion_pgcl_keys.items():
                    if k in fusion_pgcl_dict:
                        # Keep the key only when the shapes match
                        if fusion_pgcl_dict[k].shape == v.shape:
                            filtered_fusion_keys[k] = v
                        else:
                            skipped_keys.append(f"{k}: checkpoint shape {v.shape} != model shape {fusion_pgcl_dict[k].shape}")
                    else:
                        skipped_keys.append(f"{k}: key not in model")
                
                if skipped_keys:
                    print(f"  Warning: skipped keys (shape mismatch or missing from the model):")
                    for key in skipped_keys[:5]:  # print at most five keys
                        print(f"    - {key}")
                    if len(skipped_keys) > 5:
                        print(f"    ... and {len(skipped_keys) - 5} more keys were skipped")
                    print(f"  Note: branch1_projection is skipped because the channel counts differ (RSD 12 vs T 10) and keeps its random initialization")
                
                # Load the filtered weights
                missing_keys, unexpected_keys = fusion_pgcl.load_state_dict(filtered_fusion_keys, strict=False)
                if missing_keys:
                    missing_important = [k for k in missing_keys if 'branch1_projection' not in k]
                    if missing_important:
                        print(f"  Warning: missing keys: {len(missing_important)} (missing branch1_projection keys are expected)")
                
                print(f"  Loaded the dual-branch fusion PGCL")
                
                # Attach one PGCL module after each of the first four encoders (indices 0-3)
                # The four encoders share one fusion PGCL
                for i in [0, 1, 2, 3]:
                    projection = nn.Linear(128, embed_dim)
                    pgcl_module = nn.ModuleDict({
                        'fusion_pgcl': fusion_pgcl,  # shared fusion PGCL
                        'projection': projection
                    })
                    self.rsd_pgcl_layers.append(pgcl_module)
            else:
                # Fall back to the single-branch loader
                print(f"  Fusion PGCL weights were not found; using the single-branch architecture")
                for i in [0, 1, 2, 3]:
                    # Create the topic model
                    topic_model = SupervisedTopicModelForPGCL(
                        embed_dim=embed_dim,
                        num_topics=num_topics,
                        num_classes=num_classes
                    )
                    
                    # Create PGCL
                    pgcl = PhysicsGuidedConvLayer(
                        embed_dim=embed_dim,
                        num_topics=num_topics,
                        num_classes=num_classes,
                        hidden_dim=128
                    )
                    
                    projection = nn.Linear(128, embed_dim)
                    
                    # Load single-branch weights
                    topic_model_keys = {k.replace('topic_model.', ''): v 
                                       for k, v in rsd_state_dict.items() 
                                       if 'topic_model' in k and 'fusion_pgcl' not in k}
                    pgcl_keys = {k.replace('pgcl.', ''): v 
                                 for k, v in rsd_state_dict.items() 
                                 if 'pgcl' in k and 'fusion_pgcl' not in k}
                    
                    if topic_model_keys:
                        topic_model.load_state_dict(topic_model_keys, strict=False)
                    if pgcl_keys:
                        pgcl.load_state_dict(pgcl_keys, strict=False)
                    
                    # Pack topic_model, pgcl, and projection into one module
                    pgcl_module = nn.ModuleDict({
                        'topic_model': topic_model,
                        'pgcl': pgcl,
                        'projection': projection
                    })
                    
                    self.rsd_pgcl_layers.append(pgcl_module)
            
            print(f"Successfully loaded RSD PGCL layers from {rsd_model_path}")
        except Exception as e:
            print(f"Warning: Failed to load RSD PGCL layers: {e}")
            print("Creating new PGCL layers instead...")
            # If loading fails, create new layers
            from SupervisedTopicModelForPGCL import SupervisedTopicModelForPGCL
            from PhysicsGuidedConvLayer import PhysicsGuidedConvLayer
            
            for i in [0, 1, 2, 3]:
                topic_model = SupervisedTopicModelForPGCL(
                    embed_dim=embed_dim,
                    num_topics=num_topics,
                    num_classes=num_classes
                )
                pgcl = PhysicsGuidedConvLayer(
                    embed_dim=embed_dim,
                    num_topics=num_topics,
                    num_classes=num_classes,
                    hidden_dim=128
                )
                projection = nn.Linear(128, embed_dim)
                pgcl_module = nn.ModuleDict({
                    'topic_model': topic_model,
                    'pgcl': pgcl,
                    'projection': projection
                })
                self.rsd_pgcl_layers.append(pgcl_module)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 1+num_patches, embed_dim)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Run the Transformer and insert the RSD PGCL after encoders 1-4
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            
            # Apply the RSD PGCL after encoder indices 0, 1, 2, and 3
            if i in [0, 1, 2, 3] and len(self.rsd_pgcl_layers) > 0:
                # Current CLS feature
                current_cls = x[:, 0]  # (B, embed_dim)
                
                # PGCL module for this encoder
                pgcl_idx = i
                if pgcl_idx < len(self.rsd_pgcl_layers):
                    rsd_module = self.rsd_pgcl_layers[pgcl_idx]
                    
                    # Dual-branch fusion PGCL, if present
                    if 'fusion_pgcl' in rsd_module:
                        # Use the dual-branch fusion PGCL
                        # The fusion PGCL takes the CLS feature and fuses its two branches
                        enhanced_features, _, _ = rsd_module['fusion_pgcl'](
                            current_cls, raw_data=None  # T-matrix training uses only the CLS feature
                        )  # (B, hidden_dim)
                    else:
                        # Single-branch path
                        # Topic representation from the topic model
                        topic_representation, _ = rsd_module['topic_model'](current_cls)  # (B, num_topics)
                        
                        # Enhanced features from PGCL
                        enhanced_features = rsd_module['pgcl'](current_cls, topic_representation)  # (B, hidden_dim)
                    
                    # Project back to embed_dim
                    projected_features = rsd_module['projection'](enhanced_features)  # (B, embed_dim)
                    
                    # Rebuild the sequence instead of editing a view in place
                    updated_cls = current_cls + projected_features  # (B, embed_dim)
                    x = torch.cat([updated_cls.unsqueeze(1), x[:, 1:]], dim=1)

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
#     in_channels = 10
#     H, W = 7, 7
#     num_classes = 7
# 
#     model = VisionTransformer(img_size=H, in_channels=in_channels, num_classes=num_classes)
#     dummy_input = torch.randn(batch_size, in_channels, H, W)
#     output = model(dummy_input)
# 
#     print("Input shape:", dummy_input.shape)   # (32,10,7,7)
#     print("Output shape:", output.shape)      # (32,7)

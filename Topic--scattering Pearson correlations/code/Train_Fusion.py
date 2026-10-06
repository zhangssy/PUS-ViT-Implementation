"""
Two-stage training:
1. Train the RSD Vision Transformer with PGCL and keep the topic model and PGCL.
2. Train the T-matrix Vision Transformer and use the pretrained PGCL modules as guidance.
"""

import os
import copy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

from VisionTransformer_7_RSD import VisionTransformer as ViT_RSD
from VisionTransformer_7_T import VisionTransformer as ViT_T


class TxtDataset(Dataset):
    def __init__(self, txt_path):
        samples = []
        with open(txt_path, "r") as fh:
            for line in fh:
                items = line.strip().split()
                if len(items) >= 2:
                    samples.append((items[0], int(items[1])))
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def del_tensor_ele(arr, index):
    arr1 = arr[0:index]
    arr2 = arr[index + 1:]
    return torch.cat((arr1, arr2), dim=0)


def calc_loss(outputs, labels, device):
    outputs = outputs.to(device)
    labels = labels.to(device)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, labels)
    return loss.mean()


def build_dataloader(txt_path, batch_size, shuffle, num_workers=0):
    dataset = TxtDataset(txt_path)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def train_rsd_with_pgcl(config):
    """
    Train the RSD PGCL model and return its topic and PGCL modules
    """
    device = config["device"]
    use_pgcl = True

    # Read the channel count from the first training patch
    train_loader_temp = build_dataloader(config["train_txt"], batch_size=1, shuffle=False)
    actual_in_channels = None
    for inputs_list, _ in train_loader_temp:
        if len(inputs_list) > 0:
            sample_tensor = torch.load(inputs_list[0]).to(device)
            if len(sample_tensor.shape) == 4:  # (B, C, H, W)
                actual_in_channels = sample_tensor.shape[1]
            elif len(sample_tensor.shape) == 3:  # (C, H, W)
                actual_in_channels = sample_tensor.shape[0]
            break
    
    if actual_in_channels is None:
        raise RuntimeError("Could not read the channel count from the training patches. Check the data files.")
    
    print(f"Detected input channels: {actual_in_channels} (configured rsd_in_channels: {config.get('rsd_in_channels', 'not set')})")
    
    # Dual-branch architecture is on unless the config disables it
    use_dual_branch = config.get("use_dual_branch", True)
    
    model = ViT_RSD(
        embed_dim=64,
        num_classes=config["num_classes"],
        num_topics=config["num_topics"],
        use_pgcl=use_pgcl,
        depth=config.get("rsd_depth", 3),
        in_channels=actual_in_channels,  # channel count read from the data
        use_dual_branch=use_dual_branch  # use the dual-branch architecture
    ).to(device)
    
    if use_dual_branch:
        print("Dual-branch RSD architecture is enabled:")
        print("  - Branch 1: the RSD patch goes directly into the topic model")
        print("  - Branch 2: a ViT encoder extracts a CLS feature, which then enters the topic model")
        print("  - Fusion: the enhanced features of the two branches are combined")

    # Freeze PGCL parameters so they are not updated
    if use_dual_branch and hasattr(model, 'fusion_pgcl'):
        # Dual branch: freeze only branch1_pgcl and branch2_pgcl
        # topic_model and fusion_layer stay trainable
        if hasattr(model.fusion_pgcl, 'branch1_pgcl'):
            for p in model.fusion_pgcl.branch1_pgcl.parameters():
                p.requires_grad = False
        if hasattr(model.fusion_pgcl, 'branch2_pgcl'):
            for p in model.fusion_pgcl.branch2_pgcl.parameters():
                p.requires_grad = False
        print("[OK] PhysicsGuidedConvLayer (branch1_pgcl and branch2_pgcl) is frozen")
        print("  topic_model and fusion_layer remain trainable")
    elif hasattr(model, 'pgcl'):
        # Single branch: freeze every pgcl parameter
        for p in model.pgcl.parameters():
            p.requires_grad = False
        print("[OK] PhysicsGuidedConvLayer (pgcl) is frozen")

    # Optimize only parameters that require gradients
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
    
    optimizer = torch.optim.AdamW(trainable_params, lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=config["epochs"])
    
    # Report how many parameters are trainable
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params_count = sum(p.numel() for p in trainable_params)
    frozen_params_count = total_params - trainable_params_count
    print(f"  Total parameters: {total_params:,}, trainable: {trainable_params_count:,}, frozen: {frozen_params_count:,}")

    train_loader = build_dataloader(config["train_txt"], config["batch_size"], shuffle=True)
    test_loader = build_dataloader(config["test_txt"], config["batch_size"], shuffle=False)

    best_acc = 0
    best_path = None
    os.makedirs(config["save_dir"], exist_ok=True)

    print("=" * 60)
    print("Stage-1: train the RSD PGCL model")
    print("=" * 60)

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        total_cls_loss = 0
        total_topic_loss = 0
        count = 0

        for inputs_list, labels in train_loader:
            labels = labels.to(device)
            # Stack the batch from the patch files
            batch_tensor = None
            for sample_path in inputs_list:
                sample_tensor = torch.load(sample_path).to(device)  # expected shape (B, C, H, W) or (1, C, H, W)
                if batch_tensor is None:
                    batch_tensor = sample_tensor
                else:
                    batch_tensor = torch.cat((batch_tensor, sample_tensor), dim=0)

            optimizer.zero_grad()
            logits, cls_features, topic_representation = model(batch_tensor)
            cls_loss = calc_loss(logits, labels, device)
            
            # Dual branch: topic loss from both branches
            if hasattr(model, 'use_dual_branch') and model.use_dual_branch:
                # Branch-1 features from the fusion PGCL projection
                branch1_features = model.fusion_pgcl.branch1_projection(batch_tensor)
                branch1_topic_loss = model.fusion_pgcl.branch1_topic_model.compute_supervised_loss(branch1_features, labels)
                
                # Branch-2 topic loss from the CLS feature
                branch2_topic_loss = model.fusion_pgcl.branch2_topic_model.compute_supervised_loss(cls_features, labels)
                
                # Sum the two topic losses
                topic_loss = branch1_topic_loss + branch2_topic_loss
            else:
                # Single-branch mode
                topic_loss = model.topic_model.compute_supervised_loss(cls_features, labels)
            
            total_loss_batch = cls_loss + config["topic_lambda"] * topic_loss
            total_loss_batch.backward()
            optimizer.step()

            total_loss += total_loss_batch.item()
            total_cls_loss += cls_loss.item()
            total_topic_loss += topic_loss.item()
            count += 1

        avg_loss = total_loss / count
        print(f"[RSD][Epoch {epoch+1}] Loss: {avg_loss:.4f} (Cls: {total_cls_loss/count:.4f}, Topic: {total_topic_loss/count:.4f})")

        scheduler.step()

        # Validation
        acc = evaluate(model, test_loader, device, use_pgcl=True)
        if acc >= best_acc:
            best_acc = acc
            # Name the checkpoint by its validation accuracy
            best_path = os.path.join(
                config["save_dir"],
                f"pgcl_model_best_{best_acc:.2f}.pth"
            )
            torch.save(model.state_dict(), best_path)
            print(f"  Saved best PGCL model: {best_path} (Acc: {best_acc:.2f}%)")

    # Load the best checkpoint
    if best_path is not None:
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"RSD PGCL stage finished. Best accuracy: {best_acc:.2f}% (path: {best_path})")
    else:
        raise RuntimeError("Stage-1 did not save a PGCL checkpoint. Check training and validation.")

    # Dual branch: also return the fused PGCL module
    if hasattr(model, 'use_dual_branch') and model.use_dual_branch:
        guidance_fusion_pgcl = copy.deepcopy(model.fusion_pgcl).eval()
        for p in guidance_fusion_pgcl.parameters():
            p.requires_grad = False
        # Also return branch-2 topic_model and pgcl for compatibility
        guidance_topic = copy.deepcopy(model.fusion_pgcl.branch2_topic_model).eval()
        guidance_pgcl = copy.deepcopy(model.fusion_pgcl.branch2_pgcl).eval()
        for p in guidance_topic.parameters():
            p.requires_grad = False
        for p in guidance_pgcl.parameters():
            p.requires_grad = False
        return guidance_topic, guidance_pgcl, best_path, guidance_fusion_pgcl
    else:
        # Single-branch mode
        guidance_topic = copy.deepcopy(model.topic_model).eval()
        guidance_pgcl = copy.deepcopy(model.pgcl).eval()
        for p in guidance_topic.parameters():
            p.requires_grad = False
        for p in guidance_pgcl.parameters():
            p.requires_grad = False
        return guidance_topic, guidance_pgcl, best_path, None


def evaluate(model, data_loader, device, use_pgcl, in_channels=None):
    """
    Evaluate the RSD model

    Args:
        in_channels: deprecated; the channel count is read from the data
    """
    model.eval()
    total_preds = []
    total_labels = []

    with torch.no_grad():
        for inputs_list, labels in data_loader:
            labels = labels.to(device)
            batch_tensor = None
            for sample_path in inputs_list:
                sample_tensor = torch.load(sample_path).to(device)
                if batch_tensor is None:
                    batch_tensor = sample_tensor
                else:
                    batch_tensor = torch.cat((batch_tensor, sample_tensor), dim=0)

            outputs = model(batch_tensor)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            total_preds.append(preds)
            total_labels.append(labels.cpu().numpy())

    preds = np.concatenate(total_preds)
    labels = np.concatenate(total_labels)
    acc = round((preds == labels).sum() / len(labels) * 100, 2)
    print(f"  Eval Acc: {acc:.2f}%")
    return acc


class GuidedVisionTransformerT(nn.Module):
    """
    T-matrix Vision Transformer guided by a pretrained PGCL module
    """
    def __init__(self, base_model, guidance_topic, guidance_pgcl, num_classes, fusion_dim=128):
        super().__init__()
        self.backbone = base_model  # VisionTransformer_7_T
        self.guidance_topic = guidance_topic
        self.guidance_pgcl = guidance_pgcl
        self.num_classes = num_classes

        hidden_dim = fusion_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(self.backbone.embed_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        logits_base, cls_features = self.backbone(x)  # (B, num_classes), (B, embed_dim)

        with torch.no_grad():
            topic_representation, _ = self.guidance_topic(cls_features)
            guided_features = self.guidance_pgcl(cls_features, topic_representation)  # (B, hidden_dim)

        fused_feature = torch.cat([cls_features, guided_features], dim=1)
        fusion_logits = self.fusion_head(fused_feature)

        return fusion_logits, logits_base


class GuidedVisionTransformerT_NoPGCL(nn.Module):
    """
    B2 ablation: Stage 2 does not insert PGCL and does not use guidance_pgcl.
    Classify from the backbone CLS feature. Knowledge distillation to the base head is kept.
    """
    def __init__(self, base_model, num_classes, fusion_dim=128):
        super().__init__()
        self.backbone = base_model
        self.num_classes = num_classes
        hidden_dim = fusion_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(self.backbone.embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        logits_base, cls_features = self.backbone(x)
        fusion_logits = self.fusion_head(cls_features)
        return fusion_logits, logits_base


def train_t_with_guidance(config, guidance_topic, guidance_pgcl, rsd_model_path):
    """
    Stage-2 training.
    ablation:
      - \"full\": insert RSD-PGCL and use guidance_pgcl (sigmoid gate)
      - \"B2\":   no inserted PGCL and no guidance_pgcl; CLS features go to the classifier
    """
    device = config["device"]
    ablation = config.get("ablation", "B2")  # paper2_8 default is B2

    if ablation == "B2":
        print("=" * 60)
        print("Stage-2 B2: no inserted PGCL and no guidance_pgcl (CLS classifier only)")
        print("=" * 60)
        base_model = ViT_T(
            embed_dim=64,
            num_classes=config["num_classes"],
            num_topics=config["num_topics"],
            use_pgcl=False,
            rsd_model_path=None,  # do not load or insert PGCL after the encoder
        ).to(device)
        guided_model = GuidedVisionTransformerT_NoPGCL(
            base_model, num_classes=config["num_classes"]
        ).to(device)
        trainable_params = list(guided_model.backbone.parameters()) + list(
            guided_model.fusion_head.parameters()
        )
    else:
        print("=" * 60)
        print("Stage-2 full: train on the T-matrix data with PGCL guidance")
        print("=" * 60)
        base_model = ViT_T(
            embed_dim=64,
            num_classes=config["num_classes"],
            num_topics=config["num_topics"],
            use_pgcl=False,
            rsd_model_path=rsd_model_path,
        ).to(device)
        guided_model = GuidedVisionTransformerT(
            base_model,
            guidance_topic.to(device),
            guidance_pgcl.to(device),
            num_classes=config["num_classes"],
        ).to(device)
        for p in guided_model.guidance_topic.parameters():
            p.requires_grad = False
        for p in guided_model.guidance_pgcl.parameters():
            p.requires_grad = False
        print("[OK] guidance_topic and guidance_pgcl are frozen")
        trainable_params = list(guided_model.backbone.parameters()) + list(
            guided_model.fusion_head.parameters()
        )

    optimizer = torch.optim.AdamW(trainable_params, lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=config["epochs"]
    )

    train_loader = build_dataloader(config["train_txt"], config["batch_size"], shuffle=True)
    test_loader = build_dataloader(config["test_txt"], config["batch_size"], shuffle=False)

    best_acc = 0
    best_path = None
    os.makedirs(config["save_dir"], exist_ok=True)

    for epoch in range(config["epochs"]):
        guided_model.train()
        total_loss = 0
        count = 0

        for inputs_list, labels in train_loader:
            labels = labels.to(device)
            batch_tensor = None
            for sample_path in inputs_list:
                sample_tensor = torch.load(sample_path, map_location=device)
                if hasattr(sample_tensor, "to"):
                    sample_tensor = sample_tensor.to(device)
                if batch_tensor is None:
                    batch_tensor = sample_tensor
                else:
                    batch_tensor = torch.cat((batch_tensor, sample_tensor), dim=0)

            optimizer.zero_grad()
            fusion_logits, logits_base = guided_model(batch_tensor)

            ce_loss = calc_loss(fusion_logits, labels, device)
            kd_loss = torch.nn.functional.kl_div(
                torch.log_softmax(fusion_logits, dim=1),
                torch.softmax(logits_base.detach(), dim=1),
                reduction="batchmean",
            )
            total_batch_loss = ce_loss + config["kd_lambda"] * kd_loss
            total_batch_loss.backward()
            optimizer.step()

            total_loss += total_batch_loss.item()
            count += 1

        avg_loss = total_loss / count
        print(f"[T][Epoch {epoch+1}] Loss: {avg_loss:.4f}")

        scheduler.step()

        acc = evaluate_fusion(guided_model, test_loader, device, config["t_in_channels"])
        if acc >= best_acc:
            best_acc = acc
            best_path = os.path.join(
                config["save_dir"], f"fusion_model_best_{best_acc:.2f}.pth"
            )
            torch.save(guided_model.state_dict(), best_path)
            print(f"  Saved best fusion model: {best_path} (Acc: {best_acc:.2f}%)")

    if best_path is None:
        raise RuntimeError("Stage 2 did not save a fusion checkpoint. Check training and validation.")

    print(f"T-matrix stage finished. Best accuracy: {best_acc:.2f}% (path: {best_path})")
    return best_path


def evaluate_fusion(model, data_loader, device, in_channels):
    model.eval()
    total_preds = []
    total_labels = []

    with torch.no_grad():
        for inputs_list, labels in data_loader:
            labels = labels.to(device)
            batch_tensor = None
            for sample_path in inputs_list:
                sample_tensor = torch.load(sample_path).to(device)
                if batch_tensor is None:
                    batch_tensor = sample_tensor
                else:
                    batch_tensor = torch.cat((batch_tensor, sample_tensor), dim=0)

            fusion_logits, _ = model(batch_tensor)
            preds = torch.argmax(fusion_logits, dim=1).cpu().numpy()
            total_preds.append(preds)
            total_labels.append(labels.cpu().numpy())

    preds = np.concatenate(total_preds)
    labels = np.concatenate(total_labels)
    acc = round((preds == labels).sum() / len(labels) * 100, 2)
    print(f"  Fusion Eval Acc: {acc:.2f}%")
    return acc


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Ablation: B2 (Stage-2 without inserted PGCL / without guidance_pgcl)")

    # Paths are relative to this file. Point them at your own lists and
    # checkpoint folder; do not commit a machine-specific drive path.
    #   ../data/Barnaul/RSD/Train.txt
    #   ../data/Barnaul/RSD/Test.txt
    #   ../data/Barnaul/T/Train.txt
    #   ../data/Barnaul/T/Test.txt
    #   ../checkpoints/
    root = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(root, "..", "data", "Barnaul")
    save_dir = os.path.join(root, "..", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    rsd_config = {
        "device": device,
        "train_txt": os.path.join(data_root, "RSD", "Train.txt"),
        "test_txt": os.path.join(data_root, "RSD", "Test.txt"),
        "batch_size": 32,
        "epochs": 20,
        "lr": 4e-4,
        "num_classes": 9,
        "num_topics": 15,
        "rsd_depth": 3,
        "topic_lambda": 0.1,
        "rsd_in_channels": 12,
        "save_dir": save_dir,
    }

    t_config = {
        "device": device,
        "train_txt": os.path.join(data_root, "T", "Train.txt"),
        "test_txt": os.path.join(data_root, "T", "Test.txt"),
        "batch_size": 32,
        "epochs": 20,
        "lr": 4e-4,
        "num_classes": 9,
        "num_topics": 15,
        "kd_lambda": 0.2,
        "t_in_channels": 10,
        "save_dir": save_dir,
        "ablation": "B2",
    }

    result = train_rsd_with_pgcl(rsd_config)
    if len(result) == 4:
        guidance_topic, guidance_pgcl, rsd_best_path, guidance_fusion_pgcl = result
    else:
        guidance_topic, guidance_pgcl, rsd_best_path = result
        guidance_fusion_pgcl = None
    # B2: Stage 2 intentionally skips guidance_pgcl and does not insert encoder PGCL
    train_t_with_guidance(t_config, guidance_topic, guidance_pgcl, rsd_best_path)


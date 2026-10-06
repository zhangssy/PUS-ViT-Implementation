"""
Extract labeled PolSAR patches and save one ``.pth`` file per patch.

The ground-truth map uses 0 as background. Every nonzero pixel becomes one
window. Patches are split into train and test with a stratified shuffle, then
written as:

    <output_dir>/Train/<class_id>/<class_id>train_idx<class_id>_<i>.pth
    <output_dir>/Test/<class_id>/<class_id>test_idx<class_id>_<i>.pth

Example:
    python extract_data.py ^
        --gt_mat data/Ober/idx7.mat ^
        --feature_mat data/Ober/Train21.mat ^
        --output_dir data/Ober/patches ^
        --class_id 6
"""

from __future__ import annotations

import argparse
import os
import time

import h5py
import numpy as np
import torch
from sklearn.model_selection import train_test_split


def add_zero_padding(x, margin=2):
    """Pad an (H, W, C) cube with zeros."""
    out = np.zeros((x.shape[0] + 2 * margin, x.shape[1] + 2 * margin, x.shape[2]), dtype=x.dtype)
    out[margin : x.shape[0] + margin, margin : x.shape[1] + margin, :] = x
    return out


def create_image_cube(x, gt, pos, window_size):
    """Cut one (window, window, C) patch and its label at each (row, col)."""
    margin = (window_size - 1) // 2
    padded = add_zero_padding(x, margin=margin)
    patches = np.array([padded[i : i + window_size, j : j + window_size, :] for i, j in pos])
    labels = np.array([gt[i, j] for i, j in pos])
    return patches, labels


def create_pos(shape, pos, num):
    """Return ``num`` raster positions starting at ``pos``."""
    if (pos[0] + 1) * (pos[1] + 1) + num > shape[0] * shape[1]:
        num = shape[0] * shape[1] - (pos[0] * shape[1] + pos[1])
    return [(pos[0] + (pos[1] + i) // shape[1], (pos[1] + i) % shape[1]) for i in range(num)]


def labeled_positions(gt):
    """Positions whose ground-truth label is greater than 0."""
    mask = gt > 0
    return [(i, j) for i, row in enumerate(mask) for j, keep in enumerate(row) if keep]


def split_train_test(x, gt, test_ratio, random_state=111):
    """Stratified train/test split. ``test_ratio`` is the test fraction."""
    return train_test_split(x, gt, test_size=test_ratio, random_state=random_state, stratify=gt)


def create_image_patch(image, pos, window_size=65):
    """Cut 2-D windows. Retained for single-channel inputs."""
    margin = (window_size - 1) // 2
    padded = np.zeros((image.shape[0] + 2 * margin, image.shape[1] + 2 * margin), dtype=image.dtype)
    padded[margin : image.shape[0] + margin, margin : image.shape[1] + margin] = image
    return np.array([padded[i : i + window_size, j : j + window_size] for i, j in pos])


def load_h5_array(path, key):
    """Load one array from a MATLAB v7.3 file and return it as a NumPy array."""
    with h5py.File(path, "r") as handle:
        if key not in handle:
            found = ", ".join(k for k in handle.keys() if not k.startswith("#"))
            raise SystemExit(f"{path} has no variable '{key}'. Found: {found}")
        return np.array(handle[key])


def parse_args():
    parser = argparse.ArgumentParser(description="Extract labeled PolSAR patches to .pth files")
    parser.add_argument("--gt_mat", required=True, help="Ground-truth .mat (MATLAB v7.3)")
    parser.add_argument("--gt_key", default="cdata", help="Variable name of the ground-truth map")
    parser.add_argument("--feature_mat", required=True, help="Feature-cube .mat (MATLAB v7.3)")
    parser.add_argument("--feature_key", default="Train21", help="Variable name of the feature cube")
    parser.add_argument("--output_dir", default="patches", help="Root directory for Train/ and Test/")
    parser.add_argument("--class_id", default="6", help="Tag used in the patch folder and file names")
    parser.add_argument("--window_size", type=int, default=11)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1)
    return parser.parse_args()


class PatchDataset(torch.utils.data.Dataset):
    """Patches stored as (N, C, H, W). Labels are shifted so class 1 becomes 0."""

    def __init__(self, patches, labels):
        self.patches = torch.as_tensor(patches, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long) - 1

    def __getitem__(self, index):
        return self.patches[index], self.labels[index]

    def __len__(self):
        return int(self.labels.shape[0])


def save_split(loader, split_name, output_dir, class_id):
    """Save each batch as one .pth file under output_dir/split_name/class_id/."""
    folder = os.path.join(output_dir, split_name, str(class_id))
    os.makedirs(folder, exist_ok=True)
    prefix = "train" if split_name == "Train" else "test"
    for index, (patch, _label) in enumerate(loader):
        name = f"{class_id}{prefix}_idx{class_id}_{index}.pth"
        torch.save(patch, os.path.join(folder, name))
    print(f"Saved {len(loader)} {split_name.lower()} patches to {folder}")


def main():
    args = parse_args()
    started = time.time()

    gt = load_h5_array(args.gt_mat, args.gt_key)
    gt = torch.from_numpy(np.transpose(gt, (1, 0))).numpy()
    cube = load_h5_array(args.feature_mat, args.feature_key)
    cube = torch.from_numpy(np.transpose(cube, (2, 1, 0))).numpy()
    print(f"Feature cube shape (H, W, C) = {cube.shape}")

    patches, labels = create_image_cube(
        cube, gt, labeled_positions(gt), window_size=args.window_size
    )
    patches = np.asarray(patches, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    # (N, H, W, C) -> (N, C, H, W)
    patches = np.transpose(patches, (0, 3, 1, 2))

    x_train, x_test, y_train, y_test = split_train_test(
        patches, labels, args.test_ratio, random_state=args.seed
    )
    train_loader = torch.utils.data.DataLoader(
        PatchDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_loader = torch.utils.data.DataLoader(
        PatchDataset(x_test, y_test),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    save_split(train_loader, "Train", args.output_dir, args.class_id)
    save_split(test_loader, "Test", args.output_dir, args.class_id)
    print(f"Running time: {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()

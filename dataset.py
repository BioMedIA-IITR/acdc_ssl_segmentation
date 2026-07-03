import os
import glob
import random
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils import normalize_image, resize_2d_np


def find_acdc_phase_files(root_dir):
    patterns = [
        os.path.join(root_dir, "**", "*_ED.nii.gz"),
        os.path.join(root_dir, "**", "*_ES.nii.gz"),
        os.path.join(root_dir, "**", "*ED.nii.gz"),
        os.path.join(root_dir, "**", "*ES.nii.gz"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))
    files = sorted(list(set(files)))
    return files


class ACDCSliceDataset(Dataset):
    """
    2D slice dataset for self-supervised ACDC segmentation.
    Returns two augmented views of the same image slice.
    """

    def __init__(self, root_dir, image_size=256, min_std=1e-6, cache=True):
        self.root_dir = root_dir
        self.image_size = image_size
        self.min_std = min_std
        self.cache = cache
        self.items = []

        files = find_acdc_phase_files(root_dir)
        if len(files) == 0:
            raise RuntimeError(f"No ACDC ED/ES files found in: {root_dir}")

        print(f"[Dataset] Found {len(files)} ED/ES volumes.")

        if cache:
            self.slices = []
            for f in files:
                vol = nib.load(f).get_fdata()
                vol = normalize_image(vol)

                for z in range(vol.shape[2]):
                    sl = vol[:, :, z]
                    if np.std(sl) < min_std:
                        continue
                    sl = resize_2d_np(sl, image_size, mode="bilinear")
                    self.slices.append(sl.astype(np.float32))

            print(f"[Dataset] Cached {len(self.slices)} 2D slices.")
        else:
            self.slices = None
            for f in files:
                shape = nib.load(f).shape
                for z in range(shape[2]):
                    self.items.append((f, z))
            print(f"[Dataset] Indexed {len(self.items)} 2D slices.")

    def __len__(self):
        if self.cache:
            return len(self.slices)
        return len(self.items)

    def _load_slice(self, idx):
        if self.cache:
            return self.slices[idx]

        f, z = self.items[idx]
        vol = nib.load(f).get_fdata()
        vol = normalize_image(vol)
        sl = vol[:, :, z]
        sl = resize_2d_np(sl, self.image_size, mode="bilinear")
        return sl.astype(np.float32)

    def augment(self, x):
        # x: tensor [1, H, W]
        if random.random() < 0.5:
            x = torch.flip(x, dims=[2])
        if random.random() < 0.5:
            x = torch.flip(x, dims=[1])

        # small random intensity transform
        scale = 0.85 + 0.30 * random.random()
        shift = -0.15 + 0.30 * random.random()
        x = x * scale + shift

        # Gaussian noise
        x = x + torch.randn_like(x) * 0.04

        # random gamma-like contrast by tanh compression
        if random.random() < 0.3:
            x = torch.tanh(x)

        return x.float()

    def __getitem__(self, idx):
        sl = self._load_slice(idx)
        x = torch.from_numpy(sl).float().unsqueeze(0)

        x1 = self.augment(x.clone())
        x2 = self.augment(x.clone())

        return {
            "view1": x1,
            "view2": x2,
        }


class ACDCVolumeDataset(Dataset):
    """
    Volume-level dataset for inference.
    """

    def __init__(self, root_dir):
        self.files = find_acdc_phase_files(root_dir)
        if len(self.files) == 0:
            raise RuntimeError(f"No ACDC ED/ES files found in: {root_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        nii = nib.load(path)
        vol = nii.get_fdata()
        return {
            "path": path,
            "volume": vol,
            "affine": nii.affine,
            "header": nii.header,
        }

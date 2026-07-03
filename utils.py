import os
import random
import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def normalize_image(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    finite = np.isfinite(img)
    if not finite.all():
        img = np.nan_to_num(img)

    nonzero = img[np.abs(img) > 1e-8]
    if nonzero.size > 10:
        p1, p99 = np.percentile(nonzero, (1, 99))
    else:
        p1, p99 = np.percentile(img, (1, 99))

    img = np.clip(img, p1, p99)
    mean = img.mean()
    std = img.std()
    img = (img - mean) / (std + 1e-8)
    return img.astype(np.float32)


def resize_2d_np(slice_2d: np.ndarray, size: int = 256, mode: str = "bilinear") -> np.ndarray:
    x = torch.from_numpy(slice_2d).float()[None, None]
    if mode == "nearest":
        y = F.interpolate(x, size=(size, size), mode="nearest")
    else:
        y = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return y[0, 0].numpy()


def save_checkpoint(state, path: str):
    torch.save(state, path)


def load_checkpoint(path: str, model, optimizer=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt

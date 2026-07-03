import os
import argparse
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset import find_acdc_phase_files
from models import UNetCluster
from utils import normalize_image, resize_2d_np, ensure_dir


@torch.no_grad()
def segment_volume(model, nii_path, output_path, image_size, device):
    nii = nib.load(nii_path)
    vol = nii.get_fdata()
    affine = nii.affine
    header = nii.header

    vol_norm = normalize_image(vol)
    seg_vol = np.zeros(vol.shape, dtype=np.uint8)

    model.eval()

    for z in range(vol.shape[2]):
        sl = vol_norm[:, :, z]
        sl_resized = resize_2d_np(sl, image_size, mode="bilinear")

        x = torch.from_numpy(sl_resized).float()[None, None].to(device)

        _, probs, _ = model(x)
        seg = torch.argmax(probs, dim=1).float()

        seg = F.interpolate(
            seg[:, None],
            size=vol.shape[:2],
            mode="nearest",
        )

        seg_vol[:, :, z] = seg[0, 0].cpu().numpy().astype(np.uint8)

    out = nib.Nifti1Image(seg_vol, affine=affine, header=header)
    nib.save(out, output_path)


def parse_args():
    p = argparse.ArgumentParser(description="Inference for self-supervised ACDC pseudo-segmentation")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--num_clusters", type=int, default=4)
    p.add_argument("--base_channels", type=int, default=32)
    return p.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Inference] Device: {device}")

    model = UNetCluster(
        in_channels=1,
        num_clusters=args.num_clusters,
        base_channels=args.base_channels,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)

    files = find_acdc_phase_files(args.data_dir)
    print(f"[Inference] Found {len(files)} volumes.")

    for f in tqdm(files):
        name = os.path.basename(f).replace(".nii.gz", "_ssl_seg.nii.gz")
        out_path = os.path.join(args.output_dir, name)
        segment_volume(model, f, out_path, args.image_size, device)

    print("[Inference] Done.")


if __name__ == "__main__":
    main()

import argparse
import os
import glob
import numpy as np
import nibabel as nib
from scipy.ndimage import label


def keep_largest_component(mask):
    lab, n = label(mask)
    if n == 0:
        return mask

    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    largest = sizes.argmax()
    return lab == largest


def postprocess_seg(seg):
    out = np.zeros_like(seg, dtype=np.uint8)

    labels = np.unique(seg)
    labels = labels[labels != 0]

    for c in labels:
        mask = seg == c
        clean = keep_largest_component(mask)
        out[clean] = c

    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.input_dir, "*.nii.gz")))

    for f in files:
        nii = nib.load(f)
        seg = nii.get_fdata().astype(np.uint8)
        out = postprocess_seg(seg)

        out_path = os.path.join(args.output_dir, os.path.basename(f).replace(".nii.gz", "_clean.nii.gz"))
        nib.save(nib.Nifti1Image(out, nii.affine, nii.header), out_path)
        print("Saved:", out_path)


if __name__ == "__main__":
    main()

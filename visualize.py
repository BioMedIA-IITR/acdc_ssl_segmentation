import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--seg", type=str, required=True)
    p.add_argument("--slice", type=int, default=None)
    p.add_argument("--output", type=str, default="overlay.png")
    return p.parse_args()


def main():
    args = parse_args()

    img = nib.load(args.image).get_fdata()
    seg = nib.load(args.seg).get_fdata()

    if args.slice is None:
        z = img.shape[2] // 2
    else:
        z = args.slice

    im = img[:, :, z]
    sg = seg[:, :, z]

    plt.figure(figsize=(6, 6))
    plt.imshow(im.T, cmap="gray", origin="lower")
    plt.imshow(sg.T, alpha=0.35, origin="lower")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

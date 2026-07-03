import os
import argparse
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import ACDCSliceDataset
from models import UNetCluster
from losses import ssl_segmentation_loss
from utils import set_seed, ensure_dir, save_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description="Self-supervised ACDC cardiac MRI pseudo-segmentation training")

    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./runs/acdc_ssl")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--num_clusters", type=int, default=4)
    p.add_argument("--base_channels", type=int, default=32)

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--recon_weight", type=float, default=1.0)
    p.add_argument("--consistency_weight", type=float, default=0.5)
    p.add_argument("--tv_weight", type=float, default=0.05)
    p.add_argument("--entropy_weight", type=float, default=0.01)
    p.add_argument("--balance_weight", type=float, default=0.1)

    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--no_cache", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    ensure_dir(args.output_dir)
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    ensure_dir(ckpt_dir)

    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tensorboard"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")

    dataset = ACDCSliceDataset(
        root_dir=args.data_dir,
        image_size=args.image_size,
        cache=not args.no_cache,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    model = UNetCluster(
        in_channels=1,
        num_clusters=args.num_clusters,
        base_channels=args.base_channels,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-5,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        meter = defaultdict(float)

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")

        for batch in pbar:
            x1 = batch["view1"].to(device, non_blocking=True)
            x2 = batch["view2"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                _, p1, r1 = model(x1)
                _, p2, r2 = model(x2)

                loss, logs = ssl_segmentation_loss(
                    x1=x1,
                    x2=x2,
                    p1=p1,
                    p2=p2,
                    r1=r1,
                    r2=r2,
                    recon_weight=args.recon_weight,
                    consistency_weight=args.consistency_weight,
                    tv_weight=args.tv_weight,
                    entropy_weight=args.entropy_weight,
                    balance_weight=args.balance_weight,
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            for k, v in logs.items():
                meter[k] += v

            pbar.set_postfix(loss=f"{logs['loss']:.4f}")

        n = len(loader)
        avg = {k: v / n for k, v in meter.items()}
        print("[Epoch]", epoch, avg)

        for k, v in avg.items():
            writer.add_scalar(f"train/{k}", v, epoch)

        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "args": vars(args),
            "loss": avg["loss"],
        }

        if avg["loss"] < best_loss:
            best_loss = avg["loss"]
            save_checkpoint(state, os.path.join(ckpt_dir, "best.pth"))
            print(f"[Train] Saved best checkpoint: {best_loss:.5f}")

        if epoch % args.save_every == 0:
            save_checkpoint(state, os.path.join(ckpt_dir, f"epoch_{epoch}.pth"))

    save_checkpoint(state, os.path.join(ckpt_dir, "last.pth"))
    writer.close()
    print("[Train] Finished.")


if __name__ == "__main__":
    main()

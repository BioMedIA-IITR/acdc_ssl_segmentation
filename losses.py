import torch
import torch.nn.functional as F


def reconstruction_loss(recon, image):
    return F.mse_loss(recon, image)


def consistency_loss(p1, p2):
    return F.mse_loss(p1, p2)


def total_variation_loss(probs):
    dx = torch.abs(probs[:, :, :, 1:] - probs[:, :, :, :-1]).mean()
    dy = torch.abs(probs[:, :, 1:, :] - probs[:, :, :-1, :]).mean()
    return dx + dy


def entropy_loss(probs):
    # Minimizes pixel-wise entropy to encourage confident clusters
    ent = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
    return ent.mean()


def balance_loss(probs):
    # Avoids collapse into one cluster
    avg = probs.mean(dim=[0, 2, 3])
    target = torch.ones_like(avg) / probs.shape[1]
    return F.mse_loss(avg, target)


def ssl_segmentation_loss(
    x1,
    x2,
    p1,
    p2,
    r1,
    r2,
    recon_weight=1.0,
    consistency_weight=0.5,
    tv_weight=0.05,
    entropy_weight=0.01,
    balance_weight=0.1,
):
    rec = reconstruction_loss(r1, x1) + reconstruction_loss(r2, x2)
    con = consistency_loss(p1, p2)
    tv = total_variation_loss(p1) + total_variation_loss(p2)
    ent = entropy_loss(p1) + entropy_loss(p2)
    bal = balance_loss(p1) + balance_loss(p2)

    total = (
        recon_weight * rec
        + consistency_weight * con
        + tv_weight * tv
        + entropy_weight * ent
        + balance_weight * bal
    )

    logs = {
        "loss": total.item(),
        "recon": rec.item(),
        "consistency": con.item(),
        "tv": tv.item(),
        "entropy": ent.item(),
        "balance": bal.item(),
    }

    return total, logs

import os
from typing import Tuple

import click
import numpy as np
import torch
from datasets import CheXpertLabelledDataset
from models.vae import VAE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from util import get_dataset


def linear_probe_f1(
    *,
    csv_path: str,
    data_root: str,
    vae: VAE | None = None,
    vae_chkpt: str | None = None,
    batch_size: int = 32,
    image_size: int = 224,
    test_split: float = 0.2,
    device: str = "cpu",
) -> float:
    """Return F1 score of a linear probe on VAE latents."""
    if vae is None:
        if vae_chkpt is None:
            raise ValueError("Either a VAE instance or checkpoint path must be provided")
        vae = VAE.load_from_checkpoint(vae_chkpt, input_res=image_size)

    transform = torch.nn.Identity()
    dataset = CheXpertLabelledDataset(csv_path, data_root, transform=transform, norm=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    dev = torch.device(device)
    vae = vae.to(dev)
    vae.eval()

    features = []
    labels = []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(dev)
            mu, logvar = vae.encode(imgs)
            z = vae.reparameterize(mu, logvar)
            features.append(z.cpu())
            labels.append(lbls)

    X = torch.cat(features, dim=0).numpy()
    y = torch.cat(labels, dim=0).numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split, random_state=0
    )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    score = f1_score(y_test, preds, average="micro")
    return score


@click.command()
@click.argument("csv-path")
@click.argument("data-root")
@click.argument("vae-chkpt")
@click.option("--batch-size", default=32)
@click.option("--image-size", default=224)
@click.option("--test-split", default=0.2)
@click.option("--device", default="cpu")
def main(
    csv_path: str,
    data_root: str,
    vae_chkpt: str,
    batch_size: int,
    image_size: int,
    test_split: float,
    device: str,
):
    """Train a simple linear classifier on frozen VAE features."""
    score = linear_probe_f1(
        csv_path=csv_path,
        data_root=data_root,
        vae_chkpt=vae_chkpt,
        batch_size=batch_size,
        image_size=image_size,
        test_split=test_split,
        device=device,
    )
    print(f"F1 score: {score:.4f}")


if __name__ == "__main__":
    main()

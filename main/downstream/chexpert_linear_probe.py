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
    transform = torch.nn.Identity()
    dataset = CheXpertLabelledDataset(
        csv_path, data_root, transform=transform, norm=False
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    dev = torch.device(device)
    vae = VAE.load_from_checkpoint(vae_chkpt, input_res=image_size).to(dev)
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
    print(f"F1 score: {score:.4f}")


if __name__ == "__main__":
    main()

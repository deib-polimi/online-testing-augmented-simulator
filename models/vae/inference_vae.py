import pathlib

import pandas as pd
import torch
import torchmetrics
from torch.utils.data import DataLoader
from data.dataset import ImageDataset
from utils.path_utils import RESULT_DIR, MODEL_DIR
from models.vae.model import VariationalAutoEncoder


def compute_rec_error(path: pathlib.Path, ckpt_path: pathlib.Path):
    batch_size = 16
    device = "cuda:1"

    dataset = ImageDataset(path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        prefetch_factor=2,
        num_workers=16,
    )

    mae = torchmetrics.MeanAbsoluteError()
    mae = mae.to(device)

    vae = VariationalAutoEncoder.load_from_checkpoint(checkpoint_path=ckpt_path,
                                                      map_location=lambda storage, loc: storage)
    vae = vae.to(device)

    df = pd.DataFrame()
    df['path'] = [str(path.absolute()) for path in dataset.image_paths]

    losses = []
    with torch.no_grad():
        for imgs in dataloader:
            imgs = imgs.to(device)
            recs = vae(imgs)

            losses += [mae(rec, img).item() for img, rec in zip(imgs, recs)]

    df['rec_loss'] = losses

    df.to_csv(path.joinpath("vae_rec_9.csv"), index=False)


if __name__ == '__main__':

    ckpt_path = MODEL_DIR.joinpath("vae", "nominal_epoch=0009_val_loss=0.033976.ckpt")

    for directory in RESULT_DIR.joinpath('ip2p').iterdir():
        print(directory)
        compute_rec_error(directory.joinpath("before"), ckpt_path)
        compute_rec_error(directory.joinpath("after"), ckpt_path)

    # path = RESULT_DIR.joinpath('ip2p', 'make_it_cloudy-2_5', 'before')
    # ckpt_path = MODEL_DIR.joinpath("vae", "nominal_epoch=0109_val_loss=0.006922.ckpt")

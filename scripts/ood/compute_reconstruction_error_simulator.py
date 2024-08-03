import gc
import itertools
import pathlib

import numpy as np
import pandas as pd
import pyiqa
import torch
import torchmetrics
import tqdm
from PIL import Image
from tqdm.contrib.concurrent import process_map
from torch.utils.data import DataLoader

from data.dataset import ImageDataset
from models.vae.model import VariationalAutoEncoder
from utils.conf import DEFAULT_DEVICE
from utils.path_utils import get_images_from_folder, MODEL_DIR, RESULT_DIR


def run_on_folder(folder: pathlib.Path):
    # 0. Run configuration
    batch_size = 4
    output_file = folder.joinpath("vae_reconstruction.csv")
    if output_file.exists():
        return
    print(folder)

    folder = folder.joinpath("image")

    dataset = ImageDataset(folder)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        prefetch_factor=4,
        num_workers=4,

    )

    # 1. Define set of metrics
    metrics = {
        # 'vae_0009': MODEL_DIR.joinpath("vae", "nominal_epoch=0009_val_loss=0.033976.ckpt"),
        # 'vae_0099': MODEL_DIR.joinpath("vae", "nominal_epoch=0099_val_loss=0.006314.ckpt"),
        # 'vae_0199': MODEL_DIR.joinpath("vae", "nominal_epoch=0199_val_loss=0.005345.ckpt"),
        # 'vae_0299': MODEL_DIR.joinpath("vae", "nominal_epoch=0299_val_loss=0.002980.ckpt"),
        # 'vae_0399': MODEL_DIR.joinpath("vae", "nominal_epoch=0399_val_loss=0.002944.ckpt"),
        'vae_0499': MODEL_DIR.joinpath("vae", "nominal_epoch=0489_val_loss=0.002585.ckpt"),
    }

    # 2. Compute metrics
    df = pd.DataFrame()
    for metric_name, ckpt_path in metrics.items():
        vae = VariationalAutoEncoder.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            map_location=lambda storage, loc: storage
        ).to(DEFAULT_DEVICE)
        mae = torchmetrics.MeanAbsoluteError().to(DEFAULT_DEVICE)

        losses = []
        for imgs in tqdm.tqdm(dataloader):
            imgs = imgs.to(DEFAULT_DEVICE)
            recs = vae(imgs)
            losses += [mae(rec, img).item() for img, rec in zip(imgs, recs)]
        df[metric_name] = np.array(losses)

        del vae
        gc.collect()

    # 3. Save csv
    df['filename'] = [filepath.name.__str__() for filepath in get_images_from_folder(folder)]
    df.to_csv(output_file, index=False)

def get_result_folders():
    result = []
    folders =  list(RESULT_DIR.joinpath("online", "nominal").iterdir())
    result += [r.joinpath('before') for r in folders]
    result += [r.joinpath('after') for r in folders]
    # Filter out all folders that are currently being generated
    result = sorted([r for r in result if r.joinpath("log.csv").exists()])
    return result


if __name__ == '__main__':
    # Identify all folders
    # folders = get_result_folders()
    folders = get_result_folders()[::-1]

    # Run on parallel on all folders
    torch.multiprocessing.set_start_method('spawn')
    process_map(run_on_folder, folders, max_workers=2)

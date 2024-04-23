import random

import torchvision.utils
from PIL import Image
from lightning.pytorch.callbacks import ModelCheckpoint
import time as t

import lightning as pl
import torch
import torchvision.utils
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset

from augment.gan.cyclegan import CycleGAN
from augment.gan.train_cyclegan import ImagePairDataset
from utils.conf import DEFAULT_DEVICE
from utils.path_utils import PROJECT_DIR

if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')

    start_time = t.time()
    batch_size = 16
    input_shape = (3, 320, 160)
    max_epochs = 50
    accelerator = "gpu" if "cuda" in DEFAULT_DEVICE else "cpu"
    # TODO: modify folder domain a and b
    folder_domain_a = PROJECT_DIR.joinpath('log', 'sunset_cow', 'before')
    folder_domain_b = PROJECT_DIR.joinpath('log', 'sunset_cow', 'after')

    cyclegan = CycleGAN.load_from_checkpoint(PROJECT_DIR.joinpath("rainy/rainy_3_epoch=9.ckpt.ckpt"))
    # cyclegan = torch.compile(cyclegan, mode="max-autotune")

    # Model path
    # TODO: find better names
    model_path = PROJECT_DIR.joinpath(f"rainy")
    output_folder = PROJECT_DIR.joinpath(f"rainy_3")
    output_folder.mkdir(parents=True, exist_ok=True)

    domain_a = [x for x in folder_domain_a.iterdir() if x.suffix == '.jpg']

    dataset = ImagePairDataset(
        domain_a,
        [x for x in folder_domain_b.iterdir() if x.suffix == '.jpg'],
        transform=torchvision.transforms.ToTensor()
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        prefetch_factor=2,
        num_workers=16,
    )

    for i, batch in enumerate(loader):
        a, _ = batch
        a = a.to(DEFAULT_DEVICE)
        b = cyclegan(a)
        torchvision.utils.save_image(b, PROJECT_DIR.joinpath("log", "rainy_coww_9", "after", domain_a[i].name))

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
from augment.gan.cyclegan_v2 import CycleGANV2
from utils.conf import DEFAULT_DEVICE
from utils.path_utils import PROJECT_DIR


# TODO: move somewhere else
class ImagePairDataset(Dataset):
    def __init__(self, image_paths_a, image_paths_b, transform=None):
        self.image_paths_a = image_paths_a
        self.image_paths_b = image_paths_b
        self.transform = transform

    def __getitem__(self, index):
        image_paths_a = self.image_paths_a[index]
        if random.random() > 0.5:
            index = int(random.random() * len(self))
        a = Image.open(image_paths_a)
        image_paths_b = self.image_paths_b[index]
        b = Image.open(image_paths_b)
        if self.transform is not None:
            a = self.transform(a)
            b = self.transform(b)
        return a, b

    def __len__(self):
        return len(self.image_paths_a)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    start_time = t.time()
    batch_size = 4
    input_shape = (3, 320, 160)
    max_epochs = 100
    accelerator = "gpu" if "cuda" in DEFAULT_DEVICE else "cpu"
    # TODO: modify folder domain a and b
    folder_domain_a = PROJECT_DIR.joinpath('log', 'rainy', 'before')
    folder_domain_b = PROJECT_DIR.joinpath('log', 'rainy', 'after')

    cyclegan = CycleGANV2()
    # cyclegan = torch.compile(cyclegan, mode="max-autotune")

    # Model path
    # TODO: find better names
    model_path = PROJECT_DIR.joinpath(f"rainy.ckpt")
    output_folder = PROJECT_DIR.joinpath(f"rainy2")
    output_folder.mkdir(parents=True, exist_ok=True)

    dataset = ImagePairDataset(
        [x for x in folder_domain_a.iterdir() if x.suffix == '.jpg'],
        [x for x in folder_domain_b.iterdir() if x.suffix == '.jpg'],
        transform=torchvision.transforms.ToTensor()
    )
    # dataset = torch.utils.data.ConcatDataset([dataset]*4)

    loader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=batch_size,
        prefetch_factor=2,
        num_workers=16,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_folder,
        filename="rainy2v2_{epoch}.ckpt",
        save_on_train_epoch_end=True,
        every_n_train_steps=10000,
        save_top_k=-1,
        verbose=True,
    )
    wandb_logger = WandbLogger(project="banana_controlnet")
    trainer = pl.Trainer(
        accelerator=accelerator,
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback],
        devices=[1],
        logger=[wandb_logger],
        precision="bf16",
    )

    trainer.fit(
        cyclegan,
        train_dataloaders=loader,
    )
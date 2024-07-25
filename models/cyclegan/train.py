import itertools
import re
from datetime import timedelta

import lightning as pl
import torch
import torchvision.utils
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from data.dataset import ImagePairDataset
from models.cyclegan.cyclegan import CycleGAN
from utils.conf import DEFAULT_DEVICE
from utils.path_utils import PROJECT_DIR, MODEL_DIR, LOG_DIR, RESULT_DIR

if __name__ == '__main__':

    pl.seed_everything(42)

    torch.set_float32_matmul_precision('high')

    # Training settings
    version = "v2"
    max_epochs = 40
    devices = [int(DEFAULT_DEVICE.split(':')[1])]
    for prompt, approach in itertools.product(
        [
         # "A-street-in-greece-photo-taken-from-a-car",
         "A-street-in-spring-season-photo-taken-from-a-car",
         "A-street-in-italy-photo-taken-from-a-car"
         ],
        ["stable_diffusion_inpainting",
         # "stable_diffusion_inpainting_controlnet_refining"
         ]
    ):
        # prompt = "A-street-in-autumn-season-photo-taken-from-a-car"
        # approach = "stable_diffusion_inpainting"
        # approach = "stable_diffusion_inpainting_controlnet_refining"
        model_name = "dave2"
        run_name = f"online/{approach}/{model_name}/{re.sub('[^0-9a-zA-Z]+', '-', prompt)}"
        folder_domain_a = RESULT_DIR.joinpath(run_name).joinpath("before", "image")
        folder_domain_b = RESULT_DIR.joinpath(run_name).joinpath("after", "image")
        model_path = MODEL_DIR.joinpath(f"online/{approach}/{model_name}/{re.sub('[^0-9a-zA-Z]+', '-', prompt)}")
        checkpoint_name = "cyclegan_{version}_{epoch}_{step}"

        # Training process
        if version == "v1":
            cyclegan = CycleGAN()
            batch_size = 16
        elif version == "v2":
            cyclegan = CycleGAN(num_residual_blocks=4, attention=True, gen_channels=128)
            batch_size = 4

        accelerator = "gpu" if "cuda" in DEFAULT_DEVICE else "cpu"
        input_shape = (3, 320, 160)

        dataset = ImagePairDataset(
            [x for x in folder_domain_a.iterdir() if x.suffix == '.jpg'],
            [x for x in folder_domain_b.iterdir() if x.suffix == '.jpg'],
            transform=torchvision.transforms.ToTensor()
        )

        loader = DataLoader(
            dataset,
            shuffle=True,
            batch_size=batch_size,
            prefetch_factor=2,
            num_workers=16,
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath=model_path,
            filename=checkpoint_name,
            save_on_train_epoch_end=True,
            save_last="link",
            train_time_interval=timedelta(hours=1),
            save_top_k=-1,
            verbose=True,
        )

        wandb_logger = WandbLogger(project=f"cyclegan_{version}", dir=LOG_DIR.joinpath(f"cyclegan_{version}"))
        trainer = pl.Trainer(
            accelerator=accelerator,
            max_epochs=max_epochs,
            callbacks=[checkpoint_callback],
            devices=devices,
            logger=[wandb_logger],
            precision="bf16",
        )

        trainer.fit(
            cyclegan,
            train_dataloaders=loader,
        )

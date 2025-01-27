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
    max_epochs = 20
    devices = [int(DEFAULT_DEVICE.split(':')[1])]
    for prompt, approach in itertools.product(
            [
                # "Make-it-dust-storm",
                # "Make-it-night",
                # "Make-it-autumn",
                # "Make-it-summer",
                # "Make-it-afternoon",
                # "Make-it-sunny",
                # "Make-it-winter",
                # "Make-it-desert-area",
                "Make-it-forest-area",
                # "A-street-in-dust-storm-weather-photo-taken-from-a-car",
                # "A-street-during-night-photo-taken-from-a-car",
                # "A-street-in-summer-season-photo-taken-from-a-car",
                # "A-street-during-afternoon-photo-taken-from-a-car",
                # "A-street-in-sunny-weather-photo-taken-from-a-car",
                # "A-street-in-winter-season-photo-taken-from-a-car",
                # "A-street-in-autumn-season-photo-taken-from-a-car",
                # "A-street-in-forest-area-photo-taken-from-a-car",
                # "A-street-in-desert-area-photo-taken-from-a-car",
            ],
            [
                'instructpix2pix',
                # "stable_diffusion_inpainting",
                # "stable_diffusion_inpainting_controlnet_refining"
            ]
    ):
        run_name = f"online/{approach}/{re.sub('[^0-9a-zA-Z]+', '-', prompt)}"
        model_path = MODEL_DIR.joinpath(run_name)
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

        sub_datasets = []
        for model_name in ['dave2', 'chauffeur', 'epoch', 'vit']:
            run_name = f"online/{approach}/{model_name}/{re.sub('[^0-9a-zA-Z]+', '-', prompt)}"
            sub_datasets += [ImagePairDataset(
                [x for x in RESULT_DIR.joinpath(run_name).joinpath("before", "image").glob("*.jpg")],
                [x for x in RESULT_DIR.joinpath(run_name).joinpath("after", "image").glob("*.jpg")],
                transform=torchvision.transforms.ToTensor()
            )]
        dataset = torch.utils.data.ConcatDataset(sub_datasets)

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
            # train_time_interval=timedelta(hours=1),
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
            precision="bf16-mixed",
        )

        trainer.fit(
            cyclegan,
            train_dataloaders=loader,
        )

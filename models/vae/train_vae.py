import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import lightning as pl
from data.dataset import ImageDataset
from utils.path_utils import RESULT_DIR, MODEL_DIR
from models.vae.model import VariationalAutoEncoder

if __name__ == '__main__':
    max_epochs = 500
    batch_size = 16
    accelerator = "gpu"
    devices = [1]

    pl.seed_everything(42)

    dataset = ImageDataset(path=RESULT_DIR.joinpath('ip2p', 'make_it_cloudy-2_5', 'before'))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        prefetch_factor=2,
        num_workers=16,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        prefetch_factor=2,
        num_workers=16,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=MODEL_DIR.joinpath("vae"),
        filename="nominal_{epoch:04d}_{val_loss:.6f}",
        every_n_epochs=10,
        save_top_k=-1,
        verbose=True,
    )
    trainer = pl.Trainer(
        accelerator=accelerator,
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback],
        devices=devices,
        precision="bf16",
    )

    vae = VariationalAutoEncoder()

    trainer.fit(
        vae,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=MODEL_DIR.joinpath("vae", "nominal_epoch=0109_val_loss=0.006922.ckpt")
    )

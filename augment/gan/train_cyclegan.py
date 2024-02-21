import torchvision.utils
from PIL import Image
from lightning.pytorch.callbacks import ModelCheckpoint
import time as t

import lightning as pl
import torch
import torchvision.utils
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset

from augment.gan.cyclegan import CycleGAN
from utils.conf import DEFAULT_DEVICE, PROJECT_DIR


# TODO: move somewhere else
class ImagePairDataset(Dataset):
    def __init__(self, image_paths_a, image_paths_b, transform=None):
        self.image_paths_a = image_paths_a
        self.image_paths_b = image_paths_b
        self.transform = transform

    def __getitem__(self, index):
        image_paths_a = self.image_paths_a[index]
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
    batch_size = 8
    input_shape = (3, 320, 160)
    max_epochs = 100
    accelerator = DEFAULT_DEVICE
    # TODO: modify folder domain a and b
    folder_domain_a = PROJECT_DIR.joinpath('log', 'fluffy_pony', 'before')
    folder_domain_b = PROJECT_DIR.joinpath('log', 'fluffy_pony', 'after')

    cyclegan = CycleGAN()

    # Model path
    # TODO: find better names
    model_path = PROJECT_DIR.joinpath(f"a-b.ckpt")
    output_folder = PROJECT_DIR.joinpath(f"cyclegan-a-b")
    output_folder.mkdir(parents=True, exist_ok=True)

    dataset = ImagePairDataset(
        [x for x in folder_domain_a.iterdir() if x.suffix == '.jpg'],
        [x for x in folder_domain_b.iterdir() if x.suffix == '.jpg'],
        transform=torchvision.transforms.ToTensor()
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        prefetch_factor=4,
        num_workers=32
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=model_path,
        filename=f"a-b.ckpt",
        save_on_train_epoch_end=True,
        every_n_epochs=10,
        verbose=True,
    )

    trainer = pl.Trainer(
        accelerator=accelerator,
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(
        cyclegan,
        train_dataloaders=loader,
    )
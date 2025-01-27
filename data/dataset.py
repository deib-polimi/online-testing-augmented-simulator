import pathlib
import random

import torchvision
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):

    def __init__(self, path: pathlib.Path, transform=torchvision.transforms.ToTensor(), extension: str = 'jpg'):
        self.path = path
        self.transform = transform
        self.image_paths = sorted(list(path.glob(f'*.{extension}')))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image) if self.transform else image



class ImagePairDataset(Dataset):
    def __init__(self, image_paths_a, image_paths_b, transform=None):
        self.image_paths_a = image_paths_a
        self.image_paths_b = image_paths_b
        self.transform = transform

    def __getitem__(self, index):
        image_paths_a = self.image_paths_a[index]
        if random.random() > 0.9:
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
import pathlib
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

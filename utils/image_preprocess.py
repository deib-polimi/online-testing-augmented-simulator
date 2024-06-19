import os
import pathlib
import PIL
import numpy as np
import torchvision


def to_pytorch_tensor(x):
    if isinstance(x, str) or isinstance(x, pathlib.Path) or isinstance(x, os.PathLike):
        return torchvision.transforms.ToTensor()(PIL.Image.open(x))
    elif isinstance(x, PIL.Image.Image) or isinstance(x, np.array):
        return torchvision.transforms.ToTensor()(x)
    else:
        return x.squeeze(0) if len(x.shape) == 4 else x

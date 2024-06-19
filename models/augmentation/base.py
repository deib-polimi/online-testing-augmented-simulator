import PIL.Image
import numpy as np
import torch


class Augment():

    def __init__(self, name, model):
        self.name = name
        self.model = model

    def __call__(self, image: PIL.Image.Image, *args, **kwargs) -> torch.Tensor:
        if 'mask' in kwargs.keys():
            kwargs['mask'] = np.array(kwargs['mask'])[:, :, 2:] != 255
            kwargs['mask'] = kwargs['mask'].astype(np.uint8) * 255
        return self.model(image, *args, **kwargs)
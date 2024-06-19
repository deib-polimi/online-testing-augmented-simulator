import PIL.Image
import torch


class Augment():

    def __init__(self, name, model):
        self.name = name
        self.model = model

    def __call__(self, image: PIL.Image.Image, *args, **kwargs) -> torch.Tensor:
        return self.model(image, *args, **kwargs)
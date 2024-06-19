import torch


class Augment():

    def __init__(self, name, model):
        self.name = name
        self.model = model

    def __call__(self, tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.model(tensor, *args, **kwargs)
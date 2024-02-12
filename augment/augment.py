import torch


class Augmentation:

    def __init__(self, name, *args, **kwargs):
        self.name = name

    def __call__(self, tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError('Not implemented')

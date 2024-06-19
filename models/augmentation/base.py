import torch


class Augment():

    def __init__(self, name, model, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.model = model

    def __call__(self, tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.model(tensor.unsqueeze(0), *args, **kwargs).squeeze(0)
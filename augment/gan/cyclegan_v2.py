from typing import Any, Tuple

import lightning as pl
import torch
import torchinfo
import torchmetrics
from diffusers import UVit2DModel, UNet2DModel, Transformer2DModel

from augment.gan.cyclegan import Discriminator, CycleGAN, GeneratorResNet
from augment.gan.generator import TurboUNet


class CycleGANV2(CycleGAN):

    def __init__(self, learning_rate: float = 1e-4, input_shape: Tuple[int, int, int] = (3, 160, 320), *args: Any,
                 **kwargs: Any):
        super().__init__(learning_rate=learning_rate, input_shape=input_shape, *args, **kwargs)
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        # Component
        self.generator_AB = GeneratorResNet(input_shape=input_shape, num_residual_blocks=8, out_features=128, attention=True)
        self.generator_BA = GeneratorResNet(input_shape=input_shape, num_residual_blocks=8, out_features=128, attention=True)
        self.discriminator_A = Discriminator(input_shape)
        self.discriminator_B = Discriminator(input_shape)

        # Loss
        self.discriminator_loss = torch.nn.BCELoss()
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_cycle = torch.nn.L1Loss()
        self.criterion_identity = torch.nn.L1Loss()


if __name__ == '__main__':
    g = CycleGANV2()
    torchinfo.summary(g, (1, 3, 160, 320), col_names=["input_size", "output_size", "num_params"], depth=5)

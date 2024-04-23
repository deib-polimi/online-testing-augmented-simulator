import torch
import torchinfo
from diffusers import UNet2DModel
from diffusers.models.downsampling import Downsample2D
from diffusers.models.unets.uvit_2d import UVitBlock
from torch import nn

from augment.gan.cyclegan import GeneratorResNet


class TurboUNet(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = UNet2DModel(
            block_out_channels=[
                64,
                96,
                128,
                192,
            ],
        )

    def forward(self, x):
        x = self.model(x, 0)
        return x.sample


if __name__ == '__main__':
    g = TurboUNet()
    # g = GeneratorResNet((3,160,320), 8)
    # g = torch.compile(g, mode="reduce-overhead")
    torchinfo.summary(g, (1, 3, 160, 320), col_names=["input_size", "output_size", "num_params"], depth=5)

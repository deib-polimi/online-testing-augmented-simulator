import itertools
import random
from typing import Any, Tuple
import lightning as pl
import numpy as np
import torch
import torch.nn as nn
from augment.gan.module import AttnBlock


class CycleGAN(pl.LightningModule):
    def __init__(self, learning_rate: float = 1e-4, input_shape: Tuple[int, int, int] = (3, 160, 320)):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.generator_AB = GeneratorResNet(input_shape, 8)
        self.generator_BA = GeneratorResNet(input_shape, 8)
        self.discriminator_A = Discriminator(input_shape)
        self.discriminator_B = Discriminator(input_shape)
        # training
        self.discriminator_loss = torch.nn.BCELoss()
        # self.vgg_loss = torch.nn.L1Loss()
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_cycle = torch.nn.L1Loss()
        self.criterion_identity = torch.nn.L1Loss()
        # self.fake_A_buffer = ReplayBuffer()
        # self.fake_B_buffer = ReplayBuffer()
        # for multiple optimizers
        self.automatic_optimization = False

    def forward(self, x):
        # TODO
        return self.generator_AB(x)

    def training_step(self, batch, batch_idx):
        g_opt, da_opt, db_opt = self.optimizers()

        g_loss = self._generator_training_step(batch, batch_idx)
        g_opt.zero_grad()
        self.manual_backward(g_loss)
        g_opt.step()

        da_loss = self._discriminator_a_training_step(batch, batch_idx)
        da_opt.zero_grad()
        self.manual_backward(da_loss)
        da_opt.step()

        db_loss = self._discriminator_b_training_step(batch, batch_idx)
        db_opt.zero_grad()
        self.manual_backward(db_loss)
        db_opt.step()

    def _generator_training_step(self, batch, batch_idx):
        real_A, real_B = batch[0], batch[1]

        self.valid = torch.tensor(np.ones((real_A.size(0), *self.discriminator_A.output_shape)), requires_grad=True).to(
            real_A)

        # Identity loss
        loss_id_A = self.criterion_identity(self.generator_BA(real_A), real_A)
        loss_id_B = self.criterion_identity(self.generator_AB(real_B), real_B)
        loss_identity = (loss_id_A + loss_id_B) / 2

        # GAN loss
        fake_B = self.generator_AB(real_A)
        loss_GAN_AB = self.criterion_GAN(self.discriminator_B(fake_B), self.valid)
        fake_A = self.generator_BA(real_B)
        loss_GAN_BA = self.criterion_GAN(self.discriminator_A(fake_A), self.valid)
        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

        # Cycle loss
        recov_A = self.generator_BA(fake_B)
        loss_cycle_A = self.criterion_cycle(recov_A, real_A)
        recov_B = self.generator_AB(fake_A)
        loss_cycle_B = self.criterion_cycle(recov_B, real_B)
        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        # Total loss
        loss_G = loss_GAN * 10.0 * loss_cycle + 5.0 * loss_identity

        self.log("train/g/gan", loss_GAN, prog_bar=True)
        self.log("train/g/cycle", loss_cycle, prog_bar=True)
        self.log("train/g/identity", loss_identity, prog_bar=True)

        if (self.global_step % 500) == 0:
            self.logger.log_image(key="imgs", images=[real_A, real_B, fake_A, fake_B],
                                  step=max(0, self.global_step - 1), caption=['A', 'B', 'FA', 'FB'],
                                  file_type=["jpg"] * 4)

        return loss_G

    def _discriminator_b_training_step(self, batch, batch_idx):
        real_A, real_B = batch[0], batch[1]
        valid = torch.tensor(np.ones((real_A.size(0), *self.discriminator_A.output_shape)), requires_grad=True).to(
            real_A)
        fake = torch.tensor(np.zeros((real_A.size(0), *self.discriminator_A.output_shape)), requires_grad=True).to(
            real_A)

        fake_B = self.generator_AB(real_A)

        # Real loss
        loss_real = (self.criterion_GAN(self.discriminator_B(real_B), valid) +
                     self.criterion_GAN(self.discriminator_B(real_A), fake)) / 2
        # Fake loss (on batch of previously generated samples)
        loss_fake = self.criterion_GAN(self.discriminator_B(fake_B), fake)
        # Total loss
        loss_D_B = (loss_real + loss_fake) / 2

        self.log("train/d/b", loss_D_B, prog_bar=True)
        return loss_D_B

    def _discriminator_a_training_step(self, batch, batch_idx):
        real_A, real_B = batch[0], batch[1]
        valid = torch.tensor(np.ones((real_A.size(0), *self.discriminator_A.output_shape)), requires_grad=True).to(
            real_A)
        fake = torch.tensor(np.zeros((real_A.size(0), *self.discriminator_A.output_shape)), requires_grad=True).to(
            real_A)

        fake_A = self.generator_BA(real_B)

        # Real loss
        loss_real = (self.criterion_GAN(self.discriminator_A(real_A), valid) +
                     self.criterion_GAN(self.discriminator_A(real_B), fake)) / 2
        # Fake loss (on batch of previously generated samples)
        loss_fake = self.criterion_GAN(self.discriminator_A(fake_A), fake)
        # Total loss
        loss_D_A = (loss_real + loss_fake) / 2

        self.log("train/d/a", loss_D_A, prog_bar=True)
        return loss_D_A

    def configure_optimizers(self) -> Any:
        return [
            torch.optim.Adam(itertools.chain(self.generator_AB.parameters(), self.generator_BA.parameters()),
                             lr=self.learning_rate),
            torch.optim.Adam(self.discriminator_A.parameters(), lr=self.learning_rate),
            torch.optim.Adam(self.discriminator_B.parameters(), lr=self.learning_rate),
        ]

    # def _save_batch(self, real_a, real_b, fake_a, fake_b, prefix):
    #     images = torch.cat([real_a[:8], real_b[:8], fake_a[:8], fake_b[:8]], 0)
    #     grid = torchvision.utils.make_grid(images, nrow=8)
    #     torchvision.utils.save_image(grid,
    #                                  f"{self.sample_dir}/{prefix}_{self.current_epoch:0>8}_{self.global_step:0>8}.png")


##############################
#           RESNET
##############################


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks, out_features: int = 64, attention: bool = False):
        super(GeneratorResNet, self).__init__()

        channels = input_shape[0]

        # Initial convolution block
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            # nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                # nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        if attention:
            model += [AttnBlock(out_features)]

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        if attention:
            model += [AttnBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                # nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model += [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(out_features, channels, 7),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return (self.model(x) + 1) / 2


##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)


if __name__ == '__main__':
    model = CycleGAN()

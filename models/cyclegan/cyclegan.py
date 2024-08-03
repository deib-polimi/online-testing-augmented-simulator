import itertools
from typing import Any, Tuple

import lightning as pl
import numpy as np
import torch
import torchinfo

from models.cyclegan.module import GeneratorResNet, Discriminator
from utils.conf import DEFAULT_DEVICE
from utils.image_preprocess import to_pytorch_tensor
from utils.path_utils import RESULT_DIR, PROJECT_DIR


class CycleGAN(pl.LightningModule):
    def __init__(self, learning_rate: float = 1e-4, input_shape: Tuple[int, int, int] = (3, 160, 320),
                 num_residual_blocks: int = 2, attention: bool = False, gen_channels: int = 64):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.generator_AB = GeneratorResNet(input_shape, num_residual_blocks=num_residual_blocks,
                                            out_features=gen_channels, attention=attention)
        self.generator_BA = GeneratorResNet(input_shape, num_residual_blocks=num_residual_blocks,
                                            out_features=gen_channels, attention=attention)
        self.discriminator_A = Discriminator(input_shape)
        self.discriminator_B = Discriminator(input_shape)
        self.discriminator_loss = torch.nn.BCELoss()
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_cycle = torch.nn.L1Loss()
        self.criterion_identity = torch.nn.L1Loss()
        self.automatic_optimization = False

    def forward(self, x, *args, **kwargs):
        x = to_pytorch_tensor(x).to(DEFAULT_DEVICE).unsqueeze(0)
        return self.generator_AB(x).squeeze(0)

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
        loss_G = 2 * loss_GAN * 5.0 * loss_cycle + 10.0 * loss_identity

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


if __name__ == '__main__':
    model = CycleGAN()
    torchinfo.summary(model, (1, 3, 160, 320), col_names=["input_size", "output_size", "num_params"], depth=5)

    model = torch.compile(model).to(DEFAULT_DEVICE)
    model(torch.rand((1, 3, 160, 320)).to(DEFAULT_DEVICE))
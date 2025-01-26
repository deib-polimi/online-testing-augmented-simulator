import itertools
from typing import Any, Tuple

import lightning as pl
import numpy as np
import torch
import torchinfo
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity

from models.cyclegan.module import GeneratorResNet, Discriminator, ReplayBuffer
from utils.conf import DEFAULT_DEVICE
from utils.image_preprocess import to_pytorch_tensor
from utils.path_utils import RESULT_DIR, PROJECT_DIR


# def weights_init_normal(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
#     elif classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1:
#         torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
#         torch.nn.init.constant_(m.bias.data, 0.0)

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
        self.lpips = LearnedPerceptualImagePatchSimilarity()
        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()
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
            real_A) - 0.05

        # Identity loss
        fake_BA = self.generator_BA(real_A)
        fake_AB = self.generator_AB(real_B)
        loss_id_A = self.criterion_identity(fake_BA, real_A) + self.lpips(fake_BA, real_A)
        loss_id_B = self.criterion_identity(fake_AB, real_B) + self.lpips(fake_AB, real_B)
        loss_identity = (loss_id_A + loss_id_B) / 2

        # GAN loss
        fake_B = self.generator_AB(real_A)
        loss_GAN_AB = self.criterion_GAN(self.discriminator_B(fake_B), self.valid)
        fake_A = self.generator_BA(real_B)
        loss_GAN_BA = self.criterion_GAN(self.discriminator_A(fake_A), self.valid)
        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

        # Cycle loss
        recov_A = self.generator_BA(fake_B)
        loss_cycle_A = self.criterion_cycle(recov_A, real_A) + self.lpips(recov_A, real_A)
        recov_B = self.generator_AB(fake_A)
        loss_cycle_B = self.criterion_cycle(recov_B, real_B) + self.lpips(recov_B, real_B)
        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        # Total loss
        loss_G = loss_GAN + 5.0 * loss_cycle + 10.0 * loss_identity

        self.log("train/loss_G", loss_G, prog_bar=True)
        self.log("train/loss_GAN", loss_GAN, prog_bar=True)
        self.log("train/loss_cycle", loss_cycle, prog_bar=True)
        self.log("train/loss_identity", loss_identity, prog_bar=True)

        if (self.global_step % 500) == 0:
            self.logger.log_image(key="imgs", images=[real_A, real_B, fake_A, fake_B],
                                  step=max(0, self.global_step - 1), caption=['A', 'B', 'FA', 'FB'],
                                  file_type=["jpg"] * 4)

        return loss_G

    def _discriminator_b_training_step(self, batch, batch_idx):
        real_A, real_B = batch[0], batch[1]
        valid = torch.tensor(np.ones((real_A.size(0), *self.discriminator_A.output_shape)), requires_grad=True).to(
            real_A) - 0.05
        fake = torch.tensor(np.zeros((real_A.size(0), *self.discriminator_A.output_shape)), requires_grad=True).to(
            real_A) + 0.05
        fake_B = self.generator_AB(real_A)
        # fake_B = self.fake_B_buffer.push_and_pop(fake_B)
        # Real loss
        loss_real = self.criterion_GAN(self.discriminator_B(real_B), valid)
        # Fake loss (on batch of previously generated samples)
        loss_fake = self.criterion_GAN(self.discriminator_B(fake_B), fake)
        # Total loss
        loss_D_B = (loss_real + loss_fake) / 2

        self.log("train/d/b", loss_D_B, prog_bar=True)
        return loss_D_B

    def _discriminator_a_training_step(self, batch, batch_idx):
        real_A, real_B = batch[0], batch[1]
        valid = torch.tensor(np.ones((real_A.size(0), *self.discriminator_A.output_shape)), requires_grad=True).to(
            real_A) - 0.05
        fake = torch.tensor(np.zeros((real_A.size(0), *self.discriminator_A.output_shape)), requires_grad=True).to(
            real_A) + 0.05

        fake_A = self.generator_BA(real_B)
        # fake_A = self.fake_A_buffer.push_and_pop(fake_A)
        # Real loss
        loss_real = self.criterion_GAN(self.discriminator_A(real_A), valid)
        # Fake loss (on batch of previously generated samples)
        loss_fake = self.criterion_GAN(self.discriminator_A(fake_A), fake)
        # Total loss
        loss_D_A = (loss_real + loss_fake) / 2

        self.log("train/d/a", loss_D_A, prog_bar=True)
        return loss_D_A


    def on_before_zero_grad(self, optimizer):
        self.clip_gradients(
            optimizer,
            gradient_clip_val=1.0,
            gradient_clip_algorithm="norm"
        )

    def configure_optimizers(self) -> Any:
        optimizers = [
            torch.optim.Adam(itertools.chain(self.generator_AB.parameters(), self.generator_BA.parameters()),
                             lr=self.learning_rate, betas=(0.5, 0.999)),
            torch.optim.Adam(self.discriminator_A.parameters(), lr=self.learning_rate * 0.5, betas=(0.5, 0.999)),
            torch.optim.Adam(self.discriminator_B.parameters(), lr=self.learning_rate * 0.5, betas=(0.5, 0.999)),
        ]
        for opt in optimizers:
            self.configure_gradient_clipping = True
        return optimizers


if __name__ == '__main__':
    model = CycleGAN()
    torchinfo.summary(model, (1, 3, 160, 320), col_names=["input_size", "output_size", "num_params"], depth=5)

    model = torch.compile(model).to(DEFAULT_DEVICE)
    model(torch.rand((1, 3, 160, 320)).to(DEFAULT_DEVICE))
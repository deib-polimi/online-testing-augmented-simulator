from typing import Any

import lightning as pl
import torch
import torchinfo
from diffusers import AutoencoderKL
from lightning.pytorch.utilities.types import STEP_OUTPUT


class VariationalAutoEncoder(pl.LightningModule):

    def __init__(self, learning_rate: float = 3e-4, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.learning_rate = learning_rate

        self.model = AutoencoderKL(
            in_channels=3, out_channels=3, block_out_channels=(32, 32, 64), layers_per_block=3, latent_channels=8,
            norm_num_groups=32, down_block_types=('DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'),
            up_block_types=('UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'),
        )

        self.mse_loss = torch.nn.MSELoss(reduction='mean')

    def forward(self, x: torch.Tensor) -> Any:
        return self.model(x).sample

    def training_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        return self._step(batch, batch_idx, "train", *args, **kwargs)

    def validation_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        return self._step(batch, batch_idx, "val", *args, **kwargs)

    def _step(self, batch, batch_idx, phase, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        img = batch
        posterior = self.model.encode(img).latent_dist
        z = posterior.mode()
        rec = self.model.decode(z).sample

        kl_loss = posterior.kl().mean()
        mse_loss = self.mse_loss(rec, img)
        loss = 0.1 * kl_loss + mse_loss
        self.log(f"{phase}_kl_loss", kl_loss, on_step=phase == "train", on_epoch=phase == "val", prog_bar=True)
        self.log(f"{phase}_mse_loss", mse_loss, on_step=phase == "train", on_epoch=phase == "val", prog_bar=True)
        self.log(f"{phase}_loss", loss, on_step=phase == "train", on_epoch=phase == "val", prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)


if __name__ == '__main__':
    vae = VariationalAutoEncoder()
    torchinfo.summary(vae, (1, 3, 320, 160), col_names=["input_size", "output_size", "num_params"], depth=5)

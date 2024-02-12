import math
from typing import List

import lightning as pl
import torch
import torchmetrics
from torch import Tensor



def get_nn_architecture(model_name: str, input_shape: tuple[int, int, int]):
    match model_name:
        case "nvidia_dave":
            return get_nvidia_dave_model(input_shape)
        case "epoch":
            return get_epoch_model(input_shape)
        case "chauffeur":
            return get_chauffeur_model(input_shape)


def get_first_block(model_name: str) -> int:
    match model_name:
        case "nvidia_dave":
            return 2
        case "epoch":
            return 3
        case "chauffeur":
            return 4


def get_itta_blocks(model_name: str) -> tuple[List[int], List[int], List[int]]:
    match model_name:
        case "nvidia_dave":
            return [2, 6, 10], [2, 8, 32], [24, 48, 64]
        case "epoch":
            return [3, 7, 11], [4, 16, 64], [32, 64, 128]
        case "chauffeur":
            return [4, 8, 12], [2, 4, 8], [16, 20, 40]


def get_classifier_level(model_name: str) -> int:
    match model_name:
        case "nvidia_dave":
            return 11
        case "epoch":
            return 13
        case "chauffeur":
            return 25


def get_nvidia_dave_model(input_shape: tuple[int, int, int]):
    flat_shape = int(64 * math.ceil(input_shape[-2] / 32) * math.ceil(input_shape[-1] / 32))
    model = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=3, out_channels=24, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
        torch.nn.ReLU(),
        torch.nn.Conv2d(in_channels=24, out_channels=36, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
        torch.nn.ReLU(),
        torch.nn.Conv2d(in_channels=36, out_channels=48, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
        torch.nn.ReLU(),
        torch.nn.Conv2d(in_channels=48, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        torch.nn.ReLU(),
        torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        torch.nn.ReLU(),
        torch.nn.Flatten(start_dim=-3, end_dim=-1),
        torch.nn.Linear(in_features=flat_shape, out_features=100),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(in_features=100, out_features=50),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(in_features=50, out_features=10),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(in_features=10, out_features=1)
    )
    return model


def get_epoch_model(input_shape: tuple[int, int, int]):
    flat_shape = int(128 * math.floor(input_shape[-2] / 64) * math.floor(input_shape[-1] / 64))
    model = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        torch.nn.Dropout(p=0.25),
        torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        torch.nn.Dropout(p=0.25),
        torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        torch.nn.Dropout(p=0.5),
        torch.nn.Flatten(start_dim=-3, end_dim=-1),
        torch.nn.Linear(in_features=flat_shape, out_features=1024),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(in_features=1024, out_features=1)
    )
    return model


def get_chauffeur_model(input_shape: tuple[int, int, int]):
    flat_shape = int(128 * math.floor(input_shape[-2] / 64) * math.floor(input_shape[-1] / 64))
    model = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5, 5), padding=(2, 2)),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.1),
        torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        torch.nn.Conv2d(in_channels=16, out_channels=20, kernel_size=(5, 5), padding=(2, 2)),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.1),
        torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        torch.nn.Conv2d(in_channels=20, out_channels=40, kernel_size=(3, 3), padding=(1, 1)),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.1),
        torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        torch.nn.Conv2d(in_channels=40, out_channels=60, kernel_size=(3, 3), padding=(1, 1)),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.1),
        torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        torch.nn.Conv2d(in_channels=60, out_channels=80, kernel_size=(3, 3), padding=(1, 1)),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.1),
        torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        torch.nn.Conv2d(in_channels=80, out_channels=128, kernel_size=(3, 3), padding=(1, 1)),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.1),
        torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        torch.nn.Flatten(start_dim=-3, end_dim=-1),
        torch.nn.Linear(in_features=flat_shape, out_features=1)
    )
    return model


class UdacityDrivingModel(pl.LightningModule):

    def __init__(self,
                 model_name: str,
                 input_shape: tuple[int, int, int] = (3, 160, 320),
                 learning_rate: float = 2e-4,
                 ):
        super().__init__()
        self.learning_rate = learning_rate
        self.input_shape = input_shape
        self.model_name = model_name
        self.example_input_array = torch.zeros(size=self.input_shape)
        self.model = get_nn_architecture(self.model_name, self.input_shape)
        self.loss = torchmetrics.MeanSquaredError()

    def forward(self, x: Tensor):
        return self.model(x)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int = 0):
        img, true = batch
        out = self.forward(x=img)
        loss = self.loss(out, true)
        return loss

    def validation_step(self, batch: Tensor, batch_idx: int, dataloader_idx: int = 0):
        img, true = batch
        pred = self(img)
        loss = self.loss(true, pred)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/rmse", math.sqrt(loss), prog_bar=True)
        return loss

    def test_step(self, batch: Tensor, batch_idx: int, dataloader_idx: int = 0):
        img, true = batch
        pred = self(img)
        loss = self.loss(true, pred)
        self.log("test/loss", loss, prog_bar=True)
        self.log("test/rmse", math.sqrt(loss), prog_bar=True)
        return loss

    def predict_step(self, batch: Tensor, batch_idx: int, dataloader_idx: int = 0):
        img, _ = batch
        pred = self(img)
        return pred

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=self.learning_rate)]
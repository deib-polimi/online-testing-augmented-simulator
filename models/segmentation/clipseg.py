import os
import pathlib
from typing import Union
import PIL
import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from transformers import AutoProcessor, CLIPSegForImageSegmentation
from utils.conf import DEFAULT_DEVICE

colors = [
    (128, 64, 128),  # road
    (0, 0, 142),  # car
    (70, 130, 180),  # sky
    (107, 142, 35),  # tree
    (70, 70, 70),  # building
    (152, 251, 152),  # terrain
    (60, 116, 168),  # water
]


class ClipSeg(nn.Module):

    def __init__(self, input_shape: tuple[int, int, int] = (3, 160, 320),
                 resized_shape: tuple[int, int, int] = (3, 320, 640), *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model_name = "CIDAS/clipseg-rd64-refined"
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.input_shape = input_shape
        self.resized_shape = resized_shape
        self.model = CLIPSegForImageSegmentation.from_pretrained(self.model_name).to(DEFAULT_DEVICE)

    def preprocess(self, x: Union[str, pathlib.Path, os.PathLike, torch.Tensor, PIL.Image.Image]) -> PIL.Image.Image:
        if isinstance(x, str) or isinstance(x, pathlib.Path) or isinstance(x, os.PathLike):
            return Image.open(x).resize((self.resized_shape[1], self.resized_shape[2]))
        elif isinstance(x, torch.Tensor):
            return torchvision.transforms.ToPILImage()(x.squeeze(0) if len(x.shape) else x)
        else:
            return x

    def forward(self, x: Union[str, pathlib.Path, os.PathLike, torch.Tensor, PIL.Image.Image]):

        image = self.preprocess(x)

        texts = ['road', 'car', 'sky', 'tree', 'building', "terrain", 'water']
        inputs = self.processor(text=texts, images=[image] * len(texts), padding=True, return_tensors="pt").to(
            DEFAULT_DEVICE)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

            predicted_segmentation_map = torch.argmax(
                torchvision.transforms.Resize((self.input_shape[1], self.input_shape[2]))(logits), dim=0)
            predicted_segmentation_map = predicted_segmentation_map.cpu().numpy()
            colored_segmentation_map = np.zeros((predicted_segmentation_map.shape[0],
                                                 predicted_segmentation_map.shape[1], 3),
                                                dtype=np.uint8)  # height, width, 3
            for label, color in enumerate(colors[:len(texts)]):
                colored_segmentation_map[predicted_segmentation_map == label, :] = color

            return predicted_segmentation_map.astype(np.uint8), colored_segmentation_map


if __name__ == '__main__':
    x = ClipSeg()
    a = x("../log/snowy_pony/after/frame_00000001708546073078.jpg")

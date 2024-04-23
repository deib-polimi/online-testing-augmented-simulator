import os
import pathlib
from typing import Union

import PIL
import numpy as np
import torch
import torchvision
from PIL import Image
import requests
from torch import nn
from transformers import AutoProcessor, CLIPSegModel, CLIPSegForImageSegmentation
from utils.conf import DEFAULT_DEVICE

# processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
# model = CLIPSegModel.from_pretrained("CIDAS/clipseg-rd64-refined")
#
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
#
# inputs = processor(
#     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True
# )
#
# outputs = model(**inputs)
# logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
# probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model_name = "CIDAS/clipseg-rd64-refined"
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = CLIPSegForImageSegmentation.from_pretrained(self.model_name).to(DEFAULT_DEVICE)

    def preprocess(self, x: Union[str, pathlib.Path, os.PathLike, torch.Tensor, PIL.Image.Image]) -> PIL.Image.Image:
        if isinstance(x, str) or isinstance(x, pathlib.Path) or isinstance(x, os.PathLike):
            # TODO: FIXME: size should be set at initialization
            return Image.open(x).resize((320, 640))
        elif isinstance(x, torch.Tensor):
            return torchvision.transforms.ToPILImage()(x.squeeze(0) if len(x.shape) else x)
        else:
            return x

    # TODO: FIXME: define output types
    def forward(self, x: Union[str, pathlib.Path, os.PathLike, torch.Tensor, PIL.Image.Image]):

        image = self.preprocess(x)

        texts = ['road', 'car', 'sky', 'tree', 'building', "terrain", 'water']
        inputs = self.processor(text=texts, images=[image] * len(texts), padding=True, return_tensors="pt").to(
            DEFAULT_DEVICE)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

            predicted_segmentation_map = torch.argmax(torchvision.transforms.Resize((160, 320))(logits), dim=0)
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
    # b = x("../log/snowy_pony/after/frame_00000001708546073078.jpg")

    # PIL.Image.fromarray(a)

    # [list(l.color) for l in labels]

import numpy as np
import torch
from PIL import Image
import requests
from torch import nn
from transformers import AutoProcessor, CLIPSegModel, CLIPSegForImageSegmentation

from segmentation.segformer import cityscapes_palette
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
    (0, 0, 142),
    (128, 64, 128),
    (70, 130, 180),
    (107, 142, 35),
    (70, 70, 70),
    (152, 251, 152),
    (60, 116, 168),
]


class ClipSeg(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model_name = "CIDAS/clipseg-rd64-refined"
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = CLIPSegForImageSegmentation.from_pretrained(self.model_name).to(DEFAULT_DEVICE)

    def forward(self, image_path):
        image = Image.open(image_path)
        texts = ['car', 'road', 'sky', 'tree', 'building', "terrain", 'water']
        inputs = self.processor(text=texts, images=[image] * len(texts), padding=True, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

            predicted_segmentation_map = torch.argmax(logits, dim=0)
            predicted_segmentation_map = predicted_segmentation_map.cpu().numpy()
            colored_segmentation_map = np.zeros((predicted_segmentation_map.shape[0],
                                                 predicted_segmentation_map.shape[1], 3),
                                                dtype=np.uint8)  # height, width, 3
            for label, color in enumerate(colors[:len(texts)]):
                colored_segmentation_map[predicted_segmentation_map == label, :] = color

            return predicted_segmentation_map.astype(np.uint8), colored_segmentation_map
        # predicted_segmentation_map = predicted_segmentation_map.cpu().numpy()

        # colored_segmentation_map = np.zeros((predicted_segmentation_map.shape[0],
        #                                      predicted_segmentation_map.shape[1], 3),
        #                                     dtype=np.uint8)  # height, width, 3
        #
        # for label, color in enumerate(cityscapes_palette):
        #     colored_segmentation_map[predicted_segmentation_map == label, :] = color
        # # Convert to BGR
        # # colored_segmentation_map = colored_segmentation_map[..., ::-1]
        #
        # return predicted_segmentation_map.astype(np.uint8), colored_segmentation_map


if __name__ == '__main__':
    x = ClipSeg()

    a = x("../log/snowy_pony/after/frame_00000001708546073078.jpg")
    # b = x("../log/snowy_pony/after/frame_00000001708546073078.jpg")

    # PIL.Image.fromarray(a)

    # [list(l.color) for l in labels]

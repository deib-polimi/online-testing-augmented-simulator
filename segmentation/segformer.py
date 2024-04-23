from collections import namedtuple

import PIL.Image
import lightning as pl
import numpy as np
import torch
import torchvision
from torch import nn
from transformers import pipeline, AutoImageProcessor, Mask2FormerForUniversalSegmentation, SegformerImageProcessor

from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation, SegformerConfig
from PIL import Image
from collections import namedtuple

from segmentation.cityscapes import label2color
from utils.conf import DEFAULT_DEVICE



class SegFormer(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model_name = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
        self.processor = SegformerImageProcessor.from_pretrained(self.model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(self.model_name).to(DEFAULT_DEVICE)
        self.config = SegformerConfig.from_pretrained(self.model_name)

    def forward(self, image_path):
        # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        # image = Image.open(requests.get(url, stream=True).raw)
        # inputs = self.feature_extractor(images=image, return_tensors="pt")
        # outputs = self.model(**inputs)
        # logits = outputs.logits

        image = Image.open(image_path)
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(DEFAULT_DEVICE)

        with torch.no_grad():
            outputs = self.model(pixel_values)
            # logits = outputs.logit

        predicted_segmentation_map = \
            self.processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
        predicted_segmentation_map = predicted_segmentation_map.cpu().numpy()

        colored_segmentation_map = np.zeros((predicted_segmentation_map.shape[0],
                                             predicted_segmentation_map.shape[1], 3),
                                            dtype=np.uint8)  # height, width, 3

        for id, label in self.config.id2label.items():
            color = label2color[label]
            colored_segmentation_map[predicted_segmentation_map == id, :] = color
        # Convert to BGR
        # colored_segmentation_map = colored_segmentation_map[..., ::-1]

        return predicted_segmentation_map.astype(np.uint8), colored_segmentation_map


if __name__ == '__main__':
    x = SegFormer()

    a = x("../log/turbosnowy_cow/after/frame_00000001708606884500.jpg")
    # b = x("../log/snowy_pony/after/frame_00000001708546073078.jpg")

    # PIL.Image.fromarray(a)

    # [list(l.color) for l in labels]

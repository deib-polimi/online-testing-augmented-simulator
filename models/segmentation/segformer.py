import numpy as np
import torch
from PIL import Image
from torch import nn
from transformers import SegformerForSemanticSegmentation, SegformerConfig
from transformers import SegformerImageProcessor

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
        image = Image.open(image_path)
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(DEFAULT_DEVICE)

        with torch.no_grad():
            outputs = self.model(pixel_values)

        predicted_segmentation_map = \
            self.processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
        predicted_segmentation_map = predicted_segmentation_map.cpu().numpy()

        colored_segmentation_map = np.zeros((predicted_segmentation_map.shape[0],
                                             predicted_segmentation_map.shape[1], 3),
                                            dtype=np.uint8)  # height, width, 3

        for id, label in self.config.id2label.items():
            color = label2color[label]
            colored_segmentation_map[predicted_segmentation_map == id, :] = color

        return predicted_segmentation_map.astype(np.uint8), colored_segmentation_map


if __name__ == '__main__':
    x = SegFormer()

    a = x("../log/turbosnowy_cow/after/frame_00000001708606884500.jpg")

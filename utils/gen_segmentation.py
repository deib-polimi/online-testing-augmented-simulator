import pathlib

import PIL
import numpy as np
from PIL import Image

from segmentation.segformer import SegFormer


def get_model(segmentation_name):
    match segmentation_name:
        case "segformer_cityscapes":
            return SegFormer()


if __name__ == '__main__':

    folder = pathlib.Path("../log/turborainy_cow")
    segmentation_name = "segformer_cityscapes"
    segmentation_model = get_model(segmentation_name)

    before_folder = folder.joinpath("before")
    after_folder = folder.joinpath("after")
    seg_before_folder = folder.joinpath(f"seg_before/{segmentation_name}")
    seg_after_folder = folder.joinpath(f"seg_after/{segmentation_name}")
    seg_before_folder.mkdir(parents=True, exist_ok=True)
    seg_after_folder.mkdir(parents=True, exist_ok=True)

    before_images = [x for x in sorted(list(before_folder.iterdir())) if x.suffix == '.jpg']
    after_images = [x for x in sorted(list(after_folder.iterdir())) if x.suffix == '.jpg']

    for x in before_images:
        segmentation_filepath = seg_before_folder.joinpath(f"seg_{x.name}").with_suffix('.png')
        colored_filepath = seg_before_folder.joinpath(f"col_{x.name}").with_suffix('.jpg')
        segmentation, colored = segmentation_model(x)
        PIL.Image.fromarray(segmentation).save(segmentation_filepath)
        PIL.Image.fromarray(colored).save(colored_filepath)

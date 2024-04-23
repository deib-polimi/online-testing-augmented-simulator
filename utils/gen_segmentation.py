import pathlib

import PIL
import numpy as np
from PIL import Image

from segmentation.clipseg import ClipSeg
from segmentation.segformer import SegFormer
from utils.path_utils import PROJECT_DIR, RESULT_DIR
from tqdm import tqdm

def get_model(segmentation_name):
    match segmentation_name:
        case "segformer_cityscapes":
            return SegFormer()
        case "clipseg":
            return ClipSeg()


if __name__ == '__main__':

    # folder = pathlib.Path(RESULT_DIR.joinpath("ip2p", "make_it_cloudy-1_5"))
    folder = PROJECT_DIR.joinpath("log/gan_sunset_cow")
    segmentation_name = "clipseg"
    # segmentation_name = "segformer_cityscapes"
    segmentation_model = get_model(segmentation_name)

    before_folder = folder.joinpath("before")
    after_folder = folder.joinpath("after")
    seg_before_folder = folder.joinpath(f"seg_before_2/{segmentation_name}")
    seg_after_folder = folder.joinpath(f"seg_after_2/{segmentation_name}")
    seg_before_folder.mkdir(parents=True, exist_ok=True)
    seg_after_folder.mkdir(parents=True, exist_ok=True)

    before_images = [x for x in sorted(list(before_folder.iterdir())) if x.suffix == '.jpg']
    after_images = [x for x in sorted(list(after_folder.iterdir())) if x.suffix == '.jpg']

    # TODO: add tqdm
    for x in tqdm(before_images):
        segmentation_filepath = seg_before_folder.joinpath(f"seg_{x.name}").with_suffix('.png')
        colored_filepath = seg_before_folder.joinpath(f"col_{x.name}").with_suffix('.jpg')
        segmentation, colored = segmentation_model(x)
        PIL.Image.fromarray(segmentation).save(segmentation_filepath)
        PIL.Image.fromarray(colored).save(colored_filepath)

    for x in tqdm(after_images):
        segmentation_filepath = seg_after_folder.joinpath(f"seg_{x.name}").with_suffix('.png')
        colored_filepath = seg_after_folder.joinpath(f"col_{x.name}").with_suffix('.jpg')
        segmentation, colored = segmentation_model(x)
        PIL.Image.fromarray(segmentation).save(segmentation_filepath)
        PIL.Image.fromarray(colored).save(colored_filepath)

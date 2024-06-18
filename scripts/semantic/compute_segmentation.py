import gc
import pathlib

import PIL.Image
import numpy as np
import pandas as pd
import pyiqa
import torch
import torchmetrics
import torchvision.utils
import tqdm
from PIL import Image
from tqdm.contrib.concurrent import process_map
from torch.utils.data import DataLoader

from data.dataset import ImageDataset
from models.segmentation.clipseg import ClipSeg
from models.segmentation.segformer import SegFormer
from models.vae.model import VariationalAutoEncoder
from utils.conf import DEFAULT_DEVICE
from utils.path_utils import get_images_from_folder, get_result_folders, MODEL_DIR, RESULT_DIR


def run_on_folder(folder: pathlib.Path):

    # 0. Run configuration
    image_paths = [x for x in sorted(list(folder.iterdir())) if x.suffix == '.jpg']

    # 1. Define set of models
    models = {
        'segformer': SegFormer(),
        'clipseg': ClipSeg(),
    }

    # 2. Compute segmentation map
    for model_name, model in models.items():
        output_folder = folder.parent.joinpath(f"segmentation_{model_name}")
        output_folder.mkdir(parents=True, exist_ok=True)
        for image_path in tqdm.tqdm(image_paths):
            # TODO: add preprocessing the segmentation model
            # TODO: keep all images in the PIL.Image format
            # image = PIL.Image.open(image_path)
            segmentation_map, _ = model(image_path)
            # TODO: keep all outputs in PIL format
            segmentation_map = (segmentation_map == 0).astype(np.uint8) * 255
            PIL.Image.fromarray(segmentation_map).convert('L').save(output_folder.joinpath(image_path.name.replace("jpg", "png")))

if __name__ == '__main__':
    run_on_folder(RESULT_DIR.joinpath("test", "before"))

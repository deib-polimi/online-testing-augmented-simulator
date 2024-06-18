import random

import PIL
import numpy as np
import pandas as pd
import torch
import torchmetrics
import torchvision.utils
from PIL import Image
from tqdm import tqdm

from domains.instruction import ALL_INSTRUCTIONS
from domains.prompt import ALL_PROMPTS
from models.augmentation.instructpix2pix import InstructPix2Pix
from models.augmentation.stable_diffusion_inpainting import StableDiffusionInpainting
from models.augmentation.stable_diffusion_inpainting_controlnet_refining import \
    StableDiffusionInpaintingControlnetRefining
from models.augmentation.stable_diffusion_xl_inpainting import StableDiffusionXLInpainting
from models.augmentation.stable_diffusion_xl_inpainting_controlnet_refining import \
    StableDiffusionXLInpaintingControlnetRefining
from models.segmentation.unet_attention import SegmentationUnet
from utils.conf import DEFAULT_DEVICE
from utils.path_utils import DATASET_DIR, MODEL_DIR

if __name__ == '__main__':

    # Configuration settings
    image_directory = DATASET_DIR.joinpath("sampled_udacity_dataset")
    df = pd.read_csv(image_directory.joinpath("log.csv"), index_col=0)

    # Initialization settings
    random.seed(42)
    to_tensor = torchvision.transforms.ToTensor()
    to_pil = torchvision.transforms.ToPILImage()
    mask_model = SegmentationUnet.load_from_checkpoint(MODEL_DIR.joinpath("unet", "epoch_142.ckpt")).to(DEFAULT_DEVICE)
    mask_model.eval()
    miou_metric = torchmetrics.classification.BinaryJaccardIndex().to(DEFAULT_DEVICE)
    with torch.no_grad():

        for approach in [
            "instructpix2pix",
            "stable_diffusion_inpainting",
            "stable_diffusion_inpainting_controlnet_refining",
            # "stable_diffusion_xl_inpainting",
            # "stable_diffusion_xl_inpainting_controlnet_refining",
        ]:
            mious = []
            for img_name, seg_name in tqdm(zip(df["image_filename"], df['segmentation_filename']), desc=approach):
                img = to_tensor(Image.open(image_directory.joinpath(f"{approach}", img_name))).to(DEFAULT_DEVICE)
                pred_mask = mask_model(img.unsqueeze(0)).squeeze(0)
                true_mask = np.array(Image.open(image_directory.joinpath("segmentation", seg_name)))[:, :, 2:] == 255
                true_mask = to_tensor(true_mask).to(torch.long).to(DEFAULT_DEVICE)
                mious.append(miou_metric(pred_mask, true_mask).item())
            df[f'miou_{approach}'] = mious
            df.to_csv(image_directory.joinpath(f"log.csv"))

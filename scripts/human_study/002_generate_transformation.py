import random

import numpy as np
import pandas as pd
import torch
import torchvision.utils
from PIL import Image

from domains.instruction import ALL_INSTRUCTIONS
from domains.prompt import ALL_PROMPTS
from models.augmentation.instructpix2pix import InstructPix2Pix
from models.augmentation.stable_diffusion_inpainting import StableDiffusionInpainting
from models.augmentation.stable_diffusion_inpainting_controlnet_refining import \
    StableDiffusionInpaintingControlnetRefining
from models.augmentation.stable_diffusion_xl_inpainting import StableDiffusionXLInpainting
from models.augmentation.stable_diffusion_xl_inpainting_controlnet_refining import \
    StableDiffusionXLInpaintingControlnetRefining
from utils.path_utils import DATASET_DIR

if __name__ == '__main__':

    # Configuration settings
    image_directory = DATASET_DIR.joinpath("sampled_udacity_dataset")
    df = pd.read_csv(image_directory.joinpath("log.csv"), index_col=0)

    # Initialization settings
    random.seed(42)
    to_tensor = torchvision.transforms.ToTensor()
    with torch.no_grad():

        # # InstructPix2Pix
        # model = InstructPix2Pix(prompt="", guidance=2.5, num_inference_step=50)
        # target_directory = image_directory.joinpath(f"instructpix2pix")
        # target_directory.mkdir(parents=True, exist_ok=True)
        # domains = []
        # for img_name in df["image_filename"]:
        #     model.prompt = random.sample(ALL_INSTRUCTIONS, 1)[0]
        #     domains.append(model.prompt)
        #     img = image_directory.joinpath("image", img_name)
        #     aug_img = model(img)
        #     torchvision.utils.save_image(aug_img, target_directory.joinpath(img_name))
        # df['instructpix2pix'] = domains
        # df.to_csv(image_directory.joinpath("log.csv"))
        #
        # # Stable Diffusion Inpainting
        # model = StableDiffusionInpainting(prompt="", guidance=10, num_inference_step=50)
        # target_directory = image_directory.joinpath(f"stable_diffusion_inpainting")
        # target_directory.mkdir(parents=True, exist_ok=True)
        # domains = []
        # for img_name, seg_name in zip(df["image_filename"], df['segmentation_filename']):
        #     model.prompt = random.sample(ALL_PROMPTS, 1)[0]
        #     domains.append(model.prompt)
        #     img = image_directory.joinpath("image", img_name)
        #     mask = np.array(Image.open(image_directory.joinpath("segmentation", seg_name)))[:, :, 2:] != 255
        #     mask = to_tensor(mask).to(torch.float)
        #     aug_img = model(img, mask)
        #     torchvision.utils.save_image(aug_img, target_directory.joinpath(img_name))
        # df['stable_diffusion_inpainting'] = domains
        # df.to_csv(image_directory.joinpath("log.csv"))
        #
        # # Stable Diffusion Inpainting
        # model = StableDiffusionXLInpainting(prompt="", guidance=10, num_inference_step=50)
        # target_directory = image_directory.joinpath(f"stable_diffusion_xl_inpainting")
        # target_directory.mkdir(parents=True, exist_ok=True)
        # domains = []
        # for img_name, seg_name in zip(df["image_filename"], df['segmentation_filename']):
        #     model.prompt = random.sample(ALL_PROMPTS, 1)[0]
        #     domains.append(model.prompt)
        #     img = image_directory.joinpath("image", img_name)
        #     mask = np.array(Image.open(image_directory.joinpath("segmentation", seg_name)))[:, :, 2:] != 255
        #     mask = to_tensor(mask).to(torch.float)
        #     aug_img = model(img, mask)
        #     torchvision.utils.save_image(aug_img, target_directory.joinpath(img_name))
        # df['stable_diffusion_xl_inpainting'] = domains
        # df.to_csv(image_directory.joinpath("log.csv"))
        #
        # # Stable Diffusion Inpainting with Controlnet Refining
        # model = StableDiffusionInpaintingControlnetRefining(prompt="", guidance=10, num_inference_step=50)
        # target_directory = image_directory.joinpath(f"stable_diffusion_inpainting_controlnet_refining")
        # target_directory.mkdir(parents=True, exist_ok=True)
        # domains = []
        # for img_name, seg_name in zip(df["image_filename"], df['segmentation_filename']):
        #     model.prompt = random.sample(ALL_PROMPTS, 1)[0]
        #     domains.append(model.prompt)
        #     img = image_directory.joinpath("image", img_name)
        #     mask = np.array(Image.open(image_directory.joinpath("segmentation", seg_name)))[:, :, 2:] != 255
        #     mask = to_tensor(mask).to(torch.float)
        #     aug_img = model(img, mask)
        #     torchvision.utils.save_image(aug_img, target_directory.joinpath(img_name))
        # df['stable_diffusion_inpainting_controlnet_refining'] = domains
        # df.to_csv(image_directory.joinpath("log.csv"))

        # Stable Diffusion XL Inpainting with Controlnet Refining
        model = StableDiffusionXLInpaintingControlnetRefining(prompt="", guidance=10, num_inference_step=50)
        target_directory = image_directory.joinpath(f"stable_diffusion_xl_inpainting_controlnet_refining")
        target_directory.mkdir(parents=True, exist_ok=True)
        domains = []
        for img_name, seg_name in zip(df["image_filename"], df['segmentation_filename']):
            model.prompt = random.sample(ALL_PROMPTS, 1)[0]
            domains.append(model.prompt)
            img = image_directory.joinpath("image", img_name)
            mask = np.array(Image.open(image_directory.joinpath("segmentation", seg_name)))[:, :, 2:] != 255
            mask = to_tensor(mask).to(torch.float)
            aug_img = model(img, mask)
            torchvision.utils.save_image(aug_img, target_directory.joinpath(img_name))
        df['stable_diffusion_xl_inpainting_controlnet_refining'] = domains
        df.to_csv(image_directory.joinpath("log.csv"))

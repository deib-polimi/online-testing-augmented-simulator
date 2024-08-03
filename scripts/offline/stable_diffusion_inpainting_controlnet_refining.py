import itertools
import pathlib

import PIL
import numpy as np
import torchvision
import tqdm

from domains.prompt import ALL_PROMPTS
from models.augmentation.stable_diffusion_inpainting import StableDiffusionInpainting
from models.augmentation.stable_diffusion_inpainting_controlnet_refining import \
    StableDiffusionInpaintingControlnetRefining
from utils.path_utils import PROJECT_DIR

if __name__ == '__main__':

    model = StableDiffusionInpaintingControlnetRefining(prompt="", guidance=10)

    for dataset in [
        'udacity_dataset_lake',
        # 'udacity_dataset_lake_8_8_1',
        # 'udacity_dataset_lake_12_8_1',
        # 'udacity_dataset_lake_12_12_1',
    ]:

        dataset_folder = pathlib.Path(f"/home/banana/projects/udacity-gym/{dataset}/"
                                      f"lake_sunny_day")
        output_folder = pathlib.Path(f"/home/banana/projects/udacity-gym/{dataset}/"
                                     f"refining_lake_sunny_day/image")
        image_paths = list(dataset_folder.joinpath("image").glob('**/*.jpg'))
        mask_paths = list(dataset_folder.joinpath("segmentation").glob('**/*.png'))
        output_folder.mkdir(parents=True, exist_ok=True)


        for i, img_msk in enumerate(tqdm.tqdm(zip(image_paths, mask_paths))):
            image_path, mask_path = img_msk
            if output_folder.joinpath(image_path.name).exists():
                continue
            image = PIL.Image.open(image_path)
            mask = PIL.Image.open(image_path)
            mask = np.array(mask)[:, :, 2:] != 255
            mask = mask.astype(np.uint8) * 255
            model.prompt = ALL_PROMPTS[i % len(ALL_PROMPTS)]
            augmented_image = model(image, mask)
            torchvision.utils.save_image(
                augmented_image,
                output_folder.joinpath(image_path.name)
            )

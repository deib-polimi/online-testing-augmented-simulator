import itertools
import pathlib

import PIL
import torchvision
import tqdm

from domains.prompt import ALL_PROMPTS
from models.augmentation.stable_diffusion_inpainting import StableDiffusionInpainting
from utils.path_utils import PROJECT_DIR

if __name__ == '__main__':

    for track, daytime, weather in itertools.product(
            [
                #"lake",
                #"jungle",
                "mountain"
            ],
            ["day", "daynight"],
            ["sunny", "rainy", "snowy", "foggy"],
    ):

        dataset_folder = pathlib.Path(f"/home/banana/projects/udacity-gym/udacity_dataset/"
                                      f"{track}_{weather}_{daytime}/image")
        output_folder = pathlib.Path(f"/home/banana/projects/udacity-gym/udacity_dataset/"
                                     f"inpainting_{track}_{weather}_{daytime}/image")
        image_paths = list(dataset_folder.glob('**/*.jpg'))
        output_folder.mkdir(parents=True, exist_ok=True)

        model = StableDiffusionInpainting(prompt="", guidance=10)
        for i, image_path in enumerate(tqdm.tqdm(image_paths)):
            image = PIL.Image.open(image_path)
            model.prompt = ALL_PROMPTS[i % len(ALL_PROMPTS)]
            augmented_image = model(image)
            torchvision.utils.save_image(
                augmented_image,
                output_folder.joinpath(image_path.name)
            )

import pathlib

import PIL
import numpy as np
import torchvision

from models.augmentation.stable_diffusion_inpainting_controlnet_refining import \
    StableDiffusionInpaintingControlnetRefining

# Define the structuring element (kernel) for dilation
structuring_element = np.ones((5, 5), dtype=bool)
from scipy.ndimage import binary_dilation
from models.augmentation.stable_diffusion_inpainting import StableDiffusionInpainting

if __name__ == '__main__':

    folders = pathlib.Path("/home/banana/projects/InterFuser/data/eval")
    output_folder = pathlib.Path("/home/banana/projects/InterFuser/data/eval")
    output_folder.mkdir(parents=True, exist_ok=True)
    model = StableDiffusionInpaintingControlnetRefining(prompt="", guidance=10, num_inference_step=20,
                                                        input_shape=(3, 360, 480))
    prompts = [
        "A road during dust storm weather",
        "A road in a forest",
        "A road during night",
        "A road during summer",
        "A road during sunny weather",
        "A road during afternoon",
        "A road during autumn",
        "A road during desert",
        "A road during winter",
    ]

    for prompt in prompts:
        model.prompt = prompt
        for run_folder in [
            folders.joinpath("routes_town05_short_11_11_02_04_54"),
        ]:

            image_paths = list(run_folder.joinpath("meta").glob("**/rgb_*.jpg"))
            out = output_folder.joinpath(run_folder.name, "refining", prompt)
            out.mkdir(parents=True, exist_ok=True)

            for image_path in image_paths:

                o = out.joinpath(image_path.name)
                if o.exists():
                    continue

                if "left_" in image_path.name or "right_" in image_path.name:
                    model.prompt = prompt + ", lateral view"
                else:
                    model.prompt = prompt + ", front view"

                img = PIL.Image.open(image_path)
                seg = PIL.Image.open(str(image_path).replace("rgb", "seg").replace(".jpg", ".png"))

                preserved_mask = sum([np.array(seg)[:, :, :1] == x for x in [4, 5, 7, 10, 12]])
                preserved_mask = binary_dilation(
                    preserved_mask.reshape((preserved_mask.shape[0], preserved_mask.shape[1])),
                    structure=structuring_element,
                    iterations=1
                )
                inpainting_mask = 1 - preserved_mask
                inpainting_mask = inpainting_mask.astype(np.uint8) * 255
                augmented_image = model(img, inpainting_mask)
                torchvision.utils.save_image(
                    augmented_image, o
                )

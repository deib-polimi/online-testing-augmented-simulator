import pathlib

import PIL
import numpy as np
import torchvision

from scipy.ndimage import binary_dilation

from models.augmentation.instructpix2pix import InstructPix2Pix

if __name__ == '__main__':

    folders = pathlib.Path("/home/banana/projects/InterFuser/data/eval")
    output_folder = pathlib.Path("/home/banana/projects/InterFuser/data/eval")
    output_folder.mkdir(parents=True, exist_ok=True)
    model = InstructPix2Pix(prompt="", guidance=2.0, input_shape=(3, 360, 480))
    prompts = [
        "Make it during dust storm weather",
        "Make it in a forest",
        "Make it during night",
        "Make it during summer",
        "Make it during sunny weather",
        "Make it during afternoon",
        "Make it during autumn",
        "Make it during desert",
        "Make it during winter",
    ]

    # Define the structuring element (kernel) for dilation
    structuring_element = np.ones((5, 5), dtype=bool)

    for prompt in prompts:
        model.prompt = prompt
        for run_folder in [
            folders.joinpath("routes_town05_short_11_11_02_04_54"),
        ]:

            image_paths = list(run_folder.glob("**/rgb_*.jpg"))
            out = output_folder.joinpath(run_folder.name, "instruct", prompt)
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

                augmented_image = model(img)
                torchvision.utils.save_image(
                    augmented_image, o
                )

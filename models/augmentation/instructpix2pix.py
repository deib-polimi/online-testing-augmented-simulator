import re
import time

import lightning as pl
import torch
import torchvision.transforms
from PIL import Image
from diffusers import UniPCMultistepScheduler, StableDiffusionInstructPix2PixPipeline

from domains.instruction import ALL_INSTRUCTIONS
from utils.conf import DEFAULT_DEVICE
from utils.image_preprocess import to_pytorch_tensor
from utils.path_utils import PROJECT_DIR, RESULT_DIR


class InstructPix2Pix:

    def __init__(self, prompt: str, guidance: float, num_inference_step: int = 30,
                 input_shape: tuple[int, int, int] = (3, 160, 320)):
        self.prompt = prompt
        self.guidance = guidance
        self.num_inference_step = num_inference_step
        self.input_shape = input_shape

        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix",
                                                                           torch_dtype=torch.float16,
                                                                           safety_checker=None)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to(DEFAULT_DEVICE)

    def optimize(self):
        self.pipe.unet = torch.compile(self.pipe.unet, mode="max-autotune", fullgraph=True)
        self.pipe.vae.decode = torch.compile(self.pipe.vae.decode, mode="max-autotune", fullgraph=True)
        self.forward(torch.rand(self.input_shape))

    def forward(self, image):
        # 1. Convert input image to torch tensor
        image = to_pytorch_tensor(image).to(DEFAULT_DEVICE)
        # assert image.shape == self.input_shape, (f"input image shape ({image.shape}) have different size "
        #                                          f"from the expected one ({self.input_shape}).")

        # 2. Augment image
        augmented_image = self.pipe(prompt=self.prompt, image=image, num_images_per_prompt=1,
                                    num_inference_steps=self.num_inference_step, guidance_scale=10,
                                    image_guidance_scale=self.guidance, output_type='pt').images

        # 3.
        return augmented_image

    def __call__(self, x , *args, **kwargs):
        return to_pytorch_tensor(self.forward(x).detach().float().cpu())


if __name__ == '__main__':

    # 0. Generation settings
    n_runs = 10
    # base_folder = RESULT_DIR.joinpath("investigation", "offline", "instructpix2pix_mid_guidance")
    base_folder = RESULT_DIR.joinpath("investigation", "offline", "instructpix2pix_30_guidance")
    base_folder.mkdir(parents=True, exist_ok=True)
    pl.seed_everything(42)

    # 1. Read input image
    image = Image.open(PROJECT_DIR.joinpath("log/snowy_pony/before/frame_00000001708015939492.jpg"))

    # 2. Compile model to speedup generation
    model = InstructPix2Pix(prompt="", guidance=3.0)
    start_compile_time = time.time()
    # model.optimize()
    result = model(image)
    end_compile_time = time.time()
    print(f"compile time: {end_compile_time - start_compile_time} seconds.")

    # 3. Generating images
    images = []
    # for prompt in ALL_INSTRUCTIONS:
    for prompt in ['change time to night']:
        model.prompt = prompt
        inference_times = []
        for i in range(n_runs):
            start_inference_time = time.time()
            result = model(image)
            end_inference_time = time.time()
            inference_times.append(end_inference_time - start_inference_time)
            torchvision.utils.save_image(
                result, base_folder.joinpath(f"{re.sub('[^0-9a-zA-Z]+', '-', prompt)}_{i}.jpg")
            )
            images.append(result)
        print(f"inference time: {sum(inference_times) / n_runs} seconds.")
    torchvision.utils.save_image(
        torchvision.utils.make_grid(torch.concatenate(images), nrow=n_runs),
        base_folder.joinpath("overall.jpg")
    )

# InstructPix2Pix
import time

import torch
import PIL
import requests
import torch
import torchvision
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from diffusers.utils import load_image

from utils.conf import DEFAULT_DEVICE
from utils.path_utils import PROJECT_DIR


class InstructPix2Pix:

    def __init__(self, prompt: str, guidance: float, num_inference_step: int = 30):
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix",
                                                                           torch_dtype=torch.float16,
                                                                           safety_checker=None)
        # TODO:create default device
        self.pipe.to(DEFAULT_DEVICE)
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
        self.prompt = prompt
        self.guidance = guidance
        self.num_inference_step = num_inference_step

    def forward(self, x):
        return self.pipe(prompt=self.prompt, image=x, num_images_per_prompt=1,
                         num_inference_steps=self.num_inference_step, guidance_scale=10,
                         image_guidance_scale=self.guidance, output_type='pt').images

    def __call__(self, x):
        return self.forward(x)

if __name__ == '__main__':
    image = Image.open(PROJECT_DIR.joinpath("log/snowy_pony/before/frame_00000001708015939492.jpg"))
    model = InstructPix2Pix("change driving location to usa", 2.5)

    n_runs = 5
    inference_times = []
    for i in range(n_runs):
        start_inference_time = time.time()
        result = model(image)
        end_inference_time = time.time()
        inference_times.append(end_inference_time - start_inference_time)

        torchvision.utils.save_image(result, f"ip2p_{i}.jpg")
    print(f"inference time: {sum(inference_times) / 20} seconds.")
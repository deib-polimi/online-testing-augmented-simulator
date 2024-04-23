# InstructPix2Pix
import os
import pathlib
import time

import PIL
import numpy as np
import torch
import torchvision.transforms
from PIL import Image
from diffusers import EulerAncestralDiscreteScheduler, \
    StableDiffusionInpaintPipeline, UniPCMultistepScheduler

from segmentation.clipseg import ClipSeg
from utils.conf import DEFAULT_DEVICE
from utils.path_utils import PROJECT_DIR


class SDInpainting:

    def __init__(self, prompt: str, guidance: float, num_inference_step: int = 30):
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained("/media/banana/data/projects/stable-diffusion-webui-docker/data/models/Stable-diffusion/photographyAnd_10.safetensors",
                                                                   torch_dtype=torch.bfloat16)

        self.pipe = self.pipe.to(DEFAULT_DEVICE)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.prompt = prompt
        self.guidance = guidance
        self.num_inference_step = num_inference_step
        self.segmentation_mask_model = ClipSeg()

    # Compile model to speedup its generation
    def optimize(self):
        self.pipe.unet = torch.compile(self.pipe.unet, mode="max-autotune", fullgraph=True)
        self.pipe.vae.decode = torch.compile(self.pipe.vae.decode, mode="max-autotune", fullgraph=True)
        self.forward(torch.rand(3, 160, 320))

    def preprocess(self, x):
        if isinstance(x, str) or isinstance(x, pathlib.Path) or isinstance(x, os.PathLike):
            # TODO: FIXME: size should be set at initialization
            return torchvision.transforms.ToTensor()(Image.open(x))
        elif isinstance(x, PIL.Image.Image):
            return torchvision.transforms.ToTensor()(x)
        else:
            return x.squeeze(0) if len(x.shape) == 4 else x

    def forward(self, x):

        x = self.preprocess(x).to(DEFAULT_DEVICE)

        _, h, w = x.shape
        resize_original = torchvision.transforms.Resize((h, w))
        resize_large = torchvision.transforms.Resize((h * 2, w * 2))

        x = resize_large(x)

        # Generate mask for inpainting
        mask_image = self.segmentation_mask_model(x)[0]
        # Road is assigned class '0', remove everything else
        mask_image = torch.tensor((mask_image != 0).astype(np.uint8)).to(DEFAULT_DEVICE)

        return resize_original(self.pipe(prompt=self.prompt, image=x, mask_image=mask_image, height=h * 2,
                                         width=w * 2, num_inference_steps=self.num_inference_step,
                                         guidance_scale=self.guidance, output_type='pt').images)

    def __call__(self, x):
        return self.forward(x).detach().float().cpu()


if __name__ == '__main__':

    image = Image.open(PROJECT_DIR.joinpath("log/snowy_pony/before/frame_00000001708015939492.jpg"))

    model = SDInpainting("a street in netherlands with foggy weather at night, photo taken from a car", 7.5)

    start_compile_time = time.time()
    # model.optimize()
    result = model(image)
    end_compile_time = time.time()
    print(f"compile time: {end_compile_time - start_compile_time} seconds.")

    n_runs = 6
    inference_times = []
    for i in range(n_runs):
        start_inference_time = time.time()
        result = model(image)
        end_inference_time = time.time()
        inference_times.append(end_inference_time - start_inference_time)

        torchvision.utils.save_image(result, f"sd_inpainting_{i}.jpg")
    print(f"inference time: {sum(inference_times) / 20} seconds.")

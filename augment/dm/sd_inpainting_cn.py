# InstructPix2Pix
import os
import pathlib
import time

import PIL
import numpy as np
import torch
import torchvision.transforms
from PIL import Image
from controlnet_aux import CannyDetector
from diffusers import EulerAncestralDiscreteScheduler, \
    AutoPipelineForInpainting, StableDiffusionImg2ImgPipeline, StableDiffusionControlNetImg2ImgPipeline, ControlNetModel
from einops import rearrange

from segmentation.clipseg import ClipSeg
from utils.conf import DEFAULT_DEVICE
from utils.path_utils import PROJECT_DIR


class SDInpaintingCN:

    def __init__(self, prompt: str, guidance: float, num_inference_step: int = 30, strength=0.3):
        # Inpainting pipeline
        self.pipe_inpainting = AutoPipelineForInpainting.from_pretrained("runwayml/stable-diffusion-inpainting",
                                                                         torch_dtype=torch.bfloat16, variant="fp16")

        self.pipe_inpainting.to(DEFAULT_DEVICE)
        self.pipe_inpainting.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipe_inpainting.scheduler.config)
        # self.pipe_inpainting.unet = torch.compile(self.pipe_inpainting.unet, mode="reduce-overhead", fullgraph=True)
        # SD image-to-image pipeline
        self.pipe_sd = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.bfloat16,
            controlnet=ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.bfloat16),
            variant="fp16",
        )

        self.pipe_sd.to(DEFAULT_DEVICE)
        self.pipe_sd.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipe_inpainting.scheduler.config)
        # self.pipe_sd.unet = torch.compile(self.pipe_sd.unet, mode="reduce-overhead", fullgraph=True)
        self.prompt = prompt
        self.guidance = guidance
        self.strength = strength
        self.num_inference_step = num_inference_step
        self.segmentation_mask_model = ClipSeg()
        self.hint_generator = CannyDetector()

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

        #
        image = self.pipe_inpainting(prompt=self.prompt, image=x, mask_image=mask_image, height=h * 2,
                                     width=w * 2, num_inference_steps=self.num_inference_step,
                                     guidance_scale=self.guidance,
                                     output_type='pt').images

        control_image = self.hint_generator(
            rearrange(x.to("cpu"), 'c h w -> h w c') * 255,
            image_resolution=h * 2,
            detect_resolution=h * 2)
        control_image.convert(mode="L").save("control.png")

        image = self.pipe_sd(prompt=self.prompt, image=image, control_image=control_image, strength=self.strength,
                             height=h * 2, width=w * 2, num_inference_steps=self.num_inference_step,
                             guidance_scale=self.guidance, output_type='pt').images

        return resize_original(image)

    def __call__(self, x):
        return self.forward(x)


if __name__ == '__main__':
    #  TODO: Modify clipseg so that it works with an image (but also with a path)
    image = Image.open(PROJECT_DIR.joinpath("log/snowy_pony/before/frame_00000001708015939492.jpg"))
    # image = torchvision.transforms.ToTensor()(image)
    model = SDInpaintingCN("a street in the usa at night with foggy weather, photo taken from a car", guidance=7.5,
                           num_inference_step=30, strength=0.7)
    n_runs = 6
    inference_times = []
    for i in range(n_runs):
        start_inference_time = time.time()
        result = model(image)
        end_inference_time = time.time()
        inference_times.append(end_inference_time - start_inference_time)

        torchvision.utils.save_image(result, f"sd_inpainting_cnimg2img_{i}.jpg")
    print(f"inference time: {sum(inference_times) / 20} seconds.")

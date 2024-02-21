# InstructPix2Pix
import torch
import PIL
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from diffusers.utils import load_image


class InstructPix2Pix:

    def __init__(self, prompt: str, guidance: float):
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix",
                                                                           torch_dtype=torch.float16,
                                                                           safety_checker=None)
        # TODO:create default device
        self.pipe.to("cuda:1")
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
        self.prompt = prompt
        self.guidance = guidance

    def forward(self, x):
        return self.pipe(prompt=self.prompt, image=x, num_images_per_prompt=1,
                         num_inference_steps=50, guidance_scale=10,
                         image_guidance_scale=self.guidance, output_type='pt').images

    def __call__(self, x):
        return self.forward(x)
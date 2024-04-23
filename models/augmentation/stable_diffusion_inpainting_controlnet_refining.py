import re
import time
import lightning as pl
import numpy as np
import torch
import torchvision.transforms
from PIL import Image
from controlnet_aux import CannyDetector
from diffusers import StableDiffusionInpaintPipeline, UniPCMultistepScheduler, StableDiffusionControlNetImg2ImgPipeline, \
    ControlNetModel
from einops import rearrange

from domains.prompt import ALL_PROMPTS
from models.segmentation.clipseg import ClipSeg
from utils.conf import DEFAULT_DEVICE
from utils.path_utils import PROJECT_DIR, RESULT_DIR
from utils.image_preprocess import to_pytorch_tensor


class StableDiffusionInpaintingControlnetRefining:

    def __init__(self, prompt: str, guidance: float, num_inference_step: int = 30, strength: float = 0.5,
                 input_shape: tuple[int, int, int] = (3, 160, 320)):
        self.prompt = prompt
        self.guidance = guidance
        self.strength = strength
        self.num_inference_step = num_inference_step
        self.input_shape = input_shape

        self.inpainting_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.bfloat16
        )
        self.inpainting_pipe.scheduler = UniPCMultistepScheduler.from_config(self.inpainting_pipe.scheduler.config)
        self.inpainting_pipe = self.inpainting_pipe.to(DEFAULT_DEVICE)

        self.refining_pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.bfloat16,
            controlnet=ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.bfloat16),
        )
        self.refining_pipe.scheduler = UniPCMultistepScheduler.from_config(self.refining_pipe.scheduler.config)
        self.refining_pipe = self.refining_pipe.to(DEFAULT_DEVICE)
        self.hint_generator = CannyDetector()

        # TODO: move segmentation mask outside
        self.segmentation_mask_model = ClipSeg()

    def optimize(self):
        self.inpainting_pipe.unet = torch.compile(self.inpainting_pipe.unet,
                                                  mode="max-autotune",
                                                  fullgraph=True)
        self.inpainting_pipe.vae.decode = torch.compile(self.inpainting_pipe.vae.decode,
                                                        mode="max-autotune",
                                                        fullgraph=True)
        self.refining_pipe.unet = torch.compile(self.refining_pipe.unet,
                                                mode="max-autotune",
                                                fullgraph=True)
        self.refining_pipe.vae.decode = torch.compile(self.refining_pipe.vae.decode,
                                                      mode="max-autotune",
                                                      fullgraph=True)
        self.forward(torch.rand(self.input_shape))

    def forward(self, image):
        # 1. Convert input image to torch tensor
        image = to_pytorch_tensor(image).to(DEFAULT_DEVICE)
        assert image.shape == self.input_shape, (f"input image shape ({image.shape}) have different size "
                                                 f"from the expected one ({self.input_shape}).")

        # 2. Resize image to the right shape for processing
        _, h, w = self.input_shape
        resize_original = torchvision.transforms.Resize((h, w))
        resize_large = torchvision.transforms.Resize((h * 2, w * 2))
        image = resize_large(image)

        # 3. Compute segmentation map
        mask_image = self.segmentation_mask_model(image)[0]
        mask_image = torch.tensor((mask_image != 0).astype(np.uint8)).to(DEFAULT_DEVICE)

        # 4. Augment image
        inpainted_image = self.inpainting_pipe(prompt=self.prompt, image=image, mask_image=mask_image, height=h * 2,
                                               width=w * 2, num_inference_steps=self.num_inference_step,
                                               guidance_scale=self.guidance, output_type='pt').images

        # 5. Refining with Controlnet
        control_image = self.hint_generator(
            rearrange(image.to("cpu"), 'c h w -> h w c') * 255,
            image_resolution=h,
            detect_resolution=h)
        refined_image = self.refining_pipe(prompt=self.prompt, image=inpainted_image, control_image=control_image,
                                           strength=self.strength, height=h * 2, width=w * 2,
                                           num_inference_steps=self.num_inference_step, guidance_scale=self.guidance,
                                           output_type='pt').images

        # 5. Resize to original image and return
        return resize_original(refined_image)

    def __call__(self, x):
        return self.forward(x).detach().float().cpu()


if __name__ == '__main__':

    # 0. Generation settings
    n_runs = 10
    base_folder = RESULT_DIR.joinpath("investigation", "offline", "stable_diffusion_inpainting_controlnet_refining")
    base_folder.mkdir(parents=True, exist_ok=True)
    pl.seed_everything(42)

    # 1. Read input image
    image = Image.open(PROJECT_DIR.joinpath("log/snowy_pony/before/frame_00000001708015939492.jpg"))

    # 2. Compile model to speedup generation
    model = StableDiffusionInpaintingControlnetRefining(prompt="", guidance=10)
    start_compile_time = time.time()
    # model.optimize()
    result = model(image)
    end_compile_time = time.time()
    print(f"compile time: {end_compile_time - start_compile_time} seconds.")

    # 3. Generating images
    images = []
    for prompt in ALL_PROMPTS:
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

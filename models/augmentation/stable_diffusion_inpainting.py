import re
import time
import lightning as pl
import numpy as np
import torch
import torchvision.transforms
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline, UniPCMultistepScheduler
from torchvision.transforms import InterpolationMode

from domains.prompt import ALL_PROMPTS
from models.segmentation.clipseg import ClipSeg
from models.segmentation.unet_attention import SegmentationUnet
from utils.conf import DEFAULT_DEVICE
from utils.path_utils import PROJECT_DIR, RESULT_DIR, MODEL_DIR
from utils.image_preprocess import to_pytorch_tensor


class StableDiffusionInpainting:

    def __init__(self, prompt: str, guidance: float, num_inference_step: int = 30,
                 input_shape: tuple[int, int, int] = (3, 160, 320)):
        self.prompt = prompt
        self.guidance = guidance
        self.num_inference_step = num_inference_step
        self.input_shape = input_shape

        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.bfloat16
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to(DEFAULT_DEVICE)

    def optimize(self):
        self.pipe.unet = torch.compile(self.pipe.unet, mode="max-autotune", fullgraph=True)
        self.pipe.vae.decode = torch.compile(self.pipe.vae.decode, mode="max-autotune", fullgraph=True)

    def forward(self, image, mask):
        # 1. Convert input image to torch tensor
        image = to_pytorch_tensor(image).to(DEFAULT_DEVICE)
        mask = to_pytorch_tensor(mask).to(DEFAULT_DEVICE)
        assert image.shape == self.input_shape, (f"input image shape ({image.shape}) have different size "
                                                 f"from the expected one ({self.input_shape}).")

        # 2. Resize image to the right shape for processing
        _, h, w = self.input_shape
        image = torchvision.transforms.functional.resize(image, (h*2, w*2))
        mask_image = torchvision.transforms.functional.resize(mask, (h*2, w*2))

        # 3. Augment image
        augmented_image = self.pipe(prompt=self.prompt, image=image, mask_image=mask_image,
                                    negative_prompt='low quality, bad quality, blurry, cars',
                                    height=h * 2, width=w * 2, num_inference_steps=self.num_inference_step,
                                    guidance_scale=self.guidance, output_type='pt').images

        # 4. Resize to original image
        augmented_image = torchvision.transforms.functional.resize(augmented_image, (h, w))

        return augmented_image

    def __call__(self, image, mask):
        return to_pytorch_tensor(self.forward(x).detach().float().cpu())



if __name__ == '__main__':

    # 0. Generation settings
    n_runs = 10
    base_folder = RESULT_DIR.joinpath("investigation", "offline", "stable_diffusion_inpainting")
    base_folder.mkdir(parents=True, exist_ok=True)
    pl.seed_everything(42)

    # 1. Read input image
    image = Image.open(PROJECT_DIR.joinpath("log/snowy_pony/before/frame_00000001708015939492.jpg"))

    # 2. Compile model to speedup generation
    with torch.no_grad():
        model = StableDiffusionInpainting(prompt="", guidance=10)
        mask_model = SegmentationUnet.load_from_checkpoint(MODEL_DIR.joinpath("unet", "epoch_142.ckpt")).to(DEFAULT_DEVICE)
        mask = mask_model(to_pytorch_tensor(image).to(DEFAULT_DEVICE).unsqueeze(0)).squeeze(0)
        mask = (mask < 0.5).to(torch.float)

        # 3. Generating images
        images = []
        for prompt in ALL_PROMPTS:
            model.prompt = prompt
            inference_times = []
            for i in range(n_runs):
                start_inference_time = time.time()
                result = model(image, mask)
                end_inference_time = time.time()
                inference_times.append(end_inference_time - start_inference_time)
                torchvision.utils.save_image(
                    result, base_folder.joinpath(f"{re.sub('[^0-9a-zA-Z]+', '-', prompt)}_{i}.jpg")
                )
                images.append(result)
            print(f"inference time: {sum(inference_times) / n_runs} seconds.")
        # 4. Save image
        torchvision.utils.save_image(
            torchvision.utils.make_grid(torch.concatenate(images), nrow=n_runs),
            base_folder.joinpath("overall.jpg")
        )

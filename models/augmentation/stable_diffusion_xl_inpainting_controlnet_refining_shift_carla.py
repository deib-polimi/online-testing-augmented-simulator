import re
import time
import lightning as pl
import numpy as np
import torch
import torchvision.transforms
from PIL import Image
from controlnet_aux import CannyDetector
from diffusers import StableDiffusionInpaintPipeline, UniPCMultistepScheduler, \
    StableDiffusionXLControlNetImg2ImgPipeline, \
    ControlNetModel, StableDiffusionXLInpaintPipeline, StableDiffusionXLControlNetPipeline, AutoencoderKL
from einops import rearrange

from domains.prompt import ALL_PROMPTS
from models.augmentation.stable_diffusion_xl_inpainting import StableDiffusionXLInpainting
from models.segmentation.clipseg import ClipSeg
from models.segmentation.unet_attention import SegmentationUnet
from utils.conf import DEFAULT_DEVICE
from utils.path_utils import PROJECT_DIR, RESULT_DIR, MODEL_DIR, DATASET_DIR
from utils.image_preprocess import to_pytorch_tensor


class StableDiffusionXLInpaintingControlnetRefining:

    def __init__(self, prompt: str, guidance: float, num_inference_step: int = 30, strength: float = 0.5,
                 input_shape: tuple[int, int, int] = (3, 160, 320),
                 checkpoint: str = "stabilityai/stable-diffusion-xl-base-1.0"):
        self.prompt = prompt
        self.guidance = guidance
        self.strength = strength
        self.num_inference_step = num_inference_step
        self.input_shape = input_shape

        self.inpainting_pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            # "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            "stablediffusionapi/NightVision_XL",
            # variant="fp16",
            use_safetensors=False,
            torch_dtype=torch.bfloat16
        )
        self.inpainting_pipe.scheduler = UniPCMultistepScheduler.from_config(self.inpainting_pipe.scheduler.config)
        self.inpainting_pipe = self.inpainting_pipe.to(DEFAULT_DEVICE)

        self.refining_pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            "stablediffusionapi/NightVision_XL",
            # "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.bfloat16,
            use_safetensors=False,
            controlnet=ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0",
                                                       torch_dtype=torch.bfloat16),
        )
        self.refining_pipe.scheduler = UniPCMultistepScheduler.from_config(self.refining_pipe.scheduler.config)
        self.refining_pipe = self.refining_pipe.to(DEFAULT_DEVICE)
        self.hint_generator = CannyDetector()

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

    def forward(self, image, mask):
        # 1. Convert input image to torch tensor
        image = to_pytorch_tensor(image).to(DEFAULT_DEVICE)
        mask = to_pytorch_tensor(mask).to(DEFAULT_DEVICE)
        # assert image.shape == self.input_shape, (f"input image shape ({image.shape}) have different size "
        #                                          f"from the expected one ({self.input_shape}).")

        # 2. Resize image to the right shape for processing
        # _, h, w = self.input_shape
        h, w = 800, 1280
        image = torchvision.transforms.functional.resize(image, (h, w))
        mask_image = torchvision.transforms.functional.resize(mask, (h, w))

        # 3. Augment image
        inpainted_image = self.inpainting_pipe(
            prompt=self.prompt, image=image, mask_image=mask_image,
            negative_prompt='low quality, bad quality, blurry, cars, intersection',
            height=h, width=w, num_inference_steps=self.num_inference_step,
            guidance_scale=self.guidance, output_type='pt'
        ).images

        # 4. Refining with Controlnet
        control_image = self.hint_generator(
            rearrange(image.to("cpu"), 'c h w -> h w c') * 255,
            image_resolution=h,
            detect_resolution=h,
            output_type="pil",
        )

        refined_image = self.refining_pipe(
            prompt=self.prompt, image=inpainted_image, control_image=control_image,
            negative_prompt='low quality, bad quality, blurry',
            strength=self.strength, height=h, width=w,
            num_inference_steps=self.num_inference_step, guidance_scale=self.guidance,
            controlnet_conditioning_scale=1.1,
            output_type='pt'
        ).images

        # 5. Resize to original image
        refined_image = torchvision.transforms.functional.resize(refined_image, (h, w))

        torchvision.transforms.functional.to_pil_image(image).save("img.jpg")
        torchvision.transforms.functional.to_pil_image(mask_image.detach().float().cpu()).save("mask_img.jpg")
        torchvision.transforms.functional.to_pil_image(inpainted_image.squeeze(0).detach().float().cpu()).save("inpainted.jpg")
        torchvision.transforms.functional.to_pil_image(refined_image.squeeze(0).detach().float().cpu()).save("refined.jpg")
        control_image.save("control.jpg")

        return refined_image

    def __call__(self, image, mask):
        return self.forward(image, mask).detach().float().cpu()


if __name__ == '__main__':

    # 0. Generation settings
    n_runs = 10
    base_folder = RESULT_DIR.joinpath("investigation", "offline",
                                      "stable_diffusion_inpainting_xl_controlnet_refining_shift")
    base_folder.mkdir(parents=True, exist_ok=True)
    pl.seed_everything(42)

    from shift_dev import SHIFTDataset
    from shift_dev.types import Keys
    from shift_dev.utils.backend import FileBackend

    dataset = SHIFTDataset(
        data_root=DATASET_DIR.joinpath("shift"),
        split="train",
        keys_to_load=[
            Keys.images,
            Keys.intrinsics,
            Keys.boxes2d,
            # Keys.boxes2d_classes,
            # Keys.boxes2d_track_ids,
            Keys.segmentation_masks,
        ],
        views_to_load=["front"],
        framerate="images",
        shift_type="discrete",
        backend=FileBackend(),  # also supports HDF5Backend(), FileBackend()
        verbose=True,
    )

    image = dataset[0]['front']['images'].squeeze(0) / 255
    seg_mask = dataset[0]['front']['segmentation_masks']
    torchvision.utils.save_image(
        image, base_folder.joinpath(f"original.jpg")
    )
    exit(0)
    preserved_classes = [
        4, # Pedestrian
        7, # Road
        10, # Vehicle
    ]

    preserved_mask = 1 - sum([seg_mask == cl for cl in preserved_classes])
    mask = preserved_mask.to(torch.float)

    # resize = torchvision.transforms.CenterCrop(size=(400, 640))
    # image = resize(image)
    # mask = resize(mask)

    # 1. Read input image
    # image = Image.open(PROJECT_DIR.joinpath("example_carla.jpg"))

    # 2. Compile model to speedup generation
    with torch.no_grad():
        model = StableDiffusionXLInpainting(
            prompt="", guidance=10, num_inference_step=50,
            # strength=0.9,
            # checkpoint="stablediffusionapi/NightVision_XL",
            input_shape=(3, 800, 1280))
        # mask_model: SegmentationUnet = SegmentationUnet.load_from_checkpoint(
        #     MODEL_DIR.joinpath("unet", "epoch_142.ckpt")).to(
        #     DEFAULT_DEVICE)
        # mask = mask_model(to_pytorch_tensor(image.resize((320,160))).to(DEFAULT_DEVICE).unsqueeze(0)).squeeze(0)
        # mask = (mask < 0.5).to(torch.float)

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
                break
            print(f"inference time: {sum(inference_times) / n_runs} seconds.")
        torchvision.utils.save_image(
            torchvision.utils.make_grid(torch.concatenate(images), nrow=n_runs),
            base_folder.joinpath("overall.jpg")
        )


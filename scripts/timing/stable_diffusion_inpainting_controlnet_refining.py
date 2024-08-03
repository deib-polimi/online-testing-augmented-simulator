import time

import lightning as pl
import torch
from PIL import Image
from scipy.stats import sem

from models.augmentation.stable_diffusion_inpainting_controlnet_refining import \
    StableDiffusionInpaintingControlnetRefining
from models.segmentation.unet_attention import SegmentationUnet
from utils.conf import DEFAULT_DEVICE
from utils.image_preprocess import to_pytorch_tensor
from utils.path_utils import PROJECT_DIR, MODEL_DIR

if __name__ == '__main__':

    # 0. Generation settings
    n_runs = 20
    pl.seed_everything(42)

    # 1. Read input image
    image = Image.open(PROJECT_DIR.joinpath("log/snowy_pony/before/frame_00000001708015939492.jpg"))

    # 2. Compile model to speedup generation
    with torch.no_grad():
        model = StableDiffusionInpaintingControlnetRefining(prompt="", guidance=10)
        mask_model = SegmentationUnet.load_from_checkpoint(MODEL_DIR.joinpath("unet", "epoch_142.ckpt")).to(DEFAULT_DEVICE)
        mask = mask_model(to_pytorch_tensor(image).to(DEFAULT_DEVICE).unsqueeze(0)).squeeze(0)
        mask = (mask < 0.5).to(torch.float)

        # 3. Generating images
        for prompt in ['a street in autumn']:
            model.prompt = prompt
            inference_times = []
            for i in range(n_runs):
                start_inference_time = time.time()
                result = model(image, mask)
                end_inference_time = time.time()
                inference_times.append(end_inference_time - start_inference_time)
            print(f"inference time: {sum(inference_times) / n_runs} seconds, {sem(inference_times)}.")

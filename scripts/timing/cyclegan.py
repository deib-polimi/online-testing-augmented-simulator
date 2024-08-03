import time

import lightning as pl
from PIL import Image
from scipy.stats import sem

from models.augmentation.instructpix2pix import InstructPix2Pix
from models.cyclegan.cyclegan import CycleGAN
from utils.conf import DEFAULT_DEVICE
from utils.path_utils import PROJECT_DIR

if __name__ == '__main__':

    # 0. Generation settings
    n_runs = 20
    # base_folder = RESULT_DIR.joinpath("investigation", "offline", "instructpix2pix_mid_guidance")
    pl.seed_everything(42)

    # 1. Read input image
    image = Image.open(PROJECT_DIR.joinpath("log/snowy_pony/before/frame_00000001708015939492.jpg"))

    # 2. Compile model to speedup generation
    model = CycleGAN(num_residual_blocks=4, attention=True, gen_channels=128).to(DEFAULT_DEVICE)
    start_compile_time = time.time()
    # model.optimize()
    result = model(image)
    end_compile_time = time.time()
    print(f"compile time: {end_compile_time - start_compile_time} seconds.")

    # 3. Generating images
    # for prompt in ALL_INSTRUCTIONS:
    inference_times = []
    for i in range(n_runs):
        start_inference_time = time.time()
        result = model(image)
        end_inference_time = time.time()
        inference_times.append(end_inference_time - start_inference_time)
    print(f"inference time: {sum(inference_times) / n_runs} seconds, {sem(inference_times)}.")

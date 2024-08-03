import time

import lightning as pl
from PIL import Image
from scipy.stats import sem

from models.augmentation.instructpix2pix import InstructPix2Pix
from utils.path_utils import PROJECT_DIR

if __name__ == '__main__':

    # 0. Generation settings
    n_runs = 20
    pl.seed_everything(42)

    # 1. Read input image
    image = Image.open(PROJECT_DIR.joinpath("log/snowy_pony/before/frame_00000001708015939492.jpg"))

    # 2. Compile model to speedup generation
    model = InstructPix2Pix(prompt="", guidance=2.5)
    start_compile_time = time.time()
    # model.optimize()
    result = model(image)
    end_compile_time = time.time()
    print(f"compile time: {end_compile_time - start_compile_time} seconds.")

    # 3. Generating images
    # for prompt in ALL_INSTRUCTIONS:
    for prompt in ['change season to autumn']:
        model.prompt = prompt
        inference_times = []
        for i in range(n_runs):
            start_inference_time = time.time()
            result = model(image)
            end_inference_time = time.time()
            inference_times.append(end_inference_time - start_inference_time)
        print(f"inference time: {sum(inference_times) / n_runs} seconds, {sem(inference_times)}.")

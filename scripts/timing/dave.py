import time

import eventlet
from udacity_gym import UdacityObservation

eventlet.monkey_patch()
import lightning as pl
from PIL import Image
from scipy.stats import sem
from udacity_gym.agent import EndToEndLaneKeepingAgent

from models.augmentation.instructpix2pix import InstructPix2Pix
from models.cyclegan.cyclegan import CycleGAN
from utils.conf import DEFAULT_DEVICE
from utils.path_utils import PROJECT_DIR, MODEL_DIR

if __name__ == '__main__':

    # 0. Generation settings
    n_runs = 20
    pl.seed_everything(42)

    # 1. Read input image
    image = Image.open(PROJECT_DIR.joinpath("log/snowy_pony/before/frame_00000001708015939492.jpg"))
    observation = UdacityObservation(
        input_image=image,
        semantic_segmentation=None,
        position=(0.0, 0.0, 0.0),
        steering_angle=0.0,
        throttle=0.0,
        speed=0.0,
        cte=0.0,
        lap=0,
        sector=0,
        next_cte=0.0,
        time=-1
    )
    # 2. Compile model to speedup generation
    model_name = "epoch"
    checkpoint = MODEL_DIR.joinpath(model_name, f"{model_name}.ckpt")
    model = EndToEndLaneKeepingAgent(
        model_name=model_name,
        checkpoint_path=checkpoint,
    )
    start_compile_time = time.time()
    # model.optimize()
    result = model(observation)
    end_compile_time = time.time()
    print(f"compile time: {end_compile_time - start_compile_time} seconds.")

    # 3. Generating images
    inference_times = []
    for i in range(n_runs):
        start_inference_time = time.time()
        result = model(observation)
        end_inference_time = time.time()
        inference_times.append(end_inference_time - start_inference_time)
    print(f"inference time: {sum(inference_times) / n_runs} seconds, {sem(inference_times)}.")

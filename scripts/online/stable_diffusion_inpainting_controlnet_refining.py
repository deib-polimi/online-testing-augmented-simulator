import json
import re
import time

import torch
from tqdm import tqdm

from ads.agent import LaneKeepingAgent, PauseSimulationCallback, ResumeSimulationCallback, LogObservationCallback, \
    TransformObservationCallback
from ads.model import UdacityDrivingModel
from augment.nn_augment import NNAugmentation
from domains.prompt import ALL_PROMPTS
from models.augmentation.stable_diffusion_inpainting_controlnet_refining import StableDiffusionInpaintingControlnetRefining
from udacity.gym import UdacityGym
from udacity.simulator import UdacitySimulator
from utils.conf import DEFAULT_DEVICE
from utils.path_utils import RESULT_DIR
from utils.net_utils import is_port_in_use

# 0. Experiment Configuration
host = "127.0.0.1"
port = 9993
simulator_exe_path = "simulator/udacity.x86_64"
checkpoint = "lake_sunny_day_60_0.ckpt"
n_steps = 2000  # Represents the number of predictions done by the driving agent

while is_port_in_use(port):
    port = port + 1

# 1. Augmentation Model
augmentation_model = StableDiffusionInpaintingControlnetRefining(prompt="", guidance=1.0)
guidance = 10.0

# 2. Start Simulator
simulator = UdacitySimulator(
    sim_exe_path=simulator_exe_path,
    host=host,
    port=port,
)
simulator.start()

# 3. Create Gym
env = UdacityGym(
    simulator=simulator,
    track="lake",
)

# 4. Start Environment
observation, _ = env.reset(track="lake")
while observation.input_image is None or observation.input_image.sum() == 0:
    observation = env.observe()
    time.sleep(1)
    print("Waiting for environment to set up...")
print("Ready to drive!")

# 5. Setup Driving agent
driving_model = UdacityDrivingModel("nvidia_dave", (3, 160, 320))
driving_model.load_state_dict(torch.load(checkpoint, map_location=lambda storage, loc: storage)['state_dict'])


def get_driving_agent(simulator: UdacitySimulator, run_name: str, prompt: str, guidance: float):
    pause_callback = PauseSimulationCallback(simulator=simulator)
    log_before_callback = LogObservationCallback(path=RESULT_DIR.joinpath(f"{run_name}/before"))
    augmentation_model.prompt = prompt
    augmentation_model.guidance = guidance
    augmentation = NNAugmentation(run_name, augmentation_model)
    transform_callback = TransformObservationCallback(augmentation)
    log_after_callback = LogObservationCallback(
        path=RESULT_DIR.joinpath(f"{run_name}/after"), enable_pygame_logging=True
    )
    resume_callback = ResumeSimulationCallback(simulator=simulator)
    agent = LaneKeepingAgent(
        driving_model.model.to(DEFAULT_DEVICE),
        before_action_callbacks=[pause_callback, log_before_callback],
        transform_callbacks=[transform_callback, resume_callback],
        after_action_callbacks=[log_after_callback],
    )
    return agent


# 6. Drive
for prompt in ALL_PROMPTS:

    run_name = f"online/stable_diffusion_inpainting_controlnet_refining/{re.sub('[^0-9a-zA-Z]+', '-', prompt)}"
    if RESULT_DIR.joinpath(run_name).joinpath("after", "log.csv").exists():
        continue

    agent = get_driving_agent(
        simulator=simulator,
        run_name=run_name,
        prompt=prompt,
        guidance=guidance,
    )

    observation, _ = env.reset(track="lake")
    while observation.input_image is None:
        observation = env.observe()
        time.sleep(1)
        print("waiting for environment to set up...")

    for _ in tqdm(range(n_steps)):
        action = agent(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        time.sleep(0.1)

    json.dump(info, open(RESULT_DIR.joinpath(f"{run_name}/info.json"), "w"))
    agent.before_action_callbacks[1].save()
    agent.after_action_callbacks[0].save()

env.close()

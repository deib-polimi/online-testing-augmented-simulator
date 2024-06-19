from udacity_gym import UdacitySimulator, UdacityGym
import json
import pathlib
import re
import time
import torch
from tqdm import tqdm
from udacity_gym.agent import DaveUdacityAgent
from udacity_gym.agent_callback import PauseSimulationCallback, LogObservationCallback, TransformObservationCallback, \
    ResumeSimulationCallback
from domains.instruction import ALL_INSTRUCTIONS
from domains.prompt import ALL_PROMPTS
from models.augmentation.base import Augment
from models.augmentation.stable_diffusion_inpainting import StableDiffusionInpainting
from utils.conf import DEFAULT_DEVICE
from utils.path_utils import RESULT_DIR, MODEL_DIR
from utils.net_utils import is_port_in_use

if __name__ == '__main__':

    # 0. Experiment Configuration
    host = "127.0.0.1"
    port = 9993
    simulator_exe_path = "simulatorv2/udacity.x86_64"
    checkpoint = MODEL_DIR.joinpath("dave2", "dave2-v3.ckpt")
    n_steps = 2000

    while is_port_in_use(port):
        port = port + 1

    track = "lake"
    daytime = "day"
    weather = "sunny"

    # 1. Augmentation Model
    augmentation_model = StableDiffusionInpainting(prompt="", guidance=1.0)
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
    )

    # 4. Start Environment
    observation, _ = env.reset(track=f"{track}", weather=f"{weather}", daytime=f"{daytime}")
    while not observation or not observation.is_ready():
        observation = env.observe()
        time.sleep(1)
        print("Waiting for environment to set up...")
    print("Ready to drive!")

    # 5. Setup driving agent
    def get_driving_agent(simulator: UdacitySimulator, run_name: str, prompt: str, guidance: float):
        pause_callback = PauseSimulationCallback(simulator=simulator)
        log_before_callback = LogObservationCallback(path=RESULT_DIR.joinpath(f"{run_name}", "before"))
        augmentation_model.prompt = prompt
        augmentation_model.guidance = guidance
        augmentation = Augment(run_name, augmentation_model)
        transform_callback = TransformObservationCallback(augmentation)
        log_after_callback = LogObservationCallback(
            path=RESULT_DIR.joinpath(f"{run_name}", "after"), enable_pygame_logging=True
        )
        resume_callback = ResumeSimulationCallback(simulator=simulator)
        agent = DaveUdacityAgent(
            checkpoint_path=checkpoint,
            before_action_callbacks=[pause_callback, log_before_callback],
            transform_callbacks=[transform_callback],
            after_action_callbacks=[log_after_callback, resume_callback],
        )
        return agent


    # 6. Drive
    for prompt in ALL_PROMPTS:

        run_name = f"online/stable_diffusion_inpainting/{re.sub('[^0-9a-zA-Z]+', '-', prompt)}"
        if RESULT_DIR.joinpath(run_name).joinpath("after", "log.csv").exists():
            continue

        agent = get_driving_agent(
            simulator=simulator,
            run_name=run_name,
            prompt=prompt,
            guidance=guidance,
        )
        agent.model = agent.model.to(DEFAULT_DEVICE)

        observation, _ = env.reset(track=f"{track}", weather=f"{weather}", daytime=f"{daytime}")
        while observation.input_image is None:
            observation = env.observe()
            time.sleep(1)
            print("waiting for environment to set up...")

        for _ in tqdm(range(n_steps)):
            action = agent(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            time.sleep(0.1)

        json.dump(info, open(RESULT_DIR.joinpath(f"{run_name}", "info.json"), "w"))
        agent.before_action_callbacks[1].save()
        agent.after_action_callbacks[0].save()

    env.close()

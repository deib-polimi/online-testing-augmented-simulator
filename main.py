import json
import time

import torch
from tqdm import tqdm

from ads.agent import LaneKeepingAgent, PauseSimulationCallback, ResumeSimulationCallback, LogObservationCallback, \
    TransformObservationCallback
from ads.model import get_nn_architecture, UdacityDrivingModel
from augment.dm.ip2p import InstructPix2Pix
from augment.gan.cyclegan import CycleGAN
from augment.nn_augment import NNAugmentation
from udacity.udacity_controller import UdacitySimController, send_pause, send_resume
from udacity.gym import UdacityGym, UdacityAction
from udacity.simulator import UdacitySimulator
import torchvision.transforms as t

from utils.conf import DEFAULT_DEVICE, RESULT_DIR


# TODO: fix parameter names
def get_ip2p_callbacks(simulator, run_name, prompt, guidance):
    pause_callback = PauseSimulationCallback(simulator=simulator)
    log_before_callback = LogObservationCallback(path=RESULT_DIR.joinpath(f"{run_name}/before"))
    ip2p = InstructPix2Pix(prompt, guidance=guidance)
    augmentation = NNAugmentation(run_name, ip2p)
    transform_callback = TransformObservationCallback(augmentation)
    log_after_callback = LogObservationCallback(path=RESULT_DIR.joinpath(f"{run_name}/after"),
                                                enable_pygame_logging=True)
    resume_callback = ResumeSimulationCallback(simulator=simulator)
    return [pause_callback, log_before_callback], [transform_callback], [log_after_callback, resume_callback]


# This file represents an experimental run

# Take arguments
host = "127.0.0.1"
port = 4567

simulator_exe_path = "simulator/udacity.x86_64"
checkpoint = "lake_sunny_day_60_0.ckpt"

prompt = "make it rainy"
guidance = 1.5
n_steps = 100

# torch.set_default_device(DEFAULT_DEVICE)

# Start Simulator
simulator = UdacitySimulator(
    sim_exe_path=simulator_exe_path,
    host=host,
    port=port,
)
simulator.start()

# Create Gym
env = UdacityGym(
    simulator=simulator,
    track="lake",
)

# Create Agent
model = UdacityDrivingModel("nvidia_dave", (3, 160, 320))
model.load_state_dict(torch.load(checkpoint, map_location=lambda storage, loc: storage)['state_dict'])
# pause_callback = PauseSimulationCallback(simulator=simulator)
# log_before_callback = LogObservationCallback(path=f"log/{run_name}/before")
# TODO: find better name for x
# checkpoint = "cyclegan_foggy.ckpt"
# x = CycleGAN().to(DEFAULT_DEVICE)
# x.load_state_dict(torch.load(checkpoint, map_location=lambda storage, loc: storage)['state_dict'])
# augmentation = NNAugmentation(checkpoint, x)

# #TODO: find better way to add augmentation
# ip2p = InstructPix2Pix("make it rainy", guidance=1.4)
# augmentation = NNAugmentation(checkpoint, ip2p)
# transform_callback = TransformObservationCallback(augmentation)
# log_after_callback = LogObservationCallback(path=f"log/{run_name}/after", enable_pygame_logging=True)
# resume_callback = ResumeSimulationCallback(simulator=simulator)

# before_action_callbacks, transform_callbacks, after_action_callbacks = get_ip2p_callbacks(
#     simulator=simulator,
#     run_name=run_name,
#     prompt=prompt,
#     guidance=guidance,
# )
#
# agent = LaneKeepingAgent(
#     model.model.to(DEFAULT_DEVICE),
#     before_action_callbacks=before_action_callbacks,
#     transform_callbacks=transform_callbacks,
#     after_action_callbacks=after_action_callbacks,
# )

# Observe the environment
# TODO: Remove 'done' and put in 'info'
# TODO: Track is hardcoded
# observation, _ = env.reset(track="lake")
#
# # Wait for environment to set up
# while observation.input_image is None:
#     observation = env.observe()
#     time.sleep(1)
#     print("Waiting for environment to set up...")
# observation, done, info = env.observe()

print("Ready to drive!")
# Drive
# for _ in tqdm(range(200)):
#     with torch.inference_mode():
#         action = agent(observation)
#         observation, reward, terminated, truncated, info = env.step(action)
#         print(info)
#         time.sleep(0.1)

for prompt in ["make it rainy", "make it foggy", "make it cloudy", "make it foggy"]:
    for guidance in [1.5, 2.0, 2.5]:
        run_name = f"ip2p/{prompt.replace(' ', '_')}-{str(guidance).replace('.', '_')}"

        before_action_callbacks, transform_callbacks, after_action_callbacks = get_ip2p_callbacks(
            simulator=simulator,
            run_name=run_name,
            prompt=prompt,
            guidance=guidance,
        )

        agent = LaneKeepingAgent(
            model.model.to(DEFAULT_DEVICE),
            before_action_callbacks=before_action_callbacks,
            transform_callbacks=transform_callbacks,
            after_action_callbacks=after_action_callbacks,
        )

        observation, _ = env.reset(track="lake")

        while observation.input_image is None:
            observation = env.observe()
            time.sleep(1)
            print("waiting for environment to set up...")

        for _ in tqdm(range(n_steps)):
            with torch.inference_mode():
                action = agent(observation)
                observation, reward, terminated, truncated, info = env.step(action)
                time.sleep(0.1)

        # TODO: before calling reset. Save info
        # TODO: track is hardcoded
        json.dump(info, open(RESULT_DIR.joinpath(f"{run_name}/info.json"), "w"))
        # TODO: save automatically from agent. It is the component that automatically know that episode ended
        before_action_callbacks[1].save()
        after_action_callbacks[0].save()
        # TODO: I need to recreate all callbacks

# TODO: Save info after each episode
# TODO: Plot steering angle
# TODO: Generate video
# TODO: Error analysis
# TODO: Timestamp of metric is wrong. Value is wrong
env.close()

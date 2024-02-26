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

from utils.conf import DEFAULT_DEVICE

# This file represents an experimental run

# Take arguments
host = "127.0.0.1"
port = 4567

simulator_exe_path = "simulator/udacity.x86_64"
checkpoint = "lake_sunny_day_60_0.ckpt"
run_name = "turborainy_cow"

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
)

# Create Agent
model = UdacityDrivingModel("nvidia_dave", (3, 160, 320))
model.load_state_dict(torch.load(checkpoint, map_location=lambda storage, loc: storage)['state_dict'])
pause_callback = PauseSimulationCallback(simulator=simulator)
log_before_callback = LogObservationCallback(path=f"log/{run_name}/before")
# TODO: find better name for x
# checkpoint = "cyclegan_foggy.ckpt"
# x = CycleGAN().to(DEFAULT_DEVICE)
# x.load_state_dict(torch.load(checkpoint, map_location=lambda storage, loc: storage)['state_dict'])
# augmentation = NNAugmentation(checkpoint, x)

# #TODO: find better way to add augmentation
ip2p = InstructPix2Pix("make it rainy", guidance=1.4)
augmentation = NNAugmentation(checkpoint, ip2p)
transform_callback = TransformObservationCallback(augmentation)
log_after_callback = LogObservationCallback(path=f"log/{run_name}/after", enable_pygame_logging=True)
resume_callback = ResumeSimulationCallback(simulator=simulator)
agent = LaneKeepingAgent(model.model.to(DEFAULT_DEVICE),
                         before_action_callbacks=[pause_callback, log_before_callback],
                         transform_callbacks=[transform_callback],
                         after_action_callbacks=[log_after_callback, resume_callback],
                         )

# Observe the environment
# TODO: Remove 'done' and put in 'info'
observation, _ = env.reset()

# Wait for environment to set up
while observation.input_image is None:
    observation = env.observe()
    time.sleep(1)
    print("waiting for environment to set up...")
# observation, done, info = env.observe()


print("ready to drive")
# Drive
for _ in tqdm(range(5000)):
    with torch.inference_mode():
        action = agent(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        time.sleep(0.1)

# TODO: Save episode information
# Save all data and close experiment
log_before_callback.save()
log_after_callback.save()
env.close()

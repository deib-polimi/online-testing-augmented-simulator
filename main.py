import time

import torch
from tqdm import tqdm

from ads.agent import LaneKeepingAgent, PauseSimulationCallback, ResumeSimulationCallback, LogObservationCallback, \
    TransformObservationCallback
from ads.model import get_nn_architecture, UdacityDrivingModel
from augment.gan.cyclegan import CycleGAN
from augment.nn_augment import NNAugmentation
from udacity.udacity_controller import UdacitySimController, send_pause, send_resume
from udacity.udacity_gym import UdacityGym, UdacityAction
from udacity.udacity_simulator import UdacitySimulator
import torchvision.transforms as t

# This file represents an experimental run

# Take arguments
host = "127.0.0.1"
port = 4567

# TODO: if path is moved, we get a problem
simulator_exe_path = "simulator/udacity.x86_64"
# simulator_exe_path = "/media/banana/5E32C4AD32C48C09/Users/DAVID/Documents/self-driving-car-sim-new/Builds/udacity.x86_64"
checkpoint = "lake_sunny_day_60_0.ckpt"
run_name = "rainy_pony"
device = "cuda:1"

torch.device(device)

# Start Simulator
simulator = UdacitySimulator(
    simulator_exe_path=simulator_exe_path,
    host=host,
    port=port,
)
simulator.start()

# Start Controller
controller = UdacitySimController(
    port=port,
)
controller.start()

# Create Gym
env = UdacityGym(
    simulator=simulator,
    executor=controller,
)

# Create Agent
model = UdacityDrivingModel("nvidia_dave", (3, 160, 320))
model.load_state_dict(torch.load(checkpoint, map_location=lambda storage, loc: storage)['state_dict'])
pause_callback = PauseSimulationCallback(simulator_controller=controller)
log_before_callback = LogObservationCallback(path=f"log/{run_name}/before")
# TODO: find better name for x
checkpoint = "cyclegan_rainy.ckpt"
x = CycleGAN().to(device)
x.load_state_dict(torch.load(checkpoint, map_location=lambda storage, loc: storage)['state_dict'])
augmentation = NNAugmentation(checkpoint, x)
transform_callback = TransformObservationCallback(augmentation)
log_after_callback = LogObservationCallback(path=f"log/{run_name}/after", enable_pygame_logging=True)
resume_callback = ResumeSimulationCallback(simulator_controller=controller)
agent = LaneKeepingAgent(model.model.to(device),
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
        # action = UdacityAction(steering_angle=0.0, throttle=0.1)
        observation, reward, terminated, truncated, info = env.step(action)
        # if terminated or truncated:
        #     observation, info = env.reset()
        #     break
        time.sleep(0.1)

log_before_callback.save()
log_after_callback.save()

# Close experiment
env.close()

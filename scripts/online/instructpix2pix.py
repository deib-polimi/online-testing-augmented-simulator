import itertools
import json
import re
import time

from tqdm import tqdm
from udacity_gym import UdacitySimulator, UdacityGym
from udacity_gym.agent import EndToEndLaneKeepingAgent
from udacity_gym.agent_callback import PauseSimulationCallback, LogObservationCallback, TransformObservationCallback, \
    ResumeSimulationCallback

from domains.instruction import ALL_INSTRUCTIONS
from models.augmentation.base import Augment
from models.augmentation.instructpix2pix import InstructPix2Pix
from utils.conf import DEFAULT_DEVICE
from utils.net_utils import is_port_in_use
from utils.path_utils import RESULT_DIR, MODEL_DIR

if __name__ == '__main__':

    # 0. Experiment Configuration
    host = "127.0.0.1"
    port = 9993
    simulator_exe_path = "simulatorv2/udacity.x86_64"
    n_steps = 2000

    while is_port_in_use(port):
        port = port + 1

    track = "lake"
    daytime = "day"
    weather = "sunny"

    # 1. Augmentation Model
    augmentation_model = InstructPix2Pix(prompt="", guidance=2.0)
    guidance = 2.0

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
    def get_driving_agent(simulator: UdacitySimulator, run_name: str, model_name: str, prompt: str, guidance: float):
        pause_callback = PauseSimulationCallback(simulator=simulator)
        log_before_callback = LogObservationCallback(path=RESULT_DIR.joinpath(f"{run_name}", "before"))
        augmentation_model.prompt = prompt
        augmentation_model.guidance = guidance
        augmentation = Augment(run_name, augmentation_model)
        checkpoint = MODEL_DIR.joinpath(model_name, f"{model_name}.ckpt")
        transform_callback = TransformObservationCallback(augmentation)
        log_after_callback = LogObservationCallback(
            path=RESULT_DIR.joinpath(f"{run_name}", "after"), enable_pygame_logging=True
        )
        resume_callback = ResumeSimulationCallback(simulator=simulator)
        agent = EndToEndLaneKeepingAgent(
            model_name=model_name,
            checkpoint_path=checkpoint,
            before_action_callbacks=[pause_callback, log_before_callback],
            transform_callbacks=[transform_callback],
            after_action_callbacks=[log_after_callback, resume_callback],
        )
        agent.model.eval()
        return agent


    # 6. Drive
    for prompt, model_name in list(itertools.product(
            ALL_INSTRUCTIONS,
            ['dave2', 'epoch', 'chauffeur', 'vit']
    )):

        run_name = f"online/instructpix2pix/{model_name}/{re.sub('[^0-9a-zA-Z]+', '-', prompt)}"
        if RESULT_DIR.joinpath(run_name).joinpath("after", "log.csv").exists():
            continue

        agent = get_driving_agent(
            simulator=simulator,
            model_name=model_name,
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

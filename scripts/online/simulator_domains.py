import json
import time

from tqdm import tqdm
from udacity_gym import UdacitySimulator, UdacityGym
from udacity_gym.agent import EndToEndLaneKeepingAgent
from udacity_gym.agent_callback import PauseSimulationCallback, LogObservationCallback, ResumeSimulationCallback

from utils.conf import DEFAULT_DEVICE
from utils.net_utils import is_port_in_use
from utils.path_utils import RESULT_DIR, MODEL_DIR

if __name__ == '__main__':

    # 0. Experiment Configuration
    host = "127.0.0.1"
    port = 9992
    simulator_exe_path = "simulatorv2/udacity.x86_64"
    n_steps = 2000

    while is_port_in_use(port):
        port = port + 1

    track = "lake"
    daytime = "day"
    weather = "foggy"

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
    def get_driving_agent(simulator: UdacitySimulator, run_name: str, model_name: str):
        pause_callback = PauseSimulationCallback(simulator=simulator)
        log_before_callback = LogObservationCallback(path=RESULT_DIR.joinpath(f"{run_name}", "before"))
        log_after_callback = LogObservationCallback(
            path=RESULT_DIR.joinpath(f"{run_name}", "after"), enable_pygame_logging=True
        )
        resume_callback = ResumeSimulationCallback(simulator=simulator)
        checkpoint = MODEL_DIR.joinpath(model_name, f"{model_name}.ckpt")
        agent = EndToEndLaneKeepingAgent(
            model_name=model_name,
            checkpoint_path=checkpoint,
            before_action_callbacks=[pause_callback, log_before_callback],
            after_action_callbacks=[log_after_callback, resume_callback],
        )
        agent.model.eval()
        return agent

    # 6. Drive
    for model_name in ['dave2', 'epoch', 'chauffeur', 'vit']:

        run_name = f"online/nominal/{model_name}_{track}_{weather}_{daytime}/"
        if RESULT_DIR.joinpath(run_name).joinpath("after", "log.csv").exists():
            continue

        agent = get_driving_agent(
            simulator=simulator,
            model_name=model_name,
            run_name=run_name,
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

        json.dump(info, open(RESULT_DIR.joinpath(f"{run_name}", "info.json"), "w"))
        agent.before_action_callbacks[1].save()
        agent.after_action_callbacks[0].save()

    env.close()


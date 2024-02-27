import pathlib
from multiprocessing import Manager

import socketio
from flask import Flask

from udacity.action import UdacityAction
from udacity.udacity_controller import UdacitySimController
from udacity.observation import UdacityObservation
from udacity.unity import UnityProcess
from utils.logger import CustomLogger


# TODO: it should extend an abstract simulator
class UdacitySimulator:

    def __init__(
            self,
            sim_exe_path: str = "./examples/udacity/udacity_utils/sim/udacity_sim.app",
            host: str = "127.0.0.1",
            port: int = 4567,
    ):
        # Simulator path
        self.simulator_exe_path = sim_exe_path
        self.sim_process = UnityProcess()
        # Simulator network settings
        from udacity.executor import UdacityExecutor
        self.sim_executor = UdacityExecutor(host, port)
        self.host = host
        self.port = port
        # Simulator logging
        self.logger = CustomLogger(str(self.__class__))
        # Simulator state
        self.sim_state = simulator_state

        # Verify binary location
        if not pathlib.Path(sim_exe_path).exists():
            self.logger.error(f"Executable binary to the simulator does not exists. "
                              f"Check if the path {self.simulator_exe_path} is correct.")

    def step(self, action: UdacityAction):
        self.sim_state['action'] = action

    def observe(self):
        return self.sim_state['observation']

    def pause(self):
        # TODO: change 'pause' with constant
        self.sim_state['paused'] = True

    def resume(self):
        self.sim_state['paused'] = False

    # # TODO: add other track properties
    # def set_track(self, track_name):
    #     self.sim_state['track'] = track_name

    def reset(self, new_track_name: str):
        observation = UdacityObservation(
            input_image=None,
            position=(0.0, 0.0, 0.0),
            steering_angle=0.0,
            throttle=0.0,
            speed=0.0,
            cte=0.0,
            time=-1
        )
        action = UdacityAction(
            steering_angle=0.0,
            throttle=0.0,
        )
        self.sim_state['observation'] = observation
        self.sim_state['action'] = action
        # TODO: Change new track name to enum
        self.sim_state['track'] = new_track_name
        self.sim_state['events'] = []
        self.sim_state['episode_metrics'] = None

        return observation, {}

    def start(self):
        # Start Unity simulation subprocess
        self.logger.info("Starting Unity process for Udacity simulator...")
        self.sim_process.start(
            sim_path=self.simulator_exe_path, headless=False, port=self.port
        )
        self.sim_executor.start()

    def close(self):
        self.sim_process.close()


simulator_state = {
    'observation': None,
    'action': None,
    'paused': False,
    'track': None,
    'events': [],
    'episode_metrics': None,
}

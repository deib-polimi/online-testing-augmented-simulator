# Original author: Roma Sokolkov
# Edited by Antonin Raffin
import os
import time
from typing import Optional, Tuple, Dict, Union, Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from udacity.udacity_action import UdacityAction
from udacity.udacity_observation import UdacityObservation
from utils.logger import CustomLogger


# from examples.udacity.udacity_utils.envs.udacity.config import BASE_PORT, MAX_STEERING, INPUT_DIM
# from examples.udacity.udacity_utils.envs.udacity.core.udacity_sim import UdacitySimController
# from examples.udacity.udacity_utils.envs.unity_proc import UnityProcess
# from examples.udacity.udacity_utils.global_log import GlobalLog

class UdacityGym(gym.Env):
    """
    Gym interface for udacity simulator
    """

    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    def __init__(
            self,
            simulator,
            executor,
            max_steering: float = 1.0,
            max_throttle: float = 1.0,
            input_shape: Tuple[int, int, int] = (3, 160, 320),
    ):
        # Save object properties and parameters
        self.simulator = simulator
        self.executor = executor

        self.max_steering = max_steering
        self.max_throttle = max_throttle
        self.input_shape = input_shape

        self.logger = CustomLogger(str(self.__class__))

        # Initialize the gym environment
        # steering + throttle, action space must be symmetric
        self.action_space = spaces.Box(
            low=np.array([-max_steering, -max_throttle]),
            high=np.array([max_steering, max_throttle]),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=0, high=255, shape=input_shape, dtype=np.uint8
        )

    def step(
            self,
            action: UdacityAction
    ) -> tuple[UdacityObservation, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        :param action: (np.ndarray)
        :return: (np.ndarray, float, bool, dict)
        """
        # action[0] is the steering angle
        # action[1] is the throttle

        self.executor.take_action(action)
        observation = self.observe()

        return observation, observation.cte, self.executor.is_game_over(), self.executor.is_game_over(), {}

    def reset(
            self,
            skip_generation: bool = False,
            track_string: Union[str, None] = None,
    ) -> tuple[UdacityObservation, dict[str, Any]]:

        observation, info = self.executor.reset(
            skip_generation=skip_generation,
            track_string=track_string,
        )

        # TODO: Add track choice

        return observation, info

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        if mode == "rgb_array":
            return self.executor.image_array
        return None

    def observe(self) -> UdacityObservation:
        return self.executor.observe()

    def close(self) -> None:
        if self.simulator is not None:
            self.simulator.quit()
        if self.executor is not None:
            self.executor.quit()

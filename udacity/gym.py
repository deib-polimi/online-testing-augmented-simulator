from typing import Optional, Tuple, Dict, Union, Any, SupportsFloat
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from udacity.action import UdacityAction
from udacity.observation import UdacityObservation
from utils.logger import CustomLogger


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
            track: str,
            max_steering: float = 1.0,
            max_throttle: float = 1.0,
            input_shape: Tuple[int, int, int] = (3, 160, 320),
    ):
        # Save object properties and parameters
        self.simulator = simulator

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

        self.simulator.step(action)
        observation = self.observe()

        # TODO: fix the two Falses
        return observation, observation.cte, False, False, {
            'events': self.simulator.sim_state['events'],
            'episode_metrics': self.simulator.sim_state['episode_metrics'],
        }

    def reset(self, **kwargs) -> tuple[UdacityObservation, dict[str, Any]]:

        observation, info = self.simulator.reset(kwargs['track'])

        return observation, info

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        if mode == "rgb_array":
            return self.simulator.sim_state['observation'].image_array
        return None

    def observe(self) -> UdacityObservation:
        return self.simulator.observe()

    def close(self) -> None:
        if self.simulator is not None:
            self.simulator.close()

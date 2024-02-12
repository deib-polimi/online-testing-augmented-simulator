import copy
from typing import Union

import gymnasium as gym
import numpy as np


class UdacityObservation:

    def __init__(self,
                 input_image: Union[None, np.ndarray[np.uint8]],
                 position: tuple[float, float, float],
                 steering_angle: float,
                 throttle: float,
                 speed: float,
                 n_collisions: int,
                 n_out_of_tracks: int,
                 cte: float,
                 time: int,
                 ):
        self.input_image = input_image
        self.position = position
        self.steering_angle = steering_angle
        self.throttle = throttle
        self.speed = speed
        self.n_collisions = n_collisions
        self.n_out_of_tracks = n_out_of_tracks
        self.cte = cte
        self.time = time

    def get_metrics(self):
        return {
            'pos_x': self.position[0],
            'pos_y': self.position[1],
            'pos_z': self.position[2],
            'steering_angle': self.steering_angle,
            'speed': self.speed,
            'n_collisions': self.n_collisions,
            'n_out_of_tracks': self.n_out_of_tracks,
            'cte': self.cte,
            'time': self.time,
        }

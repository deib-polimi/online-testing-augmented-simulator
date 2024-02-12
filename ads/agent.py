import logging
import pathlib
import time
from typing import Dict, Any, Callable
import pandas as pd
import numpy as np
import pygame
import torch
import torchvision

from udacity.udacity_action import UdacityAction
from udacity.udacity_controller import UdacitySimController
from udacity.udacity_observation import UdacityObservation


class LaneKeepingAgent:

    def __init__(self, model, before_action_callbacks=None, after_action_callbacks=None, transform_callbacks=None):
        self.after_action_callbacks = after_action_callbacks
        self.model = model
        self.before_action_callbacks = before_action_callbacks if before_action_callbacks is not None else []
        self.after_action_callbacks = after_action_callbacks if after_action_callbacks is not None else []
        self.transform_callbacks = transform_callbacks if transform_callbacks is not None else []

    def action(self, observation: UdacityObservation) -> UdacityAction:
        if observation.input_image is None:
            return UdacityAction(steering_angle=0.0, throttle=0.0)
        for callback in self.before_action_callbacks:
            callback(observation)
        for callback in self.transform_callbacks:
            observation = callback(observation)
        # TODO: change to variable or class attribute
        prediction = self.model(torchvision.transforms.ToTensor()(observation.input_image).to("cuda:1"))
        action = UdacityAction(steering_angle=prediction.item() * 1.4, throttle=0.2)
        for callback in self.after_action_callbacks:
            callback(observation, action=action)
        return action

    def __call__(self, observation: UdacityObservation):
        return self.action(observation)


class AgentCallback:

    def __init__(self, name: str, verbose: bool = False):
        self.name = name
        self.verbose = verbose

    def __call__(self, observation: UdacityObservation, *args, **kwargs):
        if self.verbose:
            logging.getLogger(str(self.__class__)).info(f"Activating callback {self.name}")


class PauseSimulationCallback(AgentCallback):

    def __init__(self, simulator_controller: UdacitySimController):
        super().__init__('stop_simulation')
        self.simulator_controller = simulator_controller

    def __call__(self, observation: UdacityObservation, *args, **kwargs):
        super().__call__(observation, *args, **kwargs)
        self.simulator_controller.pause()


class ResumeSimulationCallback(AgentCallback):

    def __init__(self, simulator_controller: UdacitySimController):
        super().__init__('resume_simulation')
        self.simulator_controller = simulator_controller

    def __call__(self, observation: UdacityObservation, *args, **kwargs):
        super().__call__(observation, *args, **kwargs)
        self.simulator_controller.resume()


class LogObservationCallback(AgentCallback):

    def __init__(self, path, enable_pygame_logging=False):
        super().__init__('log_observation')
        self.path = pathlib.Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.logs = []
        self.logging_file = self.path.joinpath('log.csv')
        self.enable_pygame_logging = enable_pygame_logging
        if self.enable_pygame_logging:
            pygame.init()
            self.screen = pygame.display.set_mode((320, 160))
            camera_surface = pygame.surface.Surface((320, 160), 0, 24).convert()
            self.screen.blit(camera_surface, (0, 0))

    def __call__(self, observation: UdacityObservation, *args, **kwargs):
        super().__call__(observation, *args, **kwargs)
        metrics = observation.get_metrics()
        image_name = f"frame_{observation.time:020d}.jpg"
        torchvision.utils.save_image(
            tensor=torchvision.transforms.ToTensor()(observation.input_image),
            fp=self.path.joinpath(image_name)
        )
        metrics['input_image'] = image_name
        if 'action' in kwargs.keys():
            metrics['predicted_steering_angle'] = kwargs['action'].steering_angle
            metrics['predicted_throttle'] = kwargs['action'].throttle
        self.logs.append(metrics)

        if self.enable_pygame_logging:
            pixel_array = np.swapaxes(observation.input_image, 0, 1)
            new_surface = pygame.pixelcopy.make_surface(pixel_array)
            self.screen.blit(new_surface, (0, 0))
            pygame.display.flip()

    def save(self):
        logging_dataframe = pd.DataFrame(self.logs)
        logging_dataframe = logging_dataframe.set_index('time', drop=True)
        logging_dataframe.to_csv(self.logging_file)
        if self.enable_pygame_logging:
            pygame.quit()


class TransformObservationCallback(AgentCallback):

    def __init__(self, transformation: Callable):
        super().__init__('transform_observation')
        self.transformation = transformation

    def __call__(self, observation: UdacityObservation, *args, **kwargs):
        super().__call__(observation, *args, **kwargs)
        # Change with parameter
        augmented_image: torch.Tensor = self.transformation(torchvision.transforms.ToTensor()(observation.input_image)).to("cuda:1")
        observation.input_image = (augmented_image.permute(1, 2, 0).detach().numpy() * 255).astype(np.uint8)
        return observation

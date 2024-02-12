"""
MIT License

Copyright (c) 2018 Roma Sokolkov
Copyright (c) 2018 Antonin Raffin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
# Original author: Tawn Kramer

import base64
import time
from io import BytesIO
from threading import Thread
from typing import Tuple, Dict, List, Union, Any

import numpy as np
import pygame
import socketio
import torch
import torchvision.transforms
from PIL import Image
from flask import Flask
import socketio
import eventlet.wsgi
from flask import Flask
from socketio import Server

# from examples.udacity.udacity_utils.envs.udacity.config import INPUT_DIM, MAX_CTE_ERROR
# from examples.udacity.udacity_utils.envs.udacity.core.client import start_app
# from examples.udacity.udacity_utils.global_log import GlobalLog
import logging

from augment.gan.cyclegan import CycleGAN
from udacity.udacity_gym import UdacityObservation, UdacityAction
from utils.logger import CustomLogger

sio = socketio.Server()
flask_app = Flask(__name__)

logger = CustomLogger("Controller")


# TODO: improve logging -> change print to logger

def start_app(application: Flask, socket_io: Server, port: int):
    app = socketio.Middleware(socket_io, application)
    eventlet.wsgi.server(eventlet.listen(('', port)), app)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)-8s  %(message)s',
    datefmt='(%H:%M:%S)')

# disable all loggers from different files
logging.getLogger('asyncio').setLevel(logging.INFO)
logging.getLogger('asyncio.coroutines').setLevel(logging.INFO)
logging.getLogger('websockets.server').setLevel(logging.INFO)
logging.getLogger('websockets.protocol').setLevel(logging.INFO)

# TODO: collect simulator state in just a variable
# last_observation: Union[None, UdacityObservation] = None

height = 160
width = 320
channels = 3

observation = UdacityObservation(
    input_image=None,
    position=(0.0, 0.0, 0.0),
    steering_angle=0.0,
    throttle=0.0,
    speed=0.0,
    n_collisions=0,
    n_out_of_tracks=0,
    cte=0.0,
    time=-1
)

action = UdacityAction(
    steering_angle=0.0,
    throttle=0.0
)

pause_state = False

# last_obs: np.ndarray[np.uint8] = None
# image_array: np.ndarray[np.uint8] = None
is_connect = False
deployed_track_string = None
generated_track_string = None
# steering = 0.0
# throttle = 0.0
# speed = 0.0
# cte = 0.0
# cte_pid = 0.0
# hit = 0
# oot = 0
done = False
track_sent = False
# pos_x = 0.0
# pos_y = 0.0
# pos_z = 0.0
udacity_unreactive = False


# TODO: display with gymnasium
# TODO: fix global variables
# TODO: change track from python app

@sio.on("connect")
def connect(sid, environ) -> None:
    global is_connect
    is_connect = True
    logger.info("Connect to Udacity simulator: {}".format(sid))
    # print("Connect to Udacity simulator: {}".format(sid))
    send_control(steering_angle=0.0, throttle_command=0.0)


def send_control(steering_angle: float, throttle_command: float) -> None:
    # TODO: can be managed better
    # check only when the state is changed
    # put all simulation state inside an object
    send_pause() if pause_state else send_resume()
    sio.emit(
        "steer",
        data={
            "steering_angle": steering_angle.__str__(),
            "throttle": throttle_command.__str__(),
        },
        skip_sid=True,
    )
    global udacity_unreactive

    if observation.throttle >= 0.01 and round(observation.speed, 1) == 0.0:
        print(f"Warning: Throttle is {observation.throttle} but speed is {observation.speed}\n")
        udacity_unreactive = True
    if observation.speed > 0.0 and udacity_unreactive:
        print("Warning: Udacity is reactivated\n")
        udacity_unreactive = False


def send_track(track_string: str) -> None:
    global track_sent
    if not track_sent:
        sio.emit("track", data={"track_string": track_string}, skip_sid=True)
        track_sent = True
        print("SendTrack", end="\n", flush=True)
    else:
        print("Track already sent", end="\n", flush=True)


def send_reset() -> None:
    sio.emit("reset", data={"nothing": "wow"}, skip_sid=True)
    print("Reset", end="\n", flush=True)


def send_pause() -> None:
    sio.emit("pause", data={"nothing": "wow"}, skip_sid=True)


def send_resume() -> None:
    sio.emit("resume", data={"nothing": "wow"}, skip_sid=True)


# checkpoint = "cyclegan_foggy.ckpt"
# model = CycleGAN()
# model.load_state_dict(torch.load(checkpoint, map_location=lambda storage, loc: storage)['state_dict'])

@sio.on("telemetry")
def telemetry(sid, data) -> None:
    global observation
    global action

    print(data)

    if data:
        observation = UdacityObservation(
            input_image=np.array(Image.open(BytesIO(base64.b64decode(data["image"])))),
            position=(float(data["pos_x"]), float(data["pos_y"]), float(data["pos_z"])),
            steering_angle=action.steering_angle,
            throttle=action.throttle,
            speed=float(data["speed"]) * 3.6,  # conversion m/s to km/h
            n_collisions=int(float(data["hit"])),
            n_out_of_tracks=int(float(data["oot"])),
            cte=float(data["cte"]),
            time=int(time.time() * 1000)
        )

        # if done:
        #     send_reset()
        # elif generated_track_string is not None and not track_sent:
        #     send_track(track_string=generated_track_string)
        #     time.sleep(0.5)
        # else:
        send_pause()
        send_control(steering_angle=action.steering_angle, throttle_command=action.throttle)


    else:
        print("Warning: Udacity data is None")
    if udacity_unreactive:
        print(f"Warning: Udacity Non Reactive, received {data} from sid {sid}\n")


class UdacitySimController:
    """
    Wrapper for communicating with unity simulation.
    """

    def __init__(
            self,
            port: int,
            input_shape: Tuple[int, int, int] = (height, width, channels),
            max_cte_error: float = 7.0,
    ):
        self.port = port
        # sensor size - height, width, depth
        # self.camera_img_size = INPUT_DIM
        self.max_cte_error = max_cte_error

        self.is_success = 0
        self.current_track = None
        # self.image_array = np.zeros(self.camera_img_size)
        self.logger = CustomLogger(str(self.__class__))
        self.client_thread = Thread(target=start_app, args=(flask_app, sio, self.port))
        self.client_thread.daemon = True
        self.input_shape = input_shape

        # self.logger = GlobalLog("UdacitySimController")
        #
        # self.client_thread = Thread(target=start_app, args=(flask_app, sio, self.port))
        # self.client_thread.daemon = True
        # self.client_thread.start()
        # self.logger = GlobalLog("UdacitySimController")
        #
        # while not is_connect:
        #     # print(is_connect)
        #     time.sleep(0.3)

    def start(self):
        self.client_thread.start()

    def pause(self):
        global pause_state
        pause_state = True

    def resume(self):
        global pause_state
        pause_state = False

    def reset(
            self, skip_generation: bool = False, track_string: Union[str, None] = None
    ) -> tuple[UdacityObservation, dict[str, Any]]:
        global observation
        global done
        global generated_track_string
        global track_sent

        # Remove duplicated initialization code
        observation = UdacityObservation(
            input_image=None,
            position=(0.0, 0.0, 0.0),
            steering_angle=0.0,
            throttle=0.0,
            speed=0.0,
            n_collisions=0,
            n_out_of_tracks=0,
            cte=0.0,
            time=-1
        )
        generated_track_string = None
        done = False
        track_sent = False

        self.is_success = 0
        self.current_track = None

        if not skip_generation and track_string is not None:
            generated_track_string = track_string

        return self.observe(), {}

    def generate_track(self, track_string: Union[str, None] = None):
        global generated_track_string

        if track_string is not None:
            generated_track_string = track_string

    @staticmethod
    def take_action(act: UdacityAction) -> None:

        global action
        action = act

    def observe(self) -> UdacityObservation:
        global done
        global observation

        # while last_obs is image_array:
        #    time.sleep(1.0 / 120.0)
        #    print("Waiting for new image")

        # last_obs = image_array
        # self.image_array = image_array
        #
        # done = self.is_game_over()
        # # z and y coordinates appear to be switched
        # info = {
        #     "is_success": self.is_success,
        #     "track": self.current_track,
        #     "speed": speed,
        #     "pos": (pos_x, pos_z, pos_y),
        #     "cte": cte,
        # }

        return observation

    def quit(self):
        self.logger.info("Stopping client")

    def is_game_over(self) -> bool:
        global observation

        if abs(observation.cte) > self.max_cte_error or observation.n_collisions != "none":
            if abs(observation.cte) > self.max_cte_error:
                self.is_success = 0
            else:
                self.is_success = 1
            return True
        return False

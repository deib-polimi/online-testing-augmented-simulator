import base64
import time
from io import BytesIO
from threading import Thread
import numpy as np
from flask_socketio import SocketIO, emit
from PIL import Image
from flask import Flask

from udacity.action import UdacityAction
from udacity.observation import UdacityObservation
from utils.logger import CustomLogger


class UdacityExecutor:

    def __init__(
            self,
            host: str = "127.0.0.1",
            port: int = 4567,
    ):
        # Simulator network settings
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self.sio = SocketIO(
            self.app,
            async_mode='eventlet',
            cors_allowed_origins="*",
            transports=['websocket'],
            # async_mode=async_mode,
            # cors_allowed_origins="*",
            # logger=True,
            # engineio_logger=True,
        )
        self.sio.on('connect')(self.on_connect)
        self.sio.on('car_telemetry')(self.on_telemetry)
        self.sio.on('episode_metrics')(self.on_episode_metrics)
        self.sio.on('episode_events')(self.on_episode_events)
        self.sio.on('episode_event')(self.on_episode_event)
        self.sio.on('sim_paused')(self.on_sim_paused)
        self.sio.on('sim_resumed')(self.on_sim_resumed)

        # Simulator logging
        self.logger = CustomLogger(str(self.__class__))
        # Simulator
        from udacity.simulator import simulator_state
        self.sim_state = simulator_state
        # Manage connection in separate thread
        self.client_thread = Thread(target=self._start_server)
        self.client_thread.daemon = True

    def on_telemetry(self, data):

        # self.logger.info(f"Received data from udacity client: {data}")
        # TODO: check data image, verify from sender that is not empty
        observation = UdacityObservation(
            input_image=np.array(Image.open(BytesIO(base64.b64decode(data["image"])))),
            position=(float(data["pos_x"]), float(data["pos_y"]), float(data["pos_z"])),
            steering_angle=float(data["steering_angle"]),
            throttle=float(data["throttle"]),
            speed=float(data["speed"]) * 3.6,  # conversion m/s to km/h
            # n_collisions=int(float(data["collisions"])),
            # n_out_of_tracks=int(float(data["oot"])),
            cte=float(data["cte"]),
            time=int(time.time() * 1000)
        )
        self.sim_state['observation'] = observation
        # Sending control
        self.send_control()
        if self.sim_state.get('paused', False):
            self.send_pause()
        else:
            self.send_resume()
        track = self.sim_state.get('track', None)
        if track:
            self.send_track(track)
            self.sim_state['track'] = None

    def on_connect(self):
        self.logger.info("Udacity client connected")
        track = self.sim_state.get('track', None)
        # TODO: do it in a better way
        while not track:
            time.sleep(1)
            track = self.sim_state.get('track', None)
        self.send_track(track)
        self.sim_state['track'] = None

    def on_sim_paused(self, data):
        self.sim_state['sim_state'] = 'paused'

    def on_sim_resumed(self, data):
        # TODO: change 'running' with ENUM
        self.sim_state['sim_state'] = 'running'

    def on_episode_metrics(self, data):
        self.logger.info(f"episode metrics {data}")
        self.sim_state['episode_metrics'] = data

    def on_episode_events(self, data):
        self.logger.info(f"episode events {data}")

    def on_episode_event(self, data):
        self.logger.info(f"episode event {data}")
        self.sim_state['events'].append(data)

    def send_control(self) -> None:
        # self.logger.info(f"Sending control")
        action: UdacityAction = self.sim_state.get('action', None)
        if action:
            self.sio.emit(
                "action",
                data={
                    "steering_angle": action.steering_angle.__str__(),
                    "throttle": action.throttle.__str__(),
                },
                skip_sid=True,
            )

    def send_pause(self):
        self.sio.emit("pause_sim", skip_sid=True)

    def send_resume(self):
        self.sio.emit("resume_sim", skip_sid=True)

    def send_track(self, track):
        self.sio.emit("end_episode", skip_sid=True)
        self.sio.emit("start_episode", data={"track_name": track}, skip_sid=True)

    def start(self):
        # Start Socket IO Server in separate thread
        self.client_thread.start()

    def _start_server(self):
        self.sio.run(self.app, host=self.host, port=self.port)

    def close(self):
        self.sio.stop()


if __name__ == '__main__':
    sim_executor = UdacityExecutor()
    sim_executor.start()

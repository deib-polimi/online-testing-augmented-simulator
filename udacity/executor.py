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
    # TODO: avoid cycles
    from udacity.simulator import UdacitySimulator

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

    def on_connect(self):
        self.logger.info("Udacity client connected")

    def send_control(self) -> None:
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

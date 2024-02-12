import pathlib

from udacity.udacity_controller import UdacitySimController
from udacity.udacity_unity import UnityProcess
from utils.logger import CustomLogger


# TODO: it should extend an abstract simulator
class UdacitySimulator:

    def __init__(
            self,
            simulator_exe_path: str = "./examples/udacity/udacity_utils/sim/udacity_sim.app",
            host: str = "127.0.0.1",
            port: int = 4567,
    ):
        self.simulator_exe_path = simulator_exe_path
        self.host = host
        self.port = port
        self.logger = CustomLogger(str(self.__class__))
        self.simulator = UnityProcess()

        # Verify binary location
        if not pathlib.Path(simulator_exe_path).exists():
            self.logger.error(f"Executable binary to the simulator does not exists. "
                              f"Check if the path {self.simulator_exe_path} is correct.")

    def start(self):
        # Start Unity simulation subprocess
        self.logger.info("Starting Unity process for Udacity simulator...")
        self.simulator.start(
            sim_path=self.simulator_exe_path, headless=False, port=self.port
        )

    def quit(self):
        self.simulator.quit()

import gymnasium as gym


class UdacityAction:

    def __init__(self,
                 steering_angle: float,
                 throttle: float,
                 ):
        self.steering_angle = steering_angle
        self.throttle = throttle

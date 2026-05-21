##! python3
# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# @File: stanley_controller.py
# @Description:
# @Author: Tactics2D Team
# @Version: 0.18.0

import numpy as np
from scipy.interpolate import interp1d
from shapely.geometry import LineString, Point

from tactics2d.participant.trajectory.state import State

from .acceleration_controller import AccelerationController


class StanleyController:
    """This class implements a Stanley controller to output steering and acceleration commands of the vehicle.

    The Stanley method minimizes lateral deviation (cross-track error) and heading error between
    the vehicle's front axle and the reference path. The steering angle is computed as:

        steering = heading_error + arctan(k * cross_track_error / (speed + epsilon))

    Attributes:
        k (float): The cross-track error gain. Higher values give more aggressive lateral correction.
            The default value is 1.0. It can be adjusted by `update_driving_style`.
        epsilon (float): A small softening constant to avoid division by zero at low speeds.
            The default value is 1.0 m/s.
        max_steering (float): The upper limit of the steering angle in radians.
            The default value is 0.6 rad (~34 degrees).
        accel_change_rate (float): The limitation to how quickly the acceleration can change over
            time to ensure smooth transitions. The unit is m^2/s. The default value is 3.0.
            It can be adjusted by `update_driving_style`.
        max_accel (float): The upper limit of the acceleration. The unit is m^2/s.
            The default value is 1.5. It can be adjusted by `update_driving_style`.
        min_accel (float): The lower limit of the acceleration. When negative, it describes the
            upper limit of the deceleration. The unit is m^2/s. The default value is -4.0.
            It can be adjusted by `update_driving_style`.
    """

    k = 1.0
    epsilon = 1.0
    max_steering = 0.6
    accel_change_rate = 3.0
    max_accel = 1.5
    min_accel = -4.0

    def __init__(self, k: float = 1.0, target_speed: float = 5.0):
        self.k = k

        self._k_interpolator = interp1d(
            [-1.0, 1.0], [0.5, 2.0], kind="linear", bounds_error=False, fill_value=(0.5, 2.0)
        )
        self._accel_change_rate_interpolator = interp1d(
            [-1.0, 1.0], [2.0, 6.0], kind="linear", bounds_error=False, fill_value=(2.0, 6.0)
        )
        self._max_accel_interpolator = interp1d(
            [-1.0, 1.0], [1.5, 2.5], kind="linear", bounds_error=False, fill_value=(1.5, 2.5)
        )
        self._min_accel_interpolator = interp1d(
            [-1.0, 1.0], [-3.0, -5.0], kind="linear", bounds_error=False, fill_value=(-3.0, -5.0)
        )

        self._longitudinal_control = AccelerationController(target_speed)

    def update_driving_style(self, style_id: int):
        """This method allows to adopt the controller's behavior by adjusting the internal parameters.

        Args:
            style_id (int): The index to seek for a new driving style.
        """
        self._longitudinal_control.update_driving_style(style_id)

        self.k = self._k_interpolator(style_id)
        self.accel_change_rate = self._accel_change_rate_interpolator(style_id)
        self.max_accel = self._max_accel_interpolator(style_id)
        self.min_accel = self._min_accel_interpolator(style_id)

    def _lateral_control(
        self, ego_state: State, waypoints: LineString, wheel_base: float
    ) -> float:
        # Front axle position
        front_x = ego_state.x + wheel_base * np.cos(ego_state.heading)
        front_y = ego_state.y + wheel_base * np.sin(ego_state.heading)
        front_axle = Point(front_x, front_y)

        # Find the closest point on the path to the front axle
        closest_dist = waypoints.project(front_axle)
        closest_point = waypoints.interpolate(closest_dist)

        # Path tangent direction at the closest point (use a small delta for finite difference)
        delta = 0.1
        p1 = waypoints.interpolate(max(closest_dist - delta, 0))
        p2 = waypoints.interpolate(min(closest_dist + delta, waypoints.length))
        path_heading = np.arctan2(p2.y - p1.y, p2.x - p1.x)

        # Heading error: difference between path direction and vehicle heading
        heading_error = path_heading - ego_state.heading
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))  # normalize to [-pi, pi]

        # Cross-track error: signed lateral distance from front axle to closest point
        # Positive when the path is to the left of the vehicle heading
        dx = closest_point.x - front_x
        dy = closest_point.y - front_y
        cross_track_error = np.cos(path_heading) * dy - np.sin(path_heading) * dx

        # Stanley control law
        speed = max(ego_state.speed, 0.0)
        steering = heading_error + np.arctan2(self.k * cross_track_error, speed + self.epsilon)
        steering = np.clip(steering, -self.max_steering, self.max_steering)

        return steering

    def step(self, ego_state: State, waypoints: LineString, wheel_base: float = 2.637):
        """This method outputs the steering and acceleration command based on the current state of the ego vehicle.

        Args:
            ego_state (State): The current state of the ego vehicle.
            waypoints (LineString): The reference path as a sequence of waypoints.
            wheel_base (float, optional): The wheelbase of the ego vehicle. The default unit is
                meter. Defaults to 2.637 (medium_car).

        Returns:
            steering (float): The steering command for the ego vehicle in radians.
            accel (float): The acceleration command for the ego vehicle in m/s^2.
        """
        steering = self._lateral_control(ego_state, waypoints, wheel_base)
        _, accel = self._longitudinal_control.step(ego_state)
        return steering, accel

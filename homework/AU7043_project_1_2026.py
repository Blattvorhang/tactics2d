##! python3

import sys

sys.path.append(".")

import logging

logging.basicConfig(level=logging.INFO)

import numpy as np
from car_following import CarFollowingEnv


# class MyCarFollowingModel:
#     def __init__(self):
#         # you can define the default parameters here
#         pass

#     def step(self):  # customize the input for your car following model
#         steering = 0
#         accel = 0.1
#         # steering = np.random.uniform(low=-0.5, high=0.5)
#         # accel = np.random.uniform(low=-4, high=2)
#         return steering, accel


class MyCarFollowingModel:
    def __init__(self):
        # ===== IDM parameters (longitudinal control) =====
        self.v0 = 20.0       # desired speed (m/s)
        self.T = 1.2         # desired time headway (s)
        self.s0 = 8.0        # minimum gap (m)
        self.a_max = 2.0     # maximum acceleration (m/s^2)
        self.b = 2.5         # comfortable deceleration (m/s^2)
        self.delta = 4       # acceleration exponent

        # ===== Pure Pursuit parameters (lateral control) =====
        self.k_ld = 0.8      # lookahead gain
        self.L_min = 5.0     # minimum lookahead distance (m)

        # ===== Steering smoothing =====
        self.prev_steering = 0.0
        self.alpha = 0.8     # low-pass filter coefficient
        self.max_steer_rate = 0.1  # maximum steering change per step (rad)

    def step(self, ego_state, target_state):
        # =========================
        # 1. Compute relative states
        # =========================
        dx = target_state.x - ego_state.x
        dy = target_state.y - ego_state.y

        # Project relative position onto ego heading (longitudinal distance)
        ego_heading = ego_state.heading
        d = dx * np.cos(ego_heading) + dy * np.sin(ego_heading)

        d = max(d, 0.1)  # avoid division by zero

        v = ego_state.speed
        v_lead = target_state.speed
        delta_v = v - v_lead

        # =========================
        # 2. IDM longitudinal control
        # =========================
        # Desired dynamic gap
        s_star = self.s0 + v * self.T + (v * delta_v) / (2 * np.sqrt(self.a_max * self.b))

        # IDM acceleration
        accel = self.a_max * (1 - (v / self.v0) ** self.delta - (s_star / d) ** 2)

        # Clamp acceleration to physical limits
        accel = np.clip(accel, -4.0, 2.0)

        # =========================
        # 3. Pure Pursuit lateral control
        # =========================
        # Compute lookahead distance based on speed
        Ld = max(self.k_ld * v, self.L_min)

        # Use target vehicle position as tracking point
        target_x = target_state.x
        target_y = target_state.y

        # Transform target position into ego coordinate frame
        dx_local = dx * np.cos(-ego_heading) - dy * np.sin(-ego_heading)
        dy_local = dx * np.sin(-ego_heading) + dy * np.cos(-ego_heading)

        # Avoid numerical instability
        if dx_local == 0:
            steering = 0.0
        else:
            curvature = 2 * dy_local / (Ld ** 2)
            steering = np.arctan(curvature)

        # =========================
        # 4. Steering smoothing (critical for stability)
        # =========================

        # Apply low-pass filter to remove high-frequency noise
        steering = self.alpha * self.prev_steering + (1 - self.alpha) * steering

        # Apply steering rate limit to constrain delta steering
        delta = steering - self.prev_steering
        delta = np.clip(delta, -self.max_steer_rate, self.max_steer_rate)
        steering = self.prev_steering + delta

        # Update previous steering
        self.prev_steering = steering

        return steering, accel


def main():
    env = CarFollowingEnv(dataset="ngsim", render_mode="human")
    observation, _ = env.reset()

    car_following_model = MyCarFollowingModel()

    for step in range(400):
        env.render()
        ego_state = env.get_ego_state()
        target_vehicle_state = env.get_target_state()

        action = car_following_model.step(ego_state, target_vehicle_state)
        observation = env.step(action)
        

if __name__ == "__main__":
    main()

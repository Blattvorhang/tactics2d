###! python3
# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# @File: AU7043_project_2_2025.py
# @Description:
# @Author: Tactics2D Team
# @Version:

import sys

sys.path.append(".")

import logging

logging.basicConfig(level=logging.INFO)

import numpy as np
from lane_changing import LaneChangingEnv


def _road_unit(centerline):
    """Unit direction vector along the lane travel direction."""
    s = np.array(centerline[0], dtype=float)
    e = np.array(centerline[-1], dtype=float)
    d = e - s
    length = np.linalg.norm(d)
    return d / length if length > 1e-6 else np.array([1.0, 0.0])
 
 
def _road_heading(centerline):
    """Lane heading angle in the global coordinate frame (radians)."""
    u = _road_unit(centerline)
    return np.arctan2(u[1], u[0])
 
 
def _closest_point_and_dist(x, y, centerline):
    """
    Compute the closest point on a directed line segment to (x, y),
    and the signed lateral distance.
    Sign convention: positive = left of the travel direction
    (left normal = road unit vector rotated 90 degrees CCW).
    """
    s = np.array(centerline[0], dtype=float)
    e = np.array(centerline[-1], dtype=float)
    u = _road_unit(centerline)
    p = np.array([x, y], dtype=float)
    length = np.linalg.norm(e - s)
 
    t = np.clip(np.dot(p - s, u), 0.0, length)
    closest = s + t * u
 
    perp = np.array([-u[1], u[0]])           # left normal vector
    signed_dist = np.dot(p - closest, perp)  # positive = left of lane
    abs_dist = abs(signed_dist)
    return closest, signed_dist, abs_dist
 
 
def _dist_to_lane(x, y, centerline):
    """Absolute lateral distance from point (x, y) to the lane centerline."""
    _, _, d = _closest_point_and_dist(x, y, centerline)
    return d
 
 
def _forward_proj(ego_x, ego_y, vx, vy, centerline):
    """
    Longitudinal projection distance of another vehicle relative to the ego
    along the lane travel direction. Positive = ahead of ego.
    """
    u = _road_unit(centerline)
    s = np.array(centerline[0], dtype=float)
    ego_proj = np.dot(np.array([ego_x, ego_y]) - s, u)
    v_proj   = np.dot(np.array([vx, vy]) - s, u)
    return v_proj - ego_proj


# class MyLaneChangingModel:
#     def __init__(self):
#         pass

#     def step(self):  # customize the input for your car following model
#         steering = 0
#         accel = 0.2
#         # steering = np.random.uniform(low=-0.5, high=0.5)
#         # accel = np.random.uniform(low=-4, high=2)
#         return steering, accel


class MyLaneChangingModel:
    """
    Lane-changing algorithm:
      Longitudinal: IDM (Intelligent Driver Model)
      Lateral:      Stanley controller
      LC decision:  Gap-based evaluation with safety checks, inspired by MOBIL
 
    Key parameters are documented in __init__.
    """
 
    def __init__(self):
        # ── IDM parameters ────────────────────────────────────────
        self.v0      = 50.0   # desired speed (m/s, ~180 km/h)
        self.T       = 0.8    # safe time headway (s)
        self.a_max   = 2.0    # maximum acceleration (m/s²)
        self.b       = 5.0    # comfortable deceleration (m/s²)
        self.delta   = 4      # acceleration exponent
        self.s0      = 1.0    # minimum bumper-to-bumper gap (m)
        self.ego_len = 5.0    # estimated ego vehicle length (m)
 
        # ── Stanley lateral controller parameters ─────────────────
        self.k_cte   = 1.2    # cross-track error gain
        self.k_head  = 1.0    # heading error gain (kept as interface; normally 1)
 
        # ── Lane-change state machine ─────────────────────────────
        self.target_lane_id   = None   # target lane ID; None means keep current lane
        self.lc_steps         = 0      # steps elapsed in the current lane change
        self.lc_max_steps     = 60     # max steps allowed for one lane change (timeout)
        self.cooldown         = 0      # remaining cooldown steps after a lane change
        self.cooldown_reset   = 60     # total cooldown steps after each lane change
 
        # ── Lane-change safety thresholds ─────────────────────────
        self.lc_min_front_gap = 15.0  # minimum required gap ahead in target lane (m)
        self.lc_min_rear_gap  = 12.0  # minimum required gap behind in target lane (m)
        self.lc_max_rel_speed = 8.0   # max tolerated approach speed of rear vehicle (m/s)
 
        # ── Lane-change trigger conditions ────────────────────────
        self.lc_trigger_gap   = 65.0  # trigger LC evaluation when front gap < this (m)
        self.lc_trigger_dv    = 1.0   # trigger LC evaluation when lead vehicle is slower by this (m/s)
 
        # ── highD location 1 lane groups (two travel directions) ──
        self._lane_groups = [
            ["102000", "102001", "102002"],
            ["102003", "102004", "102005"],
        ]
 
    # ── Utility methods ───────────────────────────────────────────────────────
 
    def _get_ego_lane_and_group(self, ego_x, ego_y, ego_heading, centerlines):
        """
        Identify the ego vehicle's current lane ID and the co-directional lane group.
        Co-direction criterion: angle between ego heading and lane heading < 90°.
        """
        # Find the nearest centerline
        min_d, current_lane_id = float('inf'), None
        for lid, cl in centerlines.items():
            d = _dist_to_lane(ego_x, ego_y, cl)
            if d < min_d:
                min_d, current_lane_id = d, lid
 
        # Determine the co-directional lane group
        ego_group = None
        for group in self._lane_groups:
            if current_lane_id in group:
                cl0 = centerlines[group[0]]
                road_h = _road_heading(cl0)
                diff = abs(np.arctan2(
                    np.sin(ego_heading - road_h),
                    np.cos(ego_heading - road_h)
                ))
                if diff < np.pi / 2:
                    ego_group = group
                else:
                    # Heading opposed to this group — pick the other group (safety fallback)
                    ego_group = [g for g in self._lane_groups if g != group][0]
                break
 
        if ego_group is None:
            ego_group = self._lane_groups[0]
 
        return current_lane_id, ego_group
 
    def _get_front_rear(self, ego_x, ego_y, ego_speed, lane_id, centerlines, other_states):
        """
        Find the nearest leading and following vehicles in the specified lane.
 
        Returns:
            front_v, front_gap, rear_v, rear_gap
            gap = longitudinal clearance after subtracting vehicle length estimate.
            Vehicle is None when no vehicle is found.
        """
        cl = centerlines[lane_id]
        u  = _road_unit(cl)
        s  = np.array(cl[0], dtype=float)
 
        ego_proj = np.dot(np.array([ego_x, ego_y]) - s, u)
 
        front_v, front_gap = None, float('inf')
        rear_v,  rear_gap  = None, float('inf')
 
        for vid, state in other_states.items():
            # Only consider vehicles within half lane width of the centerline
            if _dist_to_lane(state.x, state.y, cl) > 2.0:
                continue
 
            v_proj = np.dot(np.array([state.x, state.y]) - s, u)
            rel    = v_proj - ego_proj
 
            if rel > 0:                        # vehicle is ahead
                gap = rel - self.ego_len       # subtract vehicle body length
                if gap < front_gap:
                    front_gap, front_v = gap, state
            elif rel < 0:                      # vehicle is behind
                gap = -rel - self.ego_len
                if gap < rear_gap:
                    rear_gap, rear_v = gap, state
 
        return front_v, max(front_gap, 0.0), rear_v, max(rear_gap, 0.0)
 
    # ── IDM longitudinal control ──────────────────────────────────────────────
 
    def _idm_accel(self, v, front_v, front_gap):
        """
        IDM acceleration output.
        Reference: Treiber et al. (2000).
        """
        v  = max(v, 0.0)
        v0 = self.v0
 
        if front_v is None or front_gap > 200:
            # Free-flow regime: only limited by desired speed
            accel = self.a_max * (1.0 - (v / v0) ** self.delta)
        else:
            # Emergency brake: gap < 0.3 s headway → max deceleration
            if front_gap < v * 0.3:
                return -4.0

            dv     = v - front_v.speed                       # approach speed
            s_star = self.s0 + max(
                0.0,
                v * self.T + v * dv / (2.0 * np.sqrt(self.a_max * self.b))
            )
            effective_gap = max(front_gap, 0.1)
            # Use exponent 40: essentially binary braking at s* boundary;
            # emergency brake at v*0.3 provides the safety net
            accel = self.a_max * (
                1.0 - (v / v0) ** self.delta - (s_star / effective_gap) ** 40
            )
 
        return float(np.clip(accel, -4.0, 2.0))
 
    # ── Stanley lateral control ───────────────────────────────────────────────
 
    def _stanley_steer(self, ego_x, ego_y, ego_heading, ego_speed, target_cl):
        """
        Stanley lateral controller.
        steering = k_head * heading_error - arctan(k_cte * cte / speed)
 
        cte > 0: ego is left of lane centerline  → steer right (negative)
        cte < 0: ego is right of lane centerline → steer left  (positive)
        """
        road_h = _road_heading(target_cl)
        heading_error = road_h - ego_heading
        # Normalize to (-π, π]
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
 
        _, cte, _ = _closest_point_and_dist(ego_x, ego_y, target_cl)
        speed      = max(ego_speed, 1.0)
 
        steering   = self.k_head * heading_error - np.arctan2(self.k_cte * cte, speed)
        return float(np.clip(steering, -0.5, 0.5))
 
    # ── Lane-change safety check ──────────────────────────────────────────────
 
    def _is_safe_to_change(self, ego_x, ego_y, ego_speed, target_lane_id, centerlines, other_states):
        """
        Check whether it is safe to merge into the target lane:
        - Sufficient gap ahead in the target lane.
        - No rear-end collision risk from following traffic.
        """
        fv, fg, rv, rg = self._get_front_rear(
            ego_x, ego_y, ego_speed, target_lane_id, centerlines, other_states
        )
 
        req_front = max(self.lc_min_front_gap, ego_speed * 0.8)
        if fg < req_front:
            return False
 
        if rg < self.lc_min_rear_gap:
            return False
 
        if rv is not None:
            rel_speed = rv.speed - ego_speed       # positive = rear vehicle approaching
            if rel_speed > self.lc_max_rel_speed and rg < 40:
                return False
 
        return True
 
    # ── Lane-change gain evaluation (MOBIL-inspired) ─────────────────────────
 
    def _lane_change_gain(self, ego_x, ego_y, ego_speed,
                          current_lane_id, target_lane_id,
                          centerlines, other_states):
        """
        Estimate the benefit of changing lanes based on achievable speed.
        gain = speed_potential(target) − speed_potential(current)
        Includes a stuck-scaling factor: when ego is slow, any improvement is amplified.
        """
        fv_cur, fg_cur, _, _ = self._get_front_rear(
            ego_x, ego_y, ego_speed, current_lane_id, centerlines, other_states
        )
        fv_tgt, fg_tgt, _, _ = self._get_front_rear(
            ego_x, ego_y, ego_speed, target_lane_id, centerlines, other_states
        )
        cur_spd = self.v0 if fv_cur is None else min(self.v0, fv_cur.speed)
        tgt_spd = self.v0 if fv_tgt is None else min(self.v0, fv_tgt.speed)
        gain = tgt_spd - cur_spd

        # Modest gap bonus to reduce lane change frequency
        gap_bonus = (fg_tgt - fg_cur) * 0.02
        gain += gap_bonus

        # Stuck scaling: only incentivize lane changes when truly stuck
        speed_ratio = ego_speed / self.v0
        if speed_ratio < 0.55:
            gain = gain * 1.5 + 1.5
        elif speed_ratio < 0.70:
            gain = gain * 1.2 + 0.5

        return gain
 
    # ── Main decision step ────────────────────────────────────────────────────
 
    def step(self, ego_state, other_states, centerlines):
        """
        Args:
            ego_state    -- ego vehicle state (x, y, heading, vx, vy, speed, accel)
            other_states -- dict of surrounding vehicle states
            centerlines  -- dict of lane centerline point lists
        Returns:
            (steering, accel) tuple for the continuous action space
        """
        ex, ey    = ego_state.x, ego_state.y
        eh        = ego_state.heading
        ev        = ego_state.speed
 
        # 1. Identify current lane and co-directional lane group
        cur_lid, ego_group = self._get_ego_lane_and_group(ex, ey, eh, centerlines)
 
        # Decrement post-LC cooldown counter
        if self.cooldown > 0:
            self.cooldown -= 1
 
        # 2. Longitudinal control: IDM based on current lane lead vehicle
        fv, fg, _, _ = self._get_front_rear(ex, ey, ev, cur_lid, centerlines, other_states)
        accel = self._idm_accel(ev, fv, fg)
 
        # 3. Lane-change decision (stay in current lane if no target set)
        if self.target_lane_id is None:
            # Always evaluate lane change when not in cooldown;
            # safety checks and gain threshold prevent unsafe/unnecessary changes
            if self.cooldown == 0:
                    cur_idx = ego_group.index(cur_lid) if cur_lid in ego_group else 0
 
                    best_gain, best_lid = 0.0, None
                    # Prefer left (fast) lane first; fall back to right lane
                    for offset in [-1, 1]:
                        new_idx = cur_idx + offset
                        if not (0 <= new_idx < len(ego_group)):
                            continue
                        cand = ego_group[new_idx]
                        if not self._is_safe_to_change(ex, ey, ev, cand, centerlines, other_states):
                            continue
                        gain = self._lane_change_gain(
                            ex, ey, ev, cur_lid, cand, centerlines, other_states
                        )
                        if gain > best_gain:
                            best_gain, best_lid = gain, cand
 
                    if best_lid is not None:
                        self.target_lane_id = best_lid
                        self.lc_steps = 0
                        logging.info(f"[LC] Lane change initiated: {cur_lid} → {best_lid}, front gap={fg:.1f}m")
 
        # 4. Lane-change timeout or mid-LC safety abort
        if self.target_lane_id is not None:
            self.lc_steps += 1
            if self.lc_steps > self.lc_max_steps:
                logging.info(f"[LC] Lane change timed out, aborting target lane {self.target_lane_id}")
                self.target_lane_id = None
                self.cooldown = self.cooldown_reset
            elif self.lc_steps > 3:
                # Mid-LC safety monitor: check for emerging rear vehicles in target lane
                _, _, rv, rg = self._get_front_rear(
                    ex, ey, ev, self.target_lane_id, centerlines, other_states
                )
                if rv is not None and rg < 8.0:
                    rel_spd = rv.speed - ev
                    if rel_spd > 2.0:
                        logging.info(f"[LC] Abort+accelerate: fast rear vehicle in target lane "
                                    f"(gap={rg:.1f}m, rel_spd={rel_spd:.1f}m/s)")
                        self.target_lane_id = None
                        self.cooldown = self.cooldown_reset
                        # Override accel to match rear vehicle speed and avoid collision
                        accel = max(accel, min(2.0, rel_spd * 2.0))
 
        # 5. Check if lane change is complete
        if self.target_lane_id is not None:
            dist_to_target = _dist_to_lane(ex, ey, centerlines[self.target_lane_id])
            if dist_to_target < 0.6 and cur_lid == self.target_lane_id:
                logging.info(f"[LC] Lane change complete, now in lane {cur_lid}")
                self.target_lane_id = None
                self.cooldown = self.cooldown_reset
 
        # 6. Lateral control: Stanley toward target lane or current lane centerline
        steer_target = (
            centerlines[self.target_lane_id]
            if self.target_lane_id is not None
            else centerlines[cur_lid]
        )
        steering = self._stanley_steer(ex, ey, eh, ev, steer_target)
 
        return np.float32(steering), np.float32(accel)


def main(level="easy"):
    if level == "easy":
        max_step = 500
    elif level == "medium":
        max_step = 400
    elif level == "hard":
        max_step = 300

    env = LaneChangingEnv(render_mode="human", max_step=max_step)
    observation, infos = env.reset()

    # The infors include the traffic status, the ego vehicle status, the other vehicles status, and the centerlines
    logging.info(f"Infos keys: {list(infos.keys())}")
    logging.info(f"Centerline IDs: {list(infos['centerlines'].keys())}")

    lane_changing_model = MyLaneChangingModel()

    for step in range(max_step + 10):
        env.render()

        action = lane_changing_model.step(
            infos["ego_state"],
            infos["other_states"],
            infos["centerlines"],
        )
        observation, infos = env.step(action)

        status_name = infos["status"].name
        logging.debug(f"Step {step}: status={status_name}, "
                      f"speed={infos['ego_state'].speed:.1f} m/s")

        if status_name not in ["NORMAL", "COMPLETED"]:
            raise RuntimeError(
                f"Simulation failed with status: {infos['status'].name} at step {step}."
            )
        elif status_name == "COMPLETED":
            logging.info(f"Simulation completed successfully at step {step}.")
            break


if __name__ == "__main__":
    np.random.seed(0)  # define the random seed to reproduce the scenario
    main(level="medium")

from __future__ import annotations

import numpy as np

from drone_dispatch_env.config import Config
from drone_dispatch_env.world import Router


class RoleCPlanner:
    """
    Deadline- and battery-aware planning policy.

    The policy starts from the robust GreedyNearest logic, then improves its
    assignment choice by considering deadline urgency and battery feasibility.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._grid_key = None
        self._router = None
        self._fields = {}

    def _ensure_router(self, grid: np.ndarray) -> None:
        key = grid.tobytes()

        if key != self._grid_key:
            self._grid_key = key
            self._router = Router(grid, self.cfg.neighborhood)
            self._fields = {}

    def _dist(self, grid: np.ndarray, source, target) -> float:
        self._ensure_router(grid)

        source = (int(source[0]), int(source[1]))
        target = (int(target[0]), int(target[1]))

        if source not in self._fields:
            self._fields[source] = self._router.dist_field(source)

        d = float(self._fields[source][target[0], target[1]])

        if not np.isfinite(d):
            return float(abs(source[0] - target[0]) + abs(source[1] - target[1]))

        return d

    def act(self, obs):
        c = self.cfg

        mask = np.asarray(obs["action_mask"], dtype=bool)
        drones = np.asarray(obs["drones"], dtype=float)
        orders = np.asarray(obs["orders"], dtype=float)
        grid = np.asarray(obs["grid"])

        # 1. Same safe charging logic as GreedyNearest:
        # if a low-battery idle drone can charge, send it to charge.
        for drone_id in range(c.n_drones):
            charge_action = c.charge_index(drone_id)

            if mask[charge_action] and drones[drone_id, 2] < c.charge_threshold:
                return int(charge_action)

        # 2. Evaluate all valid assignments.
        # Lower score is better.
        best_action = None
        best_score = np.inf

        for action in range(c.n_drones * c.k_max):
            if not mask[action]:
                continue

            drone_id = action // c.k_max
            slot = action % c.k_max

            drone_x, drone_y, soc = (
                drones[drone_id, 0],
                drones[drone_id, 1],
                drones[drone_id, 2],
            )

            ox, oy, dx, dy, age = orders[slot]

            # Keep the same low-battery protection as the baseline.
            if soc < c.charge_threshold:
                continue

            pickup_dist = self._dist(grid, (drone_x, drone_y), (ox, oy))
            delivery_dist = self._dist(grid, (ox, oy), (dx, dy))
            total_trip = pickup_dist + delivery_dist

            # Order becomes more urgent as age approaches SLA = 60.
            remaining_slack = max(0.0, c.sla_steps - float(age))
            urgency = 1.0 - (remaining_slack / c.sla_steps)

            # Expected SoC after the entire pickup + delivery route.
            expected_soc_after = float(soc) - c.e_move * total_trip
            battery_risk = max(0.0, 0.10 - expected_soc_after)

            # Main preference remains close pickup distance.
            # Deadline urgency and battery safety adjust the choice.
            score = (
                pickup_dist
                - 0.75 * urgency
                + 12.0 * battery_risk
                + 0.05 * delivery_dist
            )

            if score < best_score:
                best_score = score
                best_action = action

        if best_action is not None:
            return int(best_action)

        return int(c.noop_index)
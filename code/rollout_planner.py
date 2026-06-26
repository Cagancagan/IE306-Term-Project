from __future__ import annotations

import numpy as np

from drone_dispatch_env.config import Config
from drone_dispatch_env.world import Router


class ShortHorizonRolloutPlanner:
    """
    Role C short-horizon rollout-inspired planning policy.

    It scores each valid dispatch action using:
    - routed pickup and delivery distance,
    - order urgency,
    - post-delivery battery safety,
    - estimated future fleet capacity.

    planning_depth controls how strongly the policy considers expected
    future demand and future drone availability.
    """

    def __init__(
        self,
        cfg: Config,
        planning_depth: int = 1,
        forecast_weight: float = 0.08,
    ):
        self.cfg = cfg
        self.planning_depth = max(1, int(planning_depth))
        self.forecast_weight = float(forecast_weight)

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

        distance = float(self._fields[source][target[0], target[1]])

        if not np.isfinite(distance):
            return float(abs(source[0] - target[0]) + abs(source[1] - target[1]))

        return distance

    def _future_capacity_penalty(
        self,
        drones: np.ndarray,
        selected_drone: int,
        expected_soc_after: float,
        delivery_dist: float,
    ) -> float:
        """
        Approximate how much assigning this drone now could reduce
        the fleet's ability to handle near-future stochastic demand.
        """
        c = self.cfg

        idle = drones[:, 4] > 0.5
        active = drones[:, 3] > 0.5

        n_idle_now = int(np.sum(idle))
        n_idle_after_assignment = max(0, n_idle_now - 1)

        other_idle_socs = [
            float(drones[d, 2])
            for d in range(c.n_drones)
            if d != selected_drone and idle[d] and active[d]
        ]

        average_other_idle_soc = (
            float(np.mean(other_idle_socs))
            if other_idle_socs
            else 0.0
        )

        # Expected number of new orders over an approximate future horizon.
        expected_future_orders = c.lam * 5.0 * self.planning_depth

        # Penalty when the number of remaining idle drones may be insufficient.
        capacity_shortage = max(
            0.0,
            expected_future_orders - n_idle_after_assignment,
        )

        # Penalty if selected drone will finish with little charge.
        reserve_risk = max(0.0, 0.25 - expected_soc_after)

        # Penalty if the other idle drones also have weak battery levels.
        fleet_battery_risk = max(0.0, 0.45 - average_other_idle_soc)

        # Longer delivery keeps the selected drone unavailable longer.
        service_time_risk = (
            delivery_dist / max(1.0, float(c.H + c.W))
        ) * self.planning_depth

        return (
            2.0 * capacity_shortage
            + 6.0 * reserve_risk
            + 1.5 * fleet_battery_risk
            + 0.5 * service_time_risk
        )

    def act(self, obs) -> int:
        c = self.cfg

        mask = np.asarray(obs["action_mask"], dtype=bool)
        drones = np.asarray(obs["drones"], dtype=float)
        orders = np.asarray(obs["orders"], dtype=float)
        grid = np.asarray(obs["grid"])

        # Keep the robust charging policy from GreedyNearest.
        for drone_id in range(c.n_drones):
            charge_action = c.charge_index(drone_id)

            if mask[charge_action] and drones[drone_id, 2] < c.charge_threshold:
                return int(charge_action)

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

            # Do not assign a drone that should be charging.
            if soc < c.charge_threshold:
                continue

            pickup_dist = self._dist(
                grid,
                (drone_x, drone_y),
                (ox, oy),
            )

            delivery_dist = self._dist(
                grid,
                (ox, oy),
                (dx, dy),
            )

            total_trip = pickup_dist + delivery_dist

            # Urgency grows as the order age approaches the SLA limit.
            remaining_slack = max(0.0, c.sla_steps - float(age))
            urgency = 1.0 - (remaining_slack / c.sla_steps)

            expected_soc_after = float(soc) - c.e_move * total_trip
            battery_risk = max(0.0, 0.10 - expected_soc_after)

            # Immediate decision quality.
            immediate_score = (
                pickup_dist
                - 0.75 * urgency
                + 12.0 * battery_risk
                + 0.05 * delivery_dist
            )

            # Planning component: approximate future service capacity.
            future_penalty = self._future_capacity_penalty(
                drones=drones,
                selected_drone=drone_id,
                expected_soc_after=expected_soc_after,
                delivery_dist=delivery_dist,
            )

            score = immediate_score + self.forecast_weight * future_penalty

            if score < best_score:
                best_score = score
                best_action = action

        if best_action is not None:
            return int(best_action)

        return int(c.noop_index)
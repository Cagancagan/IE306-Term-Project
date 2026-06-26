from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from drone_dispatch_env.config import Config


StateKey = Tuple[int, int, int, int, int]
ModelEntry = Tuple[StateKey, float, bool, np.ndarray]


@dataclass
class DynaQConfig:
    alpha: float = 0.15
    gamma: float = 0.95
    epsilon: float = 0.10
    planning_steps: int = 5


class DynaQPlanner:
    """
    Role C Dyna-Q planner for drone dispatch decisions.

    This first version defines:
    - a compact discretized state representation,
    - Q-table storage,
    - a learned transition model,
    - placeholders for action selection and planning updates.
    """

    def __init__(
        self,
        cfg: Config,
        dynaq_cfg: DynaQConfig | None = None,
        seed: int = 0,
    ):
        self.cfg = cfg
        self.dynaq_cfg = dynaq_cfg or DynaQConfig()
        self.rng = np.random.default_rng(seed)

        self.q: Dict[StateKey, np.ndarray] = defaultdict(
            lambda: np.zeros(self.cfg.n_actions, dtype=np.float32)
        )

        self.model: Dict[Tuple[StateKey, int], ModelEntry] = {}

    def encode_state(self, obs) -> StateKey:
        """
        Compress the full observation into a small tabular state.

        Components:
        1. Number of visible pending orders
        2. Number of idle drones
        3. Number of low-battery idle drones
        4. Urgency bucket of the oldest visible order
        5. Mean battery bucket across active drones
        """
        drones = np.asarray(obs["drones"], dtype=float)
        orders = np.asarray(obs["orders"], dtype=float)
        mask = np.asarray(obs["action_mask"], dtype=bool)

        assignment_mask = mask[: self.cfg.n_drones * self.cfg.k_max]
        n_visible_orders = int(
            np.sum(
                np.any(
                    assignment_mask.reshape(self.cfg.n_drones, self.cfg.k_max),
                    axis=0,
                )
            )
        )

        idle = drones[:, 4] > 0.5
        active = drones[:, 3] > 0.5
        low_battery_idle = idle & (drones[:, 2] < self.cfg.charge_threshold)

        n_idle = int(np.sum(idle))
        n_low_battery_idle = int(np.sum(low_battery_idle))

        if n_visible_orders > 0:
            ages = orders[:n_visible_orders, 4]
            oldest_age = float(np.max(ages))
        else:
            oldest_age = 0.0

        urgency_bucket = min(
            4,
            int((oldest_age / max(1, self.cfg.sla_steps)) * 5),
        )

        if np.any(active):
            mean_soc = float(np.mean(drones[active, 2]))
        else:
            mean_soc = 0.0

        mean_soc_bucket = min(4, int(mean_soc * 5))

        return (
            min(n_visible_orders, 5),
            min(n_idle, self.cfg.n_drones),
            min(n_low_battery_idle, self.cfg.n_drones),
            urgency_bucket,
            mean_soc_bucket,
        )

    def valid_actions(self, obs) -> np.ndarray:
        return np.flatnonzero(np.asarray(obs["action_mask"], dtype=bool))

    def act(self, obs) -> int:
        """
        Epsilon-greedy action selection.
        Training support will be added in the next step.
        """
        state = self.encode_state(obs)
        valid = self.valid_actions(obs)

        if len(valid) == 0:
            return int(self.cfg.noop_index)

        if self.rng.random() < self.dynaq_cfg.epsilon:
            return int(self.rng.choice(valid))

        q_values = self.q[state].copy()
        invalid = np.ones(self.cfg.n_actions, dtype=bool)
        invalid[valid] = False
        q_values[invalid] = -np.inf

        return int(np.argmax(q_values))

    def _q_update(
        self,
        state: StateKey,
        action: int,
        reward: float,
        next_state: StateKey,
        next_valid_actions: np.ndarray,
        done: bool,
    ) -> None:
        """One tabular Q-learning update."""
        alpha = self.dynaq_cfg.alpha
        gamma = self.dynaq_cfg.gamma

        current_q = float(self.q[state][action])

        if done or len(next_valid_actions) == 0:
            target = reward
        else:
            next_q = self.q[next_state][next_valid_actions]
            target = reward + gamma * float(np.max(next_q))

        self.q[state][action] = current_q + alpha * (target - current_q)

    def update(self, obs, action: int, reward: float, next_obs, done: bool) -> None:
        """
        Learn from one real environment transition, then perform Dyna-Q
        planning updates using transitions stored in the learned model.
        """
        state = self.encode_state(obs)
        next_state = self.encode_state(next_obs)
        next_valid_actions = self.valid_actions(next_obs)

        # 1. Learn from the real transition.
        self._q_update(
            state=state,
            action=int(action),
            reward=float(reward),
            next_state=next_state,
            next_valid_actions=next_valid_actions,
            done=bool(done),
        )

        # 2. Save the observed transition in the learned model.
        self.model[(state, int(action))] = (
            next_state,
            float(reward),
            bool(done),
            next_valid_actions.copy(),
        )

        # 3. Replay past transitions as planning updates.
        model_keys = list(self.model.keys())

        for _ in range(self.dynaq_cfg.planning_steps):
            sampled_state, sampled_action = model_keys[
                self.rng.integers(len(model_keys))
            ]

            (
                sampled_next_state,
                sampled_reward,
                sampled_done,
                sampled_next_valid_actions,
            ) = self.model[(sampled_state, sampled_action)]

            self._q_update(
                state=sampled_state,
                action=sampled_action,
                reward=sampled_reward,
                next_state=sampled_next_state,
                next_valid_actions=sampled_next_valid_actions,
                done=sampled_done,
            )
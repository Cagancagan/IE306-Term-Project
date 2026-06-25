import numpy as np

from drone_dispatch_env.config import Config
from drone_dispatch_env.world import Router


_cfg = Config()
_grid_key = None
_router = None
_fields = {}


def _get_router(grid):
    global _grid_key, _router, _fields

    key = np.asarray(grid).tobytes()

    if key != _grid_key:
        _grid_key = key
        _router = Router(np.asarray(grid), _cfg.neighborhood)
        _fields = {}

    return _router


def _distance_field(grid, source):
    router = _get_router(grid)

    source = (int(source[0]), int(source[1]))

    if source not in _fields:
        _fields[source] = router.dist_field(source)

    return _fields[source]


def flatten_obs(obs):
    drones = obs["drones"].astype(np.float32).copy()
    orders = obs["orders"].astype(np.float32).copy()
    time = obs["time"].astype(np.float32).copy()
    grid = obs["grid"].astype(np.float32).copy()

    drones[:, 0:2] /= 20.0
    orders[:, 0:4] /= 20.0
    orders[:, 4] /= 500.0
    grid /= 3.0

    routed_distances = np.zeros((8, 20), dtype=np.float32)

    for slot in range(20):
        field = _distance_field(obs["grid"], obs["orders"][slot, 0:2])

        for drone in range(8):
            x = int(obs["drones"][drone, 0])
            y = int(obs["drones"][drone, 1])

            distance = field[x, y]

            if not np.isfinite(distance):
                distance = 40.0

            routed_distances[drone, slot] = distance / 40.0

    return np.concatenate([
        drones.flatten(),
        orders.flatten(),
        routed_distances.flatten(),
        grid.flatten(),
        time.flatten()
    ]).astype(np.float32)


def choose_random_valid_action(obs, rng):
    valid_actions = np.flatnonzero(obs["action_mask"])

    if len(valid_actions) == 0:
        raise RuntimeError("Geçerli aksiyon bulunamadı.")

    return int(rng.choice(valid_actions))
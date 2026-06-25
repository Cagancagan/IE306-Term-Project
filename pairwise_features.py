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


def build_pairwise_features(obs):
    drones = obs["drones"]
    orders = obs["orders"]

    pair_features = np.zeros((1, 160, 11), dtype=np.float32)
    charge_features = np.zeros((1, 8, 5), dtype=np.float32)

    for drone in range(8):
        charge_features[0, drone] = np.array([
            drones[drone, 2],
            drones[drone, 3],
            drones[drone, 0] / 20.0,
            drones[drone, 1] / 20.0,
            obs["time"][0]
        ], dtype=np.float32)

    for slot in range(20):
        pickup_x = int(orders[slot, 0])
        pickup_y = int(orders[slot, 1])
        dropoff_x = int(orders[slot, 2])
        dropoff_y = int(orders[slot, 3])

        pickup_field = _distance_field(
            obs["grid"],
            (pickup_x, pickup_y)
        )

        pickup_to_dropoff = pickup_field[dropoff_x, dropoff_y]

        if not np.isfinite(pickup_to_dropoff):
            pickup_to_dropoff = 40.0

        for drone in range(8):
            x = int(drones[drone, 0])
            y = int(drones[drone, 1])

            drone_to_pickup = pickup_field[x, y]

            if not np.isfinite(drone_to_pickup):
                drone_to_pickup = 40.0

            total_distance = drone_to_pickup + pickup_to_dropoff
            battery_margin = drones[drone, 2] - total_distance / 40.0

            action = drone * 20 + slot

            pair_features[0, action] = np.array([
                drones[drone, 2],
                drones[drone, 3],
                orders[slot, 0] / 20.0,
                orders[slot, 1] / 20.0,
                orders[slot, 2] / 20.0,
                orders[slot, 3] / 20.0,
                orders[slot, 4] / 500.0,
                drone_to_pickup / 40.0,
                pickup_to_dropoff / 40.0,
                total_distance / 80.0,
                battery_margin
            ], dtype=np.float32)

    return pair_features, charge_features
# Multi-Agent RL Results

## Scope

This component uses `DroneDispatchMA-v0`, the decentralized multi-agent environment. Each drone is an independent agent with a shared-parameter IDQN network.

## Methods

- Random decentralized baseline
- Simple decentralized heuristic baseline
- Shared-parameter Independent DQN (IDQN)
- Centralized reference comparison using `DroneDispatch-v0` baselines

## Commands

Smoke test:

```bash
python code/train_idqn_ma.py train --episodes 5 --print-every 1
python code/train_idqn_ma.py eval --weights weights/idqn_ma.pt --seeds 0,1
```

Final-ish run:

```bash
python code/train_idqn_ma.py train --episodes 100 --print-every 10 --weights-path weights/idqn_ma.pt --log-path logs/idqn_ma_training.csv
python code/train_idqn_ma.py eval --weights weights/idqn_ma.pt --seeds 0,1,2,3,4 --results-path logs/multi_agent_results.json
```

## Notes for Report

`DroneDispatchMA-v0` exposes decentralized per-drone observations/actions and does not report exactly the same metrics as centralized `DroneDispatch-v0`. Therefore, the centralized policies are used as qualitative reference scores, while IDQN/heuristic/random are evaluated end-to-end on the multi-agent environment.

The main challenge is non-stationarity: each drone learns while other drones are also changing their behavior. Parameter sharing improves sample efficiency, but independent Q-learning still treats other drones as part of a changing environment.

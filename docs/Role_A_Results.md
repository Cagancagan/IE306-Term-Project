# Role A — DQN Family Results

## Scope

Role A is responsible for the value-based dispatcher family:

* DQN
* Double DQN
* Dueling DQN

All three methods were trained and evaluated on the centralized discrete dispatch environment. The primary evaluation metric is `cost_per_order`, where lower is better.

## Main Engineering Fixes

The initial Role A implementation failed mostly because of environment-integration issues rather than because DQN was inherently unsuitable.

Main fixes:

1. Used a single consistent observation preprocessing function for training, acting, and evaluation.
2. Created the Q-network only after the real observation dimension and action dimension were known.
3. Applied action masking during action selection.
4. Applied action masking during target-Q computation.
5. Added a conservative battery-safety policy mask.
6. Added a distance-aware mission-safety mask:

   * drone → pickup
   * pickup → dropoff
   * dropoff → nearest hub/charger
7. Saved and loaded configs together with model weights for reproducibility.
8. Added a `--no-target-network` flag for the target-network ablation.

## Safety-Aware Action Mask

The simulator’s environment action mask only indicates whether an action is legally valid. However, a legally valid assignment can still be unsafe if the drone does not have enough battery to complete the mission and reach a charger.

To reduce battery depletion, an additional safety mask was added on top of the environment mask.

An assignment is blocked if the drone does not appear to have enough battery for:

```text
drone → pickup → dropoff → nearest hub
```

plus a battery reserve.

Final safety parameters used:

```text
safety_threshold = 0.30
battery_reserve = 0.15
epsilon_decay = 0.9995
```

This change reduced depletion events to zero in the final evaluations and allowed the DQN family to outperform the greedy baseline.

## Training Setup

Each method was trained with 3 random seeds:

```text
train seeds: 0, 1, 2
training episodes per seed: 50
evaluation seeds: 0–9
```

Training command template:

```bash
python "code/Role A DQN family/role_a_dqn_fixed.py" train --algo <algo> --seed <seed> --episodes 50 --print-every 10 --safety-threshold 0.30 --battery-reserve 0.15 --epsilon-decay 0.9995
```

Evaluation command template:

```bash
python "code/Role A DQN family/role_a_dqn_fixed.py" eval --algo <algo> --weights weights/role_a_<algo>_seed<seed>.pt --seeds 0,1,2,3,4,5,6,7,8,9
```

## Baseline Results

Evaluation seeds: `0–9`

| Policy         | cost/order | episode return | delivered |
| -------------- | ---------: | -------------: | --------: |
| random         |    20.8761 |        -231.54 |      38.9 |
| greedy_nearest |     3.6106 |        1347.92 |     121.8 |
| milp_rolling   |     3.5940 |        1358.53 |     121.9 |

## DQN Family Results

Evaluation seeds: `0–9`

| Method      | Seed | cost/order | success rate | on-time rate | depletion events | delivered | episode return |
| ----------- | ---: | ---------: | -----------: | -----------: | ---------------: | --------: | -------------: |
| DQN         |    0 |     2.9734 |       0.8540 |       0.9220 |           0.0000 |     118.1 |      1373.9829 |
| DQN         |    1 |     3.9212 |       0.8214 |       0.7546 |           0.0000 |     110.7 |      1098.3731 |
| DQN         |    2 |     2.8351 |       0.8614 |       0.8904 |           0.0000 |     118.5 |      1376.7584 |
| Double DQN  |    0 |     3.2569 |       0.8412 |       0.8981 |           0.0000 |     116.0 |      1303.2360 |
| Double DQN  |    1 |     2.9574 |       0.8552 |       0.8735 |           0.0000 |     117.2 |      1335.0027 |
| Double DQN  |    2 |     3.0662 |       0.8524 |       0.8626 |           0.0000 |     116.6 |      1311.3425 |
| Dueling DQN |    0 |     2.9104 |       0.8608 |       0.8408 |           0.0000 |     117.5 |      1326.0149 |
| Dueling DQN |    1 |     3.0064 |       0.8611 |       0.7513 |           0.0000 |     115.6 |      1245.0359 |
| Dueling DQN |    2 |     2.8843 |       0.8597 |       0.8865 |           0.0000 |     118.2 |      1365.0509 |

## Mean Results Across Training Seeds

| Method      | mean cost/order | Notes                                                     |
| ----------- | --------------: | --------------------------------------------------------- |
| DQN         |           ~3.24 | Beats greedy on average, but seed 1 is worse than greedy. |
| Double DQN  |           ~3.09 | Stable across seeds and beats greedy consistently.        |
| Dueling DQN |           ~2.93 | Best and most stable Role A method.                       |

## Ablation: Target Network On vs Off

Ablation method: Dueling DQN, seed 0
Evaluation seeds: `0–9`

| Variant            | cost/order | success rate | on-time rate | depletion events | delivered | episode return |
| ------------------ | ---------: | -----------: | -----------: | ---------------: | --------: | -------------: |
| Target network ON  |     2.9104 |       0.8608 |       0.8408 |           0.0000 |     117.5 |      1326.0149 |
| Target network OFF |     2.9528 |       0.8637 |       0.7491 |           0.0000 |     115.9 |      1250.7527 |

The target network ON variant achieved slightly lower cost per order and noticeably better on-time rate and episode return. The target network OFF variant still beat the greedy baseline, but it was slightly less stable.

## Final Role A Model Choice

The best overall method is:

```text
Dueling DQN with distance-aware battery safety mask
```

Recommended final weights:

```text
weights/role_a_dueling_seed2.pt
```

Reason:

* Lowest observed Dueling DQN cost/order among the 3 final seeds.
* Zero depletion events.
* Beats both `greedy_nearest` and `milp_rolling` on cost/order.
* Strong success rate and delivered-order count.

## Diagnostic Summary

The first DQN attempts failed mainly because drones depleted their batteries. The environment action mask only filters illegal actions, not unsafe-but-legal assignments. After adding a conservative distance-aware battery safety mask, the agent avoided unsafe assignments, maintained zero depletion events, and learned to outperform the greedy and MILP baselines.

This suggests that in this simulator, safe action feasibility and battery-aware dispatching were more important than simply increasing network complexity.

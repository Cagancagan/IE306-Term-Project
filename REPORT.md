# IE 306 Term Project Report

**Project:** Reinforcement Learning for City-Scale Drone Delivery  
**Environment:** `drone_dispatch_env`  
**Primary metric:** mean `cost_per_order` on evaluation seeds, where lower is better.

---

## 1. Problem Summary

The simulator models an operational drone-dispatch problem on a 2-D grid city. Hubs, fleet size, no-fly zones, and simulator dynamics are fixed by configuration. Our task is to choose operational actions in real time: which drone serves which order, when drones should charge, and how idle capacity should be used.

The centralized dispatcher environment uses a discrete action space. At each decision epoch, the policy can assign an idle drone to one visible order slot, send an idle drone to charge, or defer. The main grading metric is `cost_per_order`, which combines energy, lateness, dropped orders, and depletion penalties normalized by delivered orders.

The project was divided into three role families and two joint components:

- **Role A:** value-based DQN family.
- **Role B:** policy-based / actor-critic family and continuous-control experiment.
- **Role C:** planning / model-based acceleration.
- **Offline RL:** behavior cloning, naive offline DQN failure, and conservative offline Q-learning.
- **Multi-agent RL:** decentralized per-drone agents compared with centralized references.

---

## 2. Evaluation Setup and Baselines

Unless otherwise noted, centralized dispatcher results use the standard evaluation configuration `configs/eval_standard.yaml`. The main centralized baselines are:

- `random`: random valid-action policy.
- `greedy_nearest`: nearest-drone assignment baseline with charging guard.
- `milp_rolling`: rolling-horizon matching baseline.

The final comparison table below combines the main available results from each role. Role A and Role C used evaluation seeds `0–9`; Role B reports held-out seeds `300–319`; offline and multi-agent components used their own evaluation settings because they evaluate different requirements.

| Method / Policy | Eval setting | Cost per order | Delivered orders | Episode return | Notes |
|---|---|---:|---:|---:|---|
| Random baseline | seeds 0–9 | 20.8761 | 38.9 | -231.54 | Centralized random policy |
| Greedy nearest | seeds 0–9 | 3.6106 | 121.8 | 1347.92 | Main classical baseline |
| MILP rolling | seeds 0–9 | 3.5940 | 121.9 | 1358.53 | Strong classical baseline |
| Role A Dueling DQN | train seeds 0–2, eval seeds 0–9 | **2.9337 ± 0.0643** | ~117.1 | ~1312.0 | Best Role A family mean |
| Role B MILP-guided pairwise policy | held-out seeds 300–319 | **2.9684 ± 0.5976** | 123.55 ± 2.66 | 1482.66 ± 89.60 | Imitation-trained policy |
| Role C short-horizon look-ahead | seeds 0–9 | **3.484 ± 1.860** | 123.00 ± 13.99 | — | Lightweight planning policy |

The strongest learned centralized dispatcher in our final role-level results was Role A's Dueling DQN family on mean cost per order. Role B also achieved a strong held-out cost reduction relative to greedy nearest, but it was evaluated on a different held-out seed range. Role C achieved a smaller mean improvement over greedy nearest with larger cross-seed variability.

---

## 3. Role A — Value-Based DQN Family

### 3.1 Methods

Role A implemented three value-based dispatcher variants:

1. **DQN**
2. **Double DQN**
3. **Dueling DQN**

All three methods operate on the centralized discrete dispatcher. Observations are flattened into a fixed vector representation, and Q-values are produced over the full discrete action space. The policy uses the simulator-provided action mask and an additional battery-safety mask.

### 3.2 Main Engineering Fixes

The first Role A attempts failed mainly because of environment-integration and action-feasibility issues, not because DQN was inherently unsuitable. The final Role A pipeline fixed the following issues:

1. A single consistent observation preprocessing function was used for training, acting, and evaluation.
2. The Q-network was created only after the real observation dimension and action dimension were known.
3. The action mask was applied during action selection.
4. The action mask was also applied during target-Q computation.
5. A conservative battery-safety mask was added.
6. A distance-aware mission-safety mask was added.
7. Model checkpoints saved both learned weights and relevant configuration values.
8. A `--no-target-network` flag was added for ablation.

### 3.3 Distance-Aware Battery Safety

The simulator action mask tells us whether an action is legally valid. However, a legal assignment may still be unsafe if a drone cannot complete the mission and reach a charger. We therefore added a policy-level safety mask that blocks assignments when the drone does not appear to have enough battery for:

```text
drone → pickup → dropoff → nearest hub/charger
```

plus a reserve margin.

Final safety parameters:

```text
safety_threshold = 0.30
battery_reserve = 0.15
epsilon_decay = 0.9995
```

This was the key turning point for Role A. Before the safety mask, learned DQN variants often depleted most drones and performed worse than random. After adding distance-aware mission safety, depletion events dropped to zero in final Role A evaluations and the DQN family consistently beat the greedy baseline.

### 3.4 Training Setup

Each method was trained with 3 random seeds.

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

### 3.5 Role A Results

Evaluation seeds: `0–9`.

| Method | Seed | Cost/order | Success rate | On-time rate | Depletion events | Delivered | Episode return |
|---|---:|---:|---:|---:|---:|---:|---:|
| DQN | 0 | 2.9734 | 0.8540 | 0.9220 | 0.0000 | 118.1 | 1373.9829 |
| DQN | 1 | 3.9212 | 0.8214 | 0.7546 | 0.0000 | 110.7 | 1098.3731 |
| DQN | 2 | 2.8351 | 0.8614 | 0.8904 | 0.0000 | 118.5 | 1376.7584 |
| Double DQN | 0 | 3.2569 | 0.8412 | 0.8981 | 0.0000 | 116.0 | 1303.2360 |
| Double DQN | 1 | 2.9574 | 0.8552 | 0.8735 | 0.0000 | 117.2 | 1335.0027 |
| Double DQN | 2 | 3.0662 | 0.8524 | 0.8626 | 0.0000 | 116.6 | 1311.3425 |
| Dueling DQN | 0 | 2.9104 | 0.8608 | 0.8408 | 0.0000 | 117.5 | 1326.0149 |
| Dueling DQN | 1 | 3.0064 | 0.8611 | 0.7513 | 0.0000 | 115.6 | 1245.0359 |
| Dueling DQN | 2 | 2.8843 | 0.8597 | 0.8865 | 0.0000 | 118.2 | 1365.0509 |

| Method | Mean cost/order | Interpretation |
|---|---:|---|
| DQN | 3.2432 ± 0.5912 | Beats greedy on average, but one seed is worse than greedy. |
| Double DQN | 3.0935 ± 0.1516 | Stable across seeds and consistently beats greedy. |
| Dueling DQN | **2.9337 ± 0.0643** | Best and most stable Role A method. |

### 3.6 Role A Ablation: Target Network On vs Off

Ablation method: Dueling DQN, seed 0.  
Evaluation seeds: `0–9`.

| Variant | Cost/order | Success rate | On-time rate | Depletion events | Delivered | Episode return |
|---|---:|---:|---:|---:|---:|---:|
| Target network ON | 2.9104 | 0.8608 | 0.8408 | 0.0000 | 117.5 | 1326.0149 |
| Target network OFF | 2.9528 | 0.8637 | 0.7491 | 0.0000 | 115.9 | 1250.7527 |

The target-network version achieved slightly lower cost per order and substantially better on-time rate and episode return. The no-target-network version still beat greedy nearest, but it was less stable. This supports the use of the target network for stabilizing bootstrapped value learning.

### 3.7 Final Role A Choice

Final Role A model:

```text
Dueling DQN with distance-aware battery safety mask
```

Recommended final weight:

```text
weights/role_a_dueling_seed2.pt
```

---

## 4. Role B — Policy-Based and Actor-Critic Track

### 4.1 Final Dispatcher Policy

The final Role B centralized dispatcher is a pairwise policy trained from demonstrations generated by the MILP rolling-horizon baseline. Instead of scoring all 169 discrete dispatcher actions independently, it scores each drone-order pair using shared features. This design improves generalization because the same scoring function is reused across all candidate assignments.

Although the final dispatcher is imitation-trained, the Role B development path included policy-gradient and actor-critic experiments before converging on the pairwise MILP-guided policy as the strongest dispatch solution.

### 4.2 Training Setup

- Training method: supervised imitation learning from MILP rolling-horizon decisions.
- Architecture: pairwise drone-order scoring network.
- Training runs: 3 independent runs.
- Training seed groups:
  - Run 0: seeds 0–74
  - Run 1: seeds 75–149
  - Run 2: seeds 150–224
- Held-out evaluation seeds: 300–319.

### 4.3 Held-Out Results

| Metric | MILP-guided pairwise policy |
|---|---:|
| Cost per order | 2.9684 ± 0.5976 |
| Success rate | 0.9052 ± 0.0171 |
| On-time rate | 0.9409 ± 0.0020 |
| Energy per order | 0.2210 ± 0.0009 |
| Depletion events | 1.9833 ± 0.3472 |
| Delivered orders | 123.5500 ± 2.6574 |
| Dropped orders | 13.5000 ± 2.3836 |
| Episode return | 1482.6644 ± 89.5978 |

### 4.4 Baseline Comparison

| Method | Cost per delivered order |
|---|---:|
| Greedy nearest | 3.9810 |
| MILP-guided pairwise policy | 2.9684 ± 0.5976 |

The final Role B policy reduced cost per delivered order by approximately 25% relative to greedy nearest on held-out seeds.

### 4.5 Development Path and DDPG Experiment

Role B development included:

1. REINFORCE baseline.
2. GAE-based actor-critic experiments.
3. Route-aware state representation.
4. Behavior cloning from greedy-nearest decisions.
5. Pairwise drone-order scoring architecture.
6. Battery-aware feature ablation.
7. MILP-guided imitation learning.
8. Three-run held-out evaluation.

The battery-aware feature ablation did not improve held-out performance, so the simpler eight-feature pairwise representation was retained.

For the continuous-control sub-environment `DroneControl-v0`, a DDPG implementation was created with an actor, critic, replay buffer, target actor/critic, Gaussian exploration noise, and soft target updates. The initial DDPG policy did not generalize reliably.

| Metric | Result |
|---|---:|
| Mean reward on seeds 100–104 | -1005.50 |
| Reward standard deviation | 933.14 |
| Success rate | 0.00 |

Diagnostic rollouts showed unstable heading correction and frequent battery depletion before reaching the target. This is reported as a failure-analysis result rather than a successful final controller.

---

## 5. Role C — Short-Horizon Look-Ahead Planning

### 5.1 Method

Role C implemented a short-horizon look-ahead dispatch policy for the centralized drone-delivery environment. The planner evaluates feasible drone-order assignments using three main signals:

1. **Pickup distance:** shorter travel distance to pickup is preferred.
2. **Order urgency:** orders closer to their SLA limit receive higher priority.
3. **Battery risk:** assignments likely to leave a drone with critically low state of charge are penalized.

In addition to immediate assignment quality, the planner estimates the effect of assigning a drone on future fleet capacity. This approximation considers expected incoming demand, remaining idle drones, and fleet battery levels.

The final Role C method is best understood as a lightweight short-horizon look-ahead planner rather than a full MCTS implementation or exact simulator rollout.

### 5.2 Experimental Setup

Final configuration:

```yaml
planning_depth: 1
forecast_weight: 1.0
```

Evaluation seeds:

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9
```

Primary baseline: `greedy_nearest`.

### 5.3 Main Results

| Method | Cost per order | Delivered orders | Dropped orders | On-time rate |
|---|---:|---:|---:|---:|
| Greedy nearest | 3.611 ± 1.129 | 121.80 ± 10.60 | 16.30 ± 5.95 | 0.924 ± 0.028 |
| Role C look-ahead | **3.484 ± 1.860** | 123.00 ± 13.99 | 16.00 ± 7.86 | 0.929 ± 0.019 |

The Role C planner reduced mean cost per delivered order from 3.611 to 3.484, an approximate 3.5% mean improvement over greedy nearest. It also increased average delivered orders, slightly reduced dropped orders, and improved on-time rate. The relatively large standard deviation means this should be interpreted as a mean improvement, not a statistically conclusive dominance claim.

If the figure files are present, the Role C cost comparison and depth ablation can be regenerated and viewed from:

```text
figures/role_c_vs_greedy_cost.png
figures/role_c_depth_ablation.png
```

### 5.4 Planning-Depth Ablation

| Planning depth | Cost per order |
|---|---:|
| 1 | 3.782 ± 1.179 |
| 2 | 3.931 ± 0.926 |
| 3 | 4.010 ± 0.572 |

Planning depth 1 performed best. Increasing depth worsened results in this implementation because the future fleet-capacity estimate is approximate; deeper planning accumulated more approximation error and produced overly conservative decisions.

### 5.5 Reproducibility

```powershell
python run_rollout.py --planner-config configs/rollout_role_c.yaml
python -m code.summarize_role_c
python -m code.plot_role_c_results
```

---

## 6. Offline RL Component

### 6.1 Dataset

The offline RL component used a mixed-quality static dataset generated from multiple behavior policies:

- random
- greedy
- noisy greedy
- Role A final policy
- Role B policy
- Role C policy

Dataset summary:

| Quantity | Value |
|---|---:|
| Transitions | 30,000 |
| Episodes | 214 |
| Observation dimension | 581 |
| Action dimension | 169 |
| Mean behavior episode return | 926.8404 |

The dataset was saved as:

```text
logs/offline_dataset.npz
```

### 6.2 Methods

We compared three offline methods:

1. **Behavioral cloning (BC):** supervised cross-entropy training to imitate dataset actions.
2. **Naive offline DQN:** offline Bellman backups without correcting distribution shift.
3. **CQL-style conservative DQN:** Bellman loss plus a conservative penalty of the form:
   ```text
   alpha * (logsumexp(Q(s, ·)) - Q(s, a_dataset))
   ```

Training setup:

| Quantity | Value |
|---|---:|
| Train steps | 2500 |
| Batch size | 128 |
| CQL alpha | 0.2 |
| Final BC loss | 1.4290 |
| Final naive DQN loss | 19.5918 |
| Final CQL loss | 8.5895 |

### 6.3 Offline RL Results

Evaluation seeds: `0–4`.

| Offline method | Cost/order | Success rate | On-time rate | Depletion events | Delivered | Dropped | Episode return |
|---|---:|---:|---:|---:|---:|---:|---:|
| Behavioral cloning | 10.6153 | 0.6570 | 0.9091 | 3.2 | 90.4 | 47.4 | 408.70 |
| Naive offline DQN | 645.2721 | 0.0239 | 0.9333 | 1.0 | 3.2 | 131.8 | -1989.51 |
| CQL-style DQN | 17.5961 | 0.9415 | 0.9742 | 8.0 | 24.8 | 1.6 | -63.33 |

### 6.4 Offline RL Analysis

Naive offline DQN displayed the classic offline Q-learning failure mode: it learned extremely poor behavior from a static dataset and delivered only 3.2 orders on average. Its cost per order became extremely high because it selected actions unsupported by the behavior data and overestimated their values. The high idle percentage and very low delivery count indicate that the learned policy was not effectively using the fleet.

Behavioral cloning was the best offline method in this experiment. It did not outperform the online learned policies, but it produced a functioning dispatcher by staying close to the behavior distribution.

The CQL-style conservative DQN improved substantially over naive offline DQN in cost and return, but it still did not beat behavioral cloning. It had high success and on-time rates for the few orders it delivered, but delivered too few orders and depleted all drones on average. This suggests that the conservative objective reduced some unsupported-action overestimation, but the simple implementation and limited training were not enough to produce a strong offline dispatcher.

---

## 7. Multi-Agent Component

### 7.1 Method

For the multi-agent requirement, we implemented shared-parameter Independent DQN on `DroneDispatchMA-v0`. Each drone acts as an independent agent using its own local observation, while all drones share a single Q-network. This improves sample efficiency relative to training a separate network per drone, but it does not remove the fundamental non-stationarity problem of independent learners.

The evaluated multi-agent policies were:

- `ma_random`: decentralized random policy.
- `ma_heuristic`: simple decentralized heuristic.
- `ma_learned`: shared-parameter IDQN policy.

Centralized references were evaluated separately on `DroneDispatch-v0` for qualitative comparison.

### 7.2 Multi-Agent Results

Evaluation seeds: `0–4`.

| Multi-agent policy | Episode return | Lost drones | Pending orders | Average SoC |
|---|---:|---:|---:|---:|
| MA random | 314.54 | 0.00 | 13.6 | 0.8955 |
| MA heuristic | **1707.97** | 1.00 | 11.4 | 0.5194 |
| MA learned IDQN | -1752.76 | 0.00 | 17.2 | 0.9518 |

Centralized reference results on `DroneDispatch-v0`:

| Centralized policy | Cost/order | Delivered | Dropped | Depletion events | Episode return |
|---|---:|---:|---:|---:|---:|
| Central random | 18.4979 | 40.0 | 21.2 | 8.0 | -153.93 |
| Central greedy nearest | 4.3090 | 120.0 | 19.8 | 3.6 | 1230.72 |
| Central MILP rolling | 4.2817 | 120.6 | 20.6 | 3.2 | 1251.90 |

### 7.3 Multi-Agent Analysis

The multi-agent experiment ran end-to-end, but the learned IDQN policy did not converge to a useful policy. The simple decentralized heuristic achieved the best multi-agent return, while learned IDQN produced negative returns. This is still an informative result: the independent-learner setting is non-stationary because each drone is learning while the behavior of the other drones changes. Each agent therefore observes a moving environment, which makes Q-learning unstable.

The comparison with centralized references is qualitative because `DroneDispatchMA-v0` exposes different observations, actions, and metrics from the centralized dispatcher. The centralized dispatcher has access to global information and can coordinate assignments directly, while the decentralized agents act from local views.

---

## 8. Ablations and Diagnostics

### 8.1 Role A Target-Network Ablation

| Variant | Cost/order | On-time rate | Delivered | Episode return |
|---|---:|---:|---:|---:|
| Dueling DQN, target network ON | 2.9104 | 0.8408 | 117.5 | 1326.01 |
| Dueling DQN, target network OFF | 2.9528 | 0.7491 | 115.9 | 1250.75 |

The target network improved stability and on-time performance.

### 8.2 Role C Planning-Depth Ablation

| Planning depth | Cost/order |
|---|---:|
| 1 | 3.782 ± 1.179 |
| 2 | 3.931 ± 0.926 |
| 3 | 4.010 ± 0.572 |

A shallow depth was best because deeper approximate planning accumulated more forecast error.

### 8.3 Role B Feature Ablation

Role B tested a battery-aware feature ablation. The battery-aware feature set did not improve held-out performance, so the simpler eight-feature pairwise representation was retained.

---

## 9. What Broke and How We Diagnosed It

### Role A

Initial DQN attempts failed because the pipeline was not aligned with the simulator API. The main issues were inconsistent observation flattening, creating the Q-network before knowing the true observation/action dimensions, not masking next-state target Q-values, and allowing legally valid but battery-unsafe assignments. The diagnostic signal was high drone depletion and worse-than-random cost per order. Adding target masking and distance-aware battery safety fixed the main failure mode.

### Role B

Policy-gradient and actor-critic experiments were less reliable than expected on the full dispatcher. The final successful dispatcher used MILP-guided imitation with a pairwise assignment architecture. The DDPG continuous-control experiment failed to generalize due to unstable heading correction and battery depletion.

### Role C

The planner improved greedy on mean cost but had high variance. The planning-depth ablation showed that deeper approximate look-ahead worsened performance because the future-capacity estimate was not accurate enough over longer horizons.

### Offline RL

Naive offline DQN demonstrated the expected offline Q-learning failure: severe distribution-shift and unsupported-action overestimation. CQL-style training reduced the failure relative to naive offline DQN, but behavioral cloning remained the strongest offline policy in our experiment.

### Multi-Agent

The shared-parameter IDQN implementation ran end-to-end but failed to converge. The decentralized heuristic outperformed learned IDQN. The main explanation is independent-learner non-stationarity and limited local observability.

---

## 10. Method-Origin Notes

- **DQN:** Based on deep Q-learning with replay buffer and target network. Chosen because Role A required value-based discrete-action dispatch.
- **Double DQN:** Uses online network action selection and target network evaluation to reduce maximization bias. Chosen to improve Q-learning stability.
- **Dueling DQN:** Separates state-value and action-advantage streams. Chosen because many states have similar action rankings and the value component can stabilize learning.
- **REINFORCE:** Classic Monte Carlo policy-gradient method. Used as an initial Role B policy-gradient baseline.
- **GAE / actor-critic:** Advantage estimation reduces variance in policy-gradient learning. Explored during Role B development.
- **DDPG:** Off-policy actor-critic method for continuous actions. Used for the `DroneControl-v0` continuous-control sub-environment.
- **Pairwise imitation policy:** Behavior cloning from MILP rolling-horizon decisions. Chosen because it generalized over drone-order pairs better than independent action scoring.
- **Short-horizon look-ahead planning:** Approximate rollout-style planning. Chosen for Role C because it adds future capacity awareness without expensive full search.
- **Behavioral cloning:** Supervised offline imitation of logged actions. Used as the offline RL baseline.
- **Naive offline DQN:** Included to demonstrate offline Q-learning failure under static data and out-of-distribution actions.
- **CQL-style conservative DQN:** Adds a conservative Q penalty to reduce over-optimism on unsupported actions.
- **Independent DQN for multi-agent RL:** Shared-parameter per-drone Q-learning. Chosen as a simple end-to-end decentralized baseline.

---

## 11. Reproducibility

### 11.1 Installation

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

### 11.2 Tests

```bash
python -m pytest tests -q
```

### 11.3 Main Result Table

```bash
python run_all.py --config configs/eval_standard.yaml --seeds 0,1,2,3,4,5,6,7,8,9
```

or, on Unix/Git Bash:

```bash
bash reproduce.sh configs/eval_standard.yaml 0,1,2,3,4,5,6,7,8,9
```

### 11.4 Role A

```bash
python "code/Role A DQN family/eval_role_a.py" --seeds 0,1,2,3,4,5,6,7,8,9 --include-ablation
```

### 11.5 Offline RL

```bash
python code/offline_rl_experiment.py --config configs/eval_standard.yaml --min-transitions 30000 --train-steps 2500 --eval-seeds 0,1,2,3,4
```

### 11.6 Multi-Agent

```bash
python code/train_idqn_ma.py train --episodes 100 --print-every 10 --weights-path weights/idqn_ma.pt --log-path logs/idqn_ma_training.csv
python code/train_idqn_ma.py eval --weights weights/idqn_ma.pt --seeds 0,1,2,3,4 --results-path logs/multi_agent_results.json
```

### 11.7 Role C

```powershell
python run_rollout.py --planner-config configs/rollout_role_c.yaml
python -m code.summarize_role_c
python -m code.plot_role_c_results
```

---

## 12. Final Assessment

The strongest centralized learned policies successfully beat the greedy nearest baseline on cost per delivered order. Role A's Dueling DQN and Role B's MILP-guided pairwise policy produced the strongest learned dispatcher results. Role C provided a lightweight planning policy with a smaller mean improvement over greedy nearest.

The main technical lesson was that algorithm choice alone was not enough. Correct action masking, battery feasibility, and reproducible evaluation were essential. The joint offline and multi-agent components also showed realistic limitations: naive offline Q-learning failed badly under distribution shift, and independent multi-agent Q-learning did not converge reliably under decentralized partial observations and non-stationarity.

Overall, the final repository contains working centralized learned policies, trained weights, logs, baseline comparisons, ablations, offline RL experiments, a multi-agent end-to-end experiment, and reproducibility commands.

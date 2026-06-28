# Role C — Short-Horizon Look-Ahead Planner

## Overview

Role C implements a short-horizon look-ahead dispatch policy for the drone delivery environment.

The final policy extends the provided `greedy_nearest` baseline with three additional decision signals:

* order urgency based on remaining SLA time,
* battery-risk estimation based on expected pickup and delivery distance,
* an approximate future fleet-capacity penalty based on expected incoming demand and available drone battery levels.

The selected final configuration uses a one-step planning depth because deeper planning horizons increased approximation error and produced higher average cost.

## Main Files

```text
code/role_c_planner.py          Initial deadline- and battery-aware heuristic
code/rollout_planner.py         Final Role C look-ahead planner
run_rollout.py                  Evaluates the Role C planner
run_baseline_full.py            Evaluates greedy baseline with per-seed results
code/summarize_role_c.py        Prints the experiment summary table
code/plot_role_c_results.py     Generates evaluation figures
configs/rollout_role_c.yaml     Final planner configuration
```

## Run the Final Role C Policy

Run the selected Role C configuration on the 10 evaluation seeds:

```powershell
python run_rollout.py --planner-config configs/rollout_role_c.yaml
```

The final configuration is:

```yaml
planning_depth: 1
forecast_weight: 1.0
evaluation_seeds: "0,1,2,3,4,5,6,7,8,9"
```

## Run the Greedy Baseline

```powershell
python run_baseline_full.py --policy greedy_nearest --seeds 0,1,2,3,4,5,6,7,8,9
```

## Reproduce the Result Summary

```powershell
python -m code.summarize_role_c
```

## Generate Figures

```powershell
python -m code.plot_role_c_results
```

The figures are saved under:

```text
figures/role_c_vs_greedy_cost.png
figures/role_c_depth_ablation.png
```

## Final 10-Seed Comparison

| Method            | Cost per Order | Delivered Orders | Dropped Orders |  On-Time Rate |
| ----------------- | -------------: | ---------------: | -------------: | ------------: |
| Greedy nearest    |  3.611 ± 1.129 |   121.80 ± 10.60 |   16.30 ± 5.95 | 0.924 ± 0.028 |
| Role C look-ahead |  3.484 ± 1.860 |   123.00 ± 13.99 |   16.00 ± 7.86 | 0.929 ± 0.019 |

The Role C policy reduced the mean cost per delivered order by approximately 3.5% relative to `greedy_nearest`.

## Planning-Depth Ablation

| Planning Depth | Cost per Order |
| -------------- | -------------: |
| 1              |  3.782 ± 1.179 |
| 2              |  3.931 ± 0.926 |
| 3              |  4.010 ± 0.572 |

A one-step look-ahead performed best. Increasing planning depth made the future-capacity estimate less reliable and increased the mean cost per delivered order.

## Notes

The result demonstrates a mean improvement over the greedy baseline across 10 seeds. However, the seed-level variation is substantial, so the result is reported as a mean improvement rather than a claim of statistically confirmed dominance.

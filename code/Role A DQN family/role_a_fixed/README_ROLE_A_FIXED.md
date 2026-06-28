# Role A diagnostic replacement

Use `code/Role A DQN family/role_a_dqn_fixed.py` rather than the three old train scripts while debugging.

Run from repo root:

```bash
python "code/Role A DQN family/role_a_dqn_fixed.py" train --algo double --seed 42 --episodes 300
python "code/Role A DQN family/role_a_dqn_fixed.py" eval --algo double --weights weights/role_a_double_seed42.pt --seeds 0,1,2
```

Main fixes:

1. One shared `flatten_obs()` is used in training, acting, and evaluation.
2. `obs_dim` and `action_dim` are measured before the agent/network is constructed.
3. Replay buffer stores `next_action_mask` so DQN targets do not maximize invalid actions.
4. Double DQN and Dueling DQN are selected with `--algo`; Dueling has a real value/advantage architecture.
5. Training logs real `cost_per_order` reconstructed from `env.unwrapped.stats`, not `info['metrics']['cost_per_order']` which the env does not expose.
6. Best model is saved by lowest cost per delivered order, not highest raw episode return.

Important repo issue: running Python from repo root currently makes `code/__init__.py` shadow Python's standard-library `code` module. This breaks `pytest` and may break `torch` imports. Since `code/__init__.py` is empty and not needed, delete it or run Role A scripts by file path as shown above so `torch` imports before the repo root is appended.

For the required 3-seed runs after sanity checks:

```bash
for s in 42 43 44; do
  python "code/Role A DQN family/role_a_dqn_fixed.py" train --algo dqn --seed $s --episodes 1500
  python "code/Role A DQN family/role_a_dqn_fixed.py" train --algo double --seed $s --episodes 1500
  python "code/Role A DQN family/role_a_dqn_fixed.py" train --algo dueling --seed $s --episodes 1500
done
```

First target is not immediately beating greedy. First target is: no crashes, nonzero delivered orders, nonzero logged `cost_per_order`, and stable eval over seeds.

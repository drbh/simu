# simu

This repo is a tiny exploration of RL models and teaches an ant to walk.

This is based on [gymnasium](https://github.com/Farama-Foundation/Gymnasium) and adds just implements some tiny ppo models to train the pre-existing ant environment.

### setup

zero to viewing a simulation in 5 commands

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
make apply-patch
python simu/display.py --policy_type simple
```

### training

```bash
python simu/train/ppo.py
usage: ppo.py [-h] --policy_type {simple,transformer,hybrid} [--env ENV]
              [--hidden_size HIDDEN_SIZE] [--num_episodes NUM_EPISODES]
              [--max_steps MAX_STEPS] [--learning_rate LEARNING_RATE]
```

you can train the model with the following command

```bash
python simu/train/ppo.py --policy_type simple
# Training on device: cpu
# Episode 10, Total Reward: -83.04, Avg Reward: -208.47
# Episode 20, Total Reward: -21.14, Avg Reward: -161.98
# ... (truncated)
# Episode 990, Total Reward: 1692.24, Avg Reward: 1588.09
# Episode 1000, Total Reward: 1727.82, Avg Reward: 1591.39
# Training completed at: 1719359098, Total training time: 367.23 seconds
#
# Final Evaluation - Mean Reward: 1746.32, Std Reward: 83.23
```

or the more advanced transformer model with the following command - it takes much much longer to train, doesn't seem to work as well and is likely not integrated into the policy correctly; as I don't really know much about reinforcement learning!

```bash
python simu/train/ppo.py --policy_type transformer
# Training on device: cpu
# Episode 10, Total Reward: -26.08, Avg Reward: -68.43
# Episode 20, Total Reward: -45.78, Avg Reward: -212.90
# ... (truncated)
# Episode 990, Total Reward: -252.39, Avg Reward: -328.63
# Episode 1000, Total Reward: -73.47, Avg Reward: -302.79
# Training completed at: 1719360458, Total training time: 1235.84 seconds
#
# Final Evaluation - Mean Reward: -38.75, Std Reward: 8.35
```

also you can try the hybrid model with the following command... but it's also trash

```bash
python simu/train/ppo.py --policy_type hybrid
# Training on device: cpu
# Episode 10, Total Reward: -23.88, Avg Reward: -308.28
# Episode 20, Total Reward: -84.84, Avg Reward: -256.40
# ... (truncated)
# Episode 990, Total Reward: -16.00, Avg Reward: -182.31
# Episode 1000, Total Reward: -6.83, Avg Reward: -169.03
# Training completed at: 1719360868, Total training time: 331.38 seconds
#
# Final Evaluation - Mean Reward: 312.62, Std Reward: 23.31
```

### viewing

you can open the 3d viewer with the following command and choose the trained model to apply to the ant

```bash
python simu/display.py --policy_type simple
```

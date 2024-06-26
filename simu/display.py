import gymnasium as gym
import torch
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Display trained PPO policy")
    parser.add_argument(
        "--policy_type",
        type=str,
        choices=["simple", "transformer", "hybrid"],
        required=True,
        help="Type of policy to use (simple or transformer)",
    )
    parser.add_argument(
        "--env", type=str, default="Ant-v4", help="Gymnasium environment to use"
    )
    parser.add_argument(
        "--hidden_size", type=int, default=256, help="Hidden layer size for networks"
    )
    parser.add_argument(
        "--num_episodes", type=int, default=5, help="Number of episodes to run"
    )
    parser.add_argument(
        "--max_steps", type=int, default=1000, help="Maximum steps per episode"
    )
    return parser.parse_args()


def load_policy(policy_type, input_dim, hidden_dim, output_dim, filepath):
    if policy_type == "simple":
        from simu.policies.simple import PolicyNetwork
    elif policy_type == "transformer":
        from simu.policies.transformer import PolicyNetwork
    elif policy_type == "hybrid":
        from simu.policies.hybrid import PolicyNetwork
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")

    policy = PolicyNetwork(input_dim, hidden_dim, output_dim).to(device)
    policy.load_state_dict(torch.load(filepath, map_location=device))
    policy.eval()
    return policy


def run_episode(env, policy, max_steps=1000):
    observation, _ = env.reset()
    total_reward = 0

    for _ in range(max_steps):
        state = torch.FloatTensor(observation).unsqueeze(0).to(device)
        with torch.no_grad():
            action_mean, _ = policy(state)

        action = action_mean.squeeze().cpu().numpy()
        action = np.clip(action, env.action_space.low, env.action_space.high)

        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        env.render()

        if terminated or truncated:
            break

    return total_reward


if __name__ == "__main__":
    args = parse_args()

    # Set up the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create the environment
    env = gym.make(args.env, render_mode="human")

    # Load the policy
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.shape[0]
    filepath = f"pretrained/ppo_{args.policy_type}_policy.pth"
    policy = load_policy(
        args.policy_type, input_dim, args.hidden_size, output_dim, filepath
    )

    # Run episodes
    for episode in range(args.num_episodes):
        episode_reward = run_episode(env, policy, args.max_steps)
        print(f"Episode {episode + 1} Reward: {episode_reward:.2f}")

    env.close()

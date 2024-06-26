import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
from collections import deque
import argparse
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Generic PPO training script")
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
        "--num_episodes", type=int, default=1000, help="Number of episodes to train"
    )
    parser.add_argument(
        "--max_steps", type=int, default=1000, help="Maximum steps per episode"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-4, help="Learning rate for optimizers"
    )
    return parser.parse_args()


def create_networks(policy_type, obs_dim, action_dim, hidden_size):
    if policy_type == "simple":
        from simu.policies.simple import PolicyNetwork, ValueNetwork
    elif policy_type == "transformer":
        from simu.policies.transformer import PolicyNetwork, ValueNetwork
    elif policy_type == "hybrid":
        from simu.policies.hybrid import PolicyNetwork, ValueNetwork
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")

    policy = PolicyNetwork(obs_dim, hidden_size, action_dim).to(device)
    value = ValueNetwork(obs_dim, hidden_size).to(device)
    return policy, value


def compute_gae(rewards, values, next_value, gamma, lam):
    returns = []
    gae = 0
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * next_value - values[step]
        gae = delta + gamma * lam * gae
        returns.insert(0, gae + values[step])
        next_value = values[step]
    return returns


def ppo_update(
    policy,
    value,
    policy_optimizer,
    value_optimizer,
    states,
    actions,
    log_probs,
    returns,
    advantages,
    ppo_epochs,
    batch_size,
    clip_epsilon,
):
    for _ in range(ppo_epochs):
        for start in range(0, len(states), batch_size):
            end = start + batch_size
            batch_states = states[start:end]
            batch_actions = actions[start:end]
            batch_log_probs = log_probs[start:end]
            batch_returns = returns[start:end]
            batch_advantages = advantages[start:end]

            mean, std = policy(batch_states)
            new_dist = Normal(mean, std)
            new_log_probs = new_dist.log_prob(batch_actions).sum(1)
            ratio = (new_log_probs - batch_log_probs).exp()

            surr1 = ratio * batch_advantages
            surr2 = (
                torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
                * batch_advantages
            )
            policy_loss = -torch.min(surr1, surr2).mean()

            value_pred = value(batch_states).squeeze()
            value_loss = nn.MSELoss()(value_pred, batch_returns)

            policy_optimizer.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            policy_optimizer.step()

            value_optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(value.parameters(), max_norm=0.5)
            value_optimizer.step()


def train(args):
    env = gym.make(args.env)
    policy, value = create_networks(
        args.policy_type,
        env.observation_space.shape[0],
        env.action_space.shape[0],
        args.hidden_size,
    )
    policy_optimizer = optim.Adam(policy.parameters(), lr=args.learning_rate)
    value_optimizer = optim.Adam(value.parameters(), lr=args.learning_rate)

    rewards_deque = deque(maxlen=100)
    gamma = 0.99
    clip_epsilon = 0.2
    ppo_epochs = 10
    batch_size = 64

    for episode in range(args.num_episodes):
        observation, _ = env.reset()
        states, actions, rewards, log_probs, values = [], [], [], [], []

        for step in range(args.max_steps):
            state = torch.FloatTensor(observation).unsqueeze(0).to(device)
            with torch.no_grad():
                action_mean, action_std = policy(state)
                value_pred = value(state)

            action_dist = Normal(action_mean, action_std)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action).sum(dim=-1)

            action_numpy = action.squeeze().cpu().numpy()
            action_numpy = np.clip(
                action_numpy, env.action_space.low, env.action_space.high
            )
            next_observation, reward, terminated, truncated, _ = env.step(action_numpy)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value_pred.squeeze())

            observation = next_observation

            if terminated or truncated:
                break

        next_value = value(
            torch.FloatTensor(observation).unsqueeze(0).to(device)
        ).detach()
        returns = compute_gae(rewards, values, next_value, gamma, 0.95)
        returns = torch.tensor(returns).float().to(device)
        values = torch.stack(values)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states = torch.cat(states)
        actions = torch.cat(actions)
        log_probs = torch.cat(log_probs)

        ppo_update(
            policy,
            value,
            policy_optimizer,
            value_optimizer,
            states,
            actions,
            log_probs,
            returns,
            advantages,
            ppo_epochs,
            batch_size,
            clip_epsilon,
        )

        total_reward = sum(rewards)
        rewards_deque.append(total_reward)
        avg_reward = np.mean(rewards_deque)

        if (episode + 1) % 10 == 0:
            print(
                f"Episode {episode + 1}, Total Reward: {total_reward:.2f}, Avg Reward: {avg_reward:.2f}"
            )

        if avg_reward > 3500:
            print(f"Solved in {episode + 1} episodes!")
            break

    return policy


def evaluate(env, policy, n_episodes=10, max_steps=1000):
    eval_rewards = []
    for _ in range(n_episodes):
        observation, _ = env.reset()
        episode_reward = 0
        for _ in range(max_steps):
            state = torch.FloatTensor(observation).unsqueeze(0).to(device)
            with torch.no_grad():
                action_mean, _ = policy(state)
            action = action_mean.squeeze().cpu().numpy()
            action = np.clip(action, env.action_space.low, env.action_space.high)
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break
        eval_rewards.append(episode_reward)
    return np.mean(eval_rewards), np.std(eval_rewards)


if __name__ == "__main__":
    args = parse_args()
    print(f"Training on device: {device}")
    start_time = time.time()
    trained_policy = train(args)
    end_time = time.time()
    training_time = end_time - start_time  # in seconds
    print(
        f"Training completed at: {int(end_time)}, Total training time: {training_time:.2f} seconds"
    )

    env = gym.make(args.env)
    mean_reward, std_reward = evaluate(env, trained_policy)
    print(
        f"\nFinal Evaluation - Mean Reward: {mean_reward:.2f}, Std Reward: {std_reward:.2f}"
    )

    torch.save(
        trained_policy.state_dict(), f"pretrained/ppo_{args.policy_type}_policy.pth"
    )
    env.close()

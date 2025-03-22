# test.py

import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from environment import Environment, Point  # Import Environment and Point classes
from model import Actor, Critic, t  # Import Actor, Critic, and tensor transformation function

# Setting random seeds for reproducibility
seed = 10
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

class TestAgent:
    def __init__(self, env):
        self.env = env  # Store the environment instance

        # Get state dimension and number of actions from the environment
        state_dim = env.state().shape[0]
        actions = env.actions()
        n_actions = len(actions)

        # Initialize the Actor and Critic neural networks
        self.actor = Actor(state_dim, n_actions)
        self.critic = Critic(state_dim)

        # Load the saved model weights for Actor and Critic from the 'Model' folder
        self.actor.load_state_dict(torch.load('Model/actor_model.pth'))
        self.critic.load_state_dict(torch.load('Model/critic_model.pth'))

        # Set the models to evaluation mode
        self.actor.eval()
        self.critic.eval()

    def test(self, num_episodes=10):
        """
        Run multiple test episodes to evaluate the performance of the trained models.

        Parameters:
            num_episodes (int): Number of episodes to run for testing.

        Returns:
            rewards (list): List of total rewards per episode.
        """
        rewards = []  # List to store total rewards for each episode

        for episode in range(num_episodes):
            state = self.env.reset()  # Reset the environment and get the initial state
            done = False
            total_reward = 0

            while not done:
                probs = self.actor(t(state))  # Get action probabilities from Actor
                dist = torch.distributions.Categorical(probs=probs)
                action = dist.sample()  # Sample an action

                next_state, reward, done = self.env.step(action.detach().numpy())  # Take action and get the results
                total_reward += reward

                state = next_state  # Update the current state

            rewards.append(total_reward)  # Append the total reward of this episode to the list
            print(f"Episode {episode + 1}: Total Reward = {total_reward}")

        return rewards

    def plot_rewards(self, rewards):
        """
        Plot the rewards over episodes.

        Parameters:
            rewards (list): List of total rewards per episode.

        Returns:
            None
        """
        plt.figure()
        plt.plot(rewards, marker='o')
        plt.title("Total Reward per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    env = Environment(render=True)  # Create an instance of the environment
    test_agent = TestAgent(env)  # Create an instance of TestAgent with the environment
    rewards = test_agent.test(num_episodes=20)  # Run the test for 10 episodes
    test_agent.plot_rewards(rewards)  # Plot the rewards

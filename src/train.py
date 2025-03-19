import random
from IPython import display
import numpy as np
import torch

# Set seeds for reproducibility
seed = 10
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# If using CUDA, also set the seeds for it
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU

# Ensure deterministic behavior on GPU (if applicable)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import matplotlib.pyplot as plt  # Import matplotlib for plotting

from environment import Environment, Point
from memory import Step_Memory as Memory  # Import Memory for storing experiences
from model import (  # Import the Model containing Actor and Critic classes
    Actor,
    Critic,
    t,
)

class Agent:
    def __init__(self, env):
        """
        Initialize the Agent with default parameters.

        Parameters:
            env (Environment): An instance of the Environment class.

        Returns:
            None
        """
        self.env = env  # Store the environment instance
        self.gamma = 0.99  # Set the discount factor for future rewards

        # Get state dimension and number of actions from the environment
        state_dim = env.state().shape[0]
        actions = env.actions()
        n_actions = len(actions)
        # Initialize the Actor and Critic neural networks
        self.actor = Actor(state_dim, n_actions)
        self.critic = Critic(state_dim)

        # Set up optimizers for the Actor and Critic networks
        self.adam_actor = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        self.adam_critic = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        # Initialize memory for storing experiences
        self.memory = Memory()
        self.max_steps = 50  # Set maximum steps per episode

    def train(self, memory, q_val):
        """
        Train the Actor and Critic networks using stored experiences.

        Parameters:
            memory (Memory): The memory containing experiences.
            q_val (float): The estimated Q-value for the next state.

        Returns:
            None
        """
        values = torch.stack(memory.values)  # Stack values from memory
        q_vals = np.zeros((len(memory), 1))  # Initialize Q-values array

        # Iterate through the stored experiences in reverse order
        for i, (_, _, reward, done) in enumerate(memory.reversed()):
            # Compute the target Q-value based on the reward and the next estimated Q-value
            q_val = reward + self.gamma * q_val * (1.0 - done)
            # Store the calculated Q-value in reverse order
            q_vals[len(memory) - 1 - i] = q_val

        # Calculate the advantage function
        advantage = torch.Tensor(q_vals) - values

        # Compute the loss for the Critic
        critic_loss = advantage.pow(2).mean()
        # Zero the gradients for the Critic optimizer
        self.adam_critic.zero_grad()
        # Backpropagate the critic loss
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        # Update the Critic's parameters
        self.adam_critic.step()

        # Compute the loss for the Actor
        actor_loss = (-torch.stack(memory.log_probs) * advantage.detach()).mean()
        # Zero the gradients for the Actor optimizer
        self.adam_actor.zero_grad()
        # Backpropagate the actor loss
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        # Update the Actor's parameters
        self.adam_actor.step()
    def run(self):
        """
        training loop for the agent.
        """
        episode_rewards = []  # List to store total rewards for each episode
        TRV = []
        mean_rewards = []

        # Main training loop for multiple episodes
        for episode in range(2000):
            done = False  # Flag to indicate if the episode is done
            total_reward = 0  # Initialize total reward for the episode
            state = env.reset()  # Reset the environment to get the initial state
            steps = 0  # Initialize step counter

            # Loop until the episode is done
            while not done:

                # Get action probabilities from the Actor network
                probs = self.actor(t(state))

                # Create a categorical distribution for sampling actions
                dist = torch.distributions.Categorical(probs=probs)
                # print(probs)
                # Sample an action from the distribution
                action = dist.sample()
                # print(action)
                # Take a step in the environment using the sampled action
                next_state, reward, done = self.env.step(action.detach().data.numpy())

                total_reward += (
                    reward  # Accumulate the reward received from the environment
                )

                steps += 1  # Increment the step counter

                # Store the log probability of the action, the value from the critic, the reward, and done flag in memory
                self.memory.add(
                    dist.log_prob(action), self.critic(t(state)), reward, done
                )

                # Update the current state to the next state
                state = next_state
                # Train the actor and critic if the episode is done or the number of steps exceeds max_steps
                if done or (steps > self.max_steps):
                    done = True
                    last_q_val = (
                        self.critic(t(next_state)).detach().data.numpy()
                    )  
                    # Get estimated value for the next state
                    self.train(
                        self.memory, last_q_val
                    )  # Train the networks using the stored memory
                    self.memory.clear()  # Clear the memory after training
                    # env.reset()

            # Store the total reward for this episode
            episode_rewards.append(
                total_reward
            )  
            TRV.append(total_reward)

            # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
            # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)

            # Calculate mean reward for the last 10 episodes
            if len(episode_rewards) > 20:
                mean_reward = np.mean(episode_rewards[-20:])
                mean_rewards.append(mean_reward)

            # Save the model weights at the end of each episode
            self.actor.save("actor_model.pth")
            self.critic.save("critic_model.pth")

            display.clear_output(wait=True)
            display.display(plt.gcf())
            plt.clf()
            plt.scatter(np.arange(len(episode_rewards)), TRV, s=2, label="Total Reward")

            # Plot mean rewards
            if mean_rewards:
                plt.plot(
                    np.arange(len(mean_rewards)),
                    mean_rewards,
                    color="green",
                    label="Mean Reward (Last 20 Episodes)",
                )

            plt.scatter(np.arange(len(episode_rewards)), TRV, s=2)
            plt.title("Total reward per episode")
            plt.ylabel("reward")
            plt.xlabel("episode")
            plt.show(block=False)
            plt.pause(.1)


if __name__ == "__main__":
    env = Environment(render=True)  
    agent = Agent(env)
    agent.run()

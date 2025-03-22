import numpy as np

# Memory class to store results from networks, avoiding redundant calculations of operations from states.
class Step_Memory():
    def __init__(self):
        """
        Initializes the Memory object with empty lists to store:
        - log_probs: Logarithm of probabilities from the network's output.
        - values: Predicted values (typically state values).
        - rewards: Rewards received after taking actions.
        - dones: Flags indicating the end of an episode.
        """
        # Initialize an empty list to store log probabilities of actions
        self.log_probs = []
        # Initialize an empty list to store value predictions
        self.values = []

        # Initialize an empty list to store rewards received
        self.rewards = []
        # Initialize an empty list to store done flags indicating the end of episodes
        self.dones = []



    def add(self, log_prob, value, reward, done):
        """
        Adds a new set of data to the memory.

        Parameters:
        - log_prob: The log probability of the action taken.
        - value: The predicted value associated with the state.
        - reward: The reward received after the action.
        - done: A boolean flag indicating whether the episode is finished.

        This method appends each of these parameters to their respective lists.
        """
        # Append the log probability to the log_probs list
        self.log_probs.append(log_prob)
        # Append the value to the values list
        self.values.append(value)

        # Append the reward to the rewards list
        self.rewards.append(reward)
        # Append the done flag to the dones list
        self.dones.append(done)

    def clear(self):
        """
        Clears all the stored data from the memory.

        This method empties the lists that store log_probs, values, rewards, and dones.
        """
        # Clear all stored log probabilities
        self.log_probs.clear()
        # Clear all stored values
        self.values.clear()
        # Clear all stored rewards
        self.rewards.clear()
        # Clear all stored done flags
        self.dones.clear()

    def _zip(self):
        """
        Creates an iterator that combines the stored data (log_probs, values, rewards, dones) into tuples.

        Returns:
        - An iterator where each item is a tuple (log_prob, value, reward, done).
        """
        # Zip the lists log_probs, values, rewards, and dones into an iterator of tuples
        return zip(self.log_probs,
                   self.values,
                   self.rewards,
                   self.dones)

    def __iter__(self):
        """
        Makes the Memory class iterable by returning an iterator over the stored data.

        Returns:
        - An iterator where each iteration yields a tuple (log_prob, value, reward, done).
        """
        # Iterate over the zipped data (log_prob, value, reward, done)
        for data in self._zip():
            # Return each tuple of (log_prob, value, reward, done)
            return data

    def reversed(self):
        """
        Provides a generator to iterate over the stored data in reverse order.

        Yields:
        - A tuple (log_prob, value, reward, done) in reverse order of their addition.
        """
        # Iterate over the reversed list of tuples (log_prob, value, reward, done)
        for data in list(self._zip())[::-1]:
            # Yield each tuple from the reversed list
            yield data

    def __len__(self):
        """
        Returns the number of entries stored in the memory.

        Returns:
        - An integer representing the number of elements in the rewards list,
          which corresponds to the number of stored transitions.
        """
        # Return the length of the rewards list, which represents the number of stored transitions
        return len(self.rewards)


class Episode_Memory():
    def __init__(self):
        # Initialize an empty list to store state
        self.states =[np.array(
            [float(20), float(200), (3),
             float(0), float(0),
            1,
            0,
            1,
            0
            ], dtype=np.float32
        )]
        self.scores = []

    def add_state(self, state):
        # Append the reward to the rewards list
        self.states.append(state)

    def add_score(self, score):
        self.scores.append(score)

    def Episode_Clear(self):
        self.states.clear()
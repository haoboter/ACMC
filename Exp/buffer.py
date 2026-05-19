import numpy as np
import os

"""
Replay buffer for off-policy RL training.
"""


class ReplayBuffer:
    """Fixed-size circular replay buffer backed by NumPy arrays."""

    def __init__(self, max_size, input_shape, n_actions):
        """Initialize the replay buffer storage.

        Args:
            max_size: Maximum number of transitions to keep (capacity).
            input_shape: Shape tuple for a single state observation (e.g., (8,)).
            n_actions: Action dimension. Actions are stored as shape (n_actions,).
        """
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, state_, done):
        """Store one transition in the circular buffer."""
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        """Sample a random minibatch.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            states, actions, rewards, next_states, dones
        """
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones

    def save(self, filename):
        """Persist buffer arrays to a compressed `.npz` file."""
        np.savez_compressed(filename,
                            state_memory=self.state_memory,
                            new_state_memory=self.new_state_memory,
                            action_memory=self.action_memory,
                            reward_memory=self.reward_memory,
                            terminal_memory=self.terminal_memory,
                            mem_cntr=self.mem_cntr)
        print(f"Memory successfully saved to {filename}.")

    def load(self, filename):
        """Load buffer arrays from a `.npz` file.
        """
        if os.path.exists(filename):
            data = np.load(filename)
            self.state_memory = data['state_memory']
            self.new_state_memory = data['new_state_memory']
            self.action_memory = data['action_memory']
            self.reward_memory = data['reward_memory']
            self.terminal_memory = data['terminal_memory']
            self.mem_cntr = data['mem_cntr'].item()

            print(f"Loaded memory with {self.mem_cntr} transitions.")
        else:
            print(f"File {filename} not found. Initializing new replay buffer.")
            self.state_memory = np.zeros_like(self.state_memory)
            self.new_state_memory = np.zeros_like(self.new_state_memory)
            self.action_memory = np.zeros_like(self.action_memory)
            self.reward_memory = np.zeros_like(self.reward_memory)
            self.terminal_memory = np.zeros_like(self.terminal_memory)
            self.mem_cntr = 0
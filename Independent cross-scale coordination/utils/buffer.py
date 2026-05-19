"""Fixed-size circular replay buffer; ``save``/``load`` use np.savez_compressed (any path suffix)."""
import numpy as np
import os

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones

    def save(self, filename):
        np.savez_compressed(filename,
                            state_memory=self.state_memory,
                            new_state_memory=self.new_state_memory,
                            action_memory=self.action_memory,
                            reward_memory=self.reward_memory,
                            terminal_memory=self.terminal_memory,
                            mem_cntr=self.mem_cntr)

    def load(self, filename):
        if os.path.exists(filename):
            try:
                data = np.load(filename)
                self.state_memory = data['state_memory']
                self.new_state_memory = data['new_state_memory']
                self.action_memory = data['action_memory']
                self.reward_memory = data['reward_memory']
                self.terminal_memory = data['terminal_memory']
                self.mem_cntr = data['mem_cntr'].item()
                print(f"Replay buffer loaded successfully with {self.mem_cntr} transitions")
            except Exception as e:
                print(f"Warning: failed to load replay buffer file {filename}; the file may be corrupted")
                print(f"Error details: {e}")
                print("Experience collection will restart from scratch")
        else:
            print(f"File {filename} does not exist; experience collection will start from scratch")

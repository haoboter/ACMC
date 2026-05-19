"""Actor, twin critics, and value / target_value nets; checkpoints under ``tmp/agent``."""
import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions, layer_sizes=[128, 256, 256, 128],
                 name='critic', chkpt_dir='tmp/agent'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.layer_sizes = layer_sizes
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.layer_sizes[0])
        self.ln1 = nn.LayerNorm(self.layer_sizes[0])
        self.fc2 = nn.Linear(self.layer_sizes[0], self.layer_sizes[1])
        self.ln2 = nn.LayerNorm(self.layer_sizes[1])
        self.fc3 = nn.Linear(self.layer_sizes[1], self.layer_sizes[2])
        self.ln3 = nn.LayerNorm(self.layer_sizes[2])
        self.fc4 = nn.Linear(self.layer_sizes[2], self.layer_sizes[3])
        self.q = nn.Linear(self.layer_sizes[3], 1)


        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        state = state.to(self.device)
        action = action.to(self.device)
        action_value = self.fc1(T.cat([state, action], dim=1))
        action_value = self.ln1(action_value)
        action_value = F.relu(action_value)

        action_value = self.fc2(action_value)
        action_value = self.ln2(action_value)
        action_value = F.relu(action_value)

        action_value = self.fc3(action_value)
        action_value = self.ln3(action_value)
        action_value = F.relu(action_value)

        action_value = self.fc4(action_value)
        action_value = F.relu(action_value)

        q = self.q(action_value)

        return q

    def save_checkpoint_best(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        T.save(self.state_dict(), (self.checkpoint_file + '_best'))

    def save_checkpoint_last(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        T.save(self.state_dict(), (self.checkpoint_file + '_last'))

    def load_checkpoint_best(self):
        self.load_state_dict(T.load(self.checkpoint_file + '_best', map_location='cpu'))

    def load_checkpoint_last(self):
        self.load_state_dict(T.load(self.checkpoint_file + '_last', map_location='cpu'))
    
    def save_checkpoint_custom(self, custom_name):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        custom_file = os.path.join(self.checkpoint_dir, custom_name)
        T.save(self.state_dict(), custom_file)
        print(f'Saved {self.name} model to: {custom_file}')

class ValueNetwork(nn.Module):
    def __init__(self, beta, input_dims, layer_sizes=[128, 256, 256, 128],
                 name='value', chkpt_dir='tmp/agent'):
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.layer_sizes = layer_sizes
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.fc1 = nn.Linear(*self.input_dims, self.layer_sizes[0])
        self.ln1 = nn.LayerNorm(self.layer_sizes[0])
        self.fc2 = nn.Linear(self.layer_sizes[0], self.layer_sizes[1])
        self.ln2 = nn.LayerNorm(self.layer_sizes[1])
        self.fc3 = nn.Linear(self.layer_sizes[1], self.layer_sizes[2])
        self.ln3 = nn.LayerNorm(self.layer_sizes[2])
        self.fc4 = nn.Linear(self.layer_sizes[2], self.layer_sizes[3])
        self.v = nn.Linear(self.layer_sizes[3], 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state = state.to(self.device)
        state_value = self.fc1(state)
        state_value = self.ln1(state_value)
        state_value = F.relu(state_value)

        state_value = self.fc2(state_value)
        state_value = self.ln2(state_value)
        state_value = F.relu(state_value)

        state_value = self.fc3(state_value)
        state_value = self.ln3(state_value)
        state_value = F.relu(state_value)

        state_value = self.fc4(state_value)
        state_value = F.relu(state_value)

        v = self.v(state_value)

        return v

    def save_checkpoint_best(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        T.save(self.state_dict(), (self.checkpoint_file + '_best'))

    def save_checkpoint_last(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        T.save(self.state_dict(), (self.checkpoint_file + '_last'))

    def load_checkpoint_best(self):
        self.load_state_dict(T.load(self.checkpoint_file + '_best', map_location='cpu'))

    def load_checkpoint_last(self):
        self.load_state_dict(T.load(self.checkpoint_file + '_last', map_location='cpu'))
    
    def save_checkpoint_custom(self, custom_name):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        custom_file = os.path.join(self.checkpoint_dir, custom_name)
        T.save(self.state_dict(), custom_file)
        print(f'Saved {self.name} model to: {custom_file}')

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, max_action, min_action, n_actions=4,
                 layer_sizes=[128, 256, 256, 128], name='actor', chkpt_dir='tmp/agent'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        self.action_scale = (max_action - min_action) / 2
        self.action_bias = (max_action + min_action) / 2
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(*self.input_dims, layer_sizes[0])
        self.ln1 = nn.LayerNorm(layer_sizes[0])
        self.fc2 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.ln2 = nn.LayerNorm(layer_sizes[1])
        self.fc3 = nn.Linear(layer_sizes[1], layer_sizes[2])
        self.ln3 = nn.LayerNorm(layer_sizes[2])
        self.fc4 = nn.Linear(layer_sizes[2], layer_sizes[3])
        self.mu = nn.Linear(layer_sizes[3], n_actions)
        self.sigma = nn.Linear(layer_sizes[3], n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state = state.to(self.device)
        prob = self.fc1(state)
        prob = self.ln1(prob)
        prob = F.relu(prob)

        prob = self.fc2(prob)
        prob = self.ln2(prob)
        prob = F.relu(prob)

        prob = self.fc3(prob)
        prob = self.ln3(prob)
        prob = F.relu(prob)

        prob = self.fc4(prob)
        prob = F.relu(prob)

        mu = self.mu(prob)
        sigma = self.sigma(prob)

        sigma = T.clamp(sigma, min=self.reparam_noise, max=1)

        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        action = T.tanh(actions) * T.tensor(self.action_scale).to(self.device) + T.tensor(self.action_bias).to(self.device)
        log_probs = probabilities.log_prob(actions)
        log_probs -= T.log(1 - action.pow(2) + self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs

    def save_checkpoint_best(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        T.save(self.state_dict(), (self.checkpoint_file + '_best'))

    def save_checkpoint_last(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        T.save(self.state_dict(), (self.checkpoint_file + '_last'))

    def load_checkpoint_best(self):
        self.load_state_dict(T.load(self.checkpoint_file + '_best', map_location='cpu'))

    def load_checkpoint_last(self):
        self.load_state_dict(T.load(self.checkpoint_file + '_last', map_location='cpu'))
    
    def save_checkpoint_custom(self, custom_name):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        custom_file = os.path.join(self.checkpoint_dir, custom_name)
        T.save(self.state_dict(), custom_file)
        print(f'Saved {self.name} model to: {custom_file}')

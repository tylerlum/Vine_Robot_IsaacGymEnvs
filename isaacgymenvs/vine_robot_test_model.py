# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# Must import rl_games before torch
from rl_games.algos_torch import model_builder, torch_ext
from rl_games.torch_runner import _restore

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from gym import spaces

# %%
# PARAMETERS
# Input/Output Sizes
N_ACTIONS = 1
N_OBS = 19

# Action space
RAIL_FORCE_RANGE = (-10.0, 10.0)
U_RANGE = (-0.1, 3.0)

# File paths
FOLDER = "/home/tylerlum/github_repos/Vine_Robot_IsaacGymEnvs/isaacgymenvs"
REL_CHECKPOINT_PATH = "runs/Vine5LinkMovingBase/nn/last_Vine5LinkMovingBase_ep_500_rew_182.01628.pth"  # Specific model weights
REL_CONFIG_PATH = "runs/Vine5LinkMovingBase/2022-11-07_14-50-31_rlg_config_dict.pkl"  # Config of the model
ABS_CHECKPOINT_PATH = os.path.join(FOLDER, REL_CHECKPOINT_PATH)
ABS_CONFIG_PATH = os.path.join(FOLDER, REL_CONFIG_PATH)

# %%


def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action = action * d + m
    return scaled_action


class BasePlayer(object):
    def __init__(self, params):
        self.config = config = params['config']
        self.load_networks(params)
        self.clip_actions = config.get('clip_actions', True)
        self.value_size = 1
        self.action_space = spaces.Box(np.ones(N_ACTIONS) * -1., np.ones(N_ACTIONS) * 1.)
        self.num_agents = 1

        self.observation_space = spaces.Box(np.ones(N_OBS) * -np.Inf, np.ones(N_OBS) * np.Inf)
        self.obs_shape = self.observation_space.shape

        self.states = None
        self.is_determenistic = True  # TODO: What should this be?
        self.device_name = self.config.get('device_name', 'cuda')
        self.device = torch.device(self.device_name)

    def load_networks(self, params):
        builder = model_builder.ModelBuilder()
        self.config['network'] = builder.load(params)

    def get_weights(self):
        weights = {}
        weights['model'] = self.model.state_dict()
        return weights

    def set_weights(self, weights):
        self.model.load_state_dict(weights['model'])
        if self.normalize_input and 'running_mean_std' in weights:
            self.model.running_mean_std.load_state_dict(weights['running_mean_std'])


class PpoPlayerContinuous(BasePlayer):
    def __init__(self, params):
        BasePlayer.__init__(self, params)
        self.network = self.config['network']
        self.actions_num = self.action_space.shape[0]
        self.actions_low = torch.from_numpy(self.action_space.low.copy()).float().to(self.device)
        self.actions_high = torch.from_numpy(self.action_space.high.copy()).float().to(self.device)
        self.mask = [False]

        self.normalize_input = self.config['normalize_input']
        self.normalize_value = self.config.get('normalize_value', False)

        obs_shape = self.obs_shape
        config = {
            'actions_num': self.actions_num,
            'input_shape': obs_shape,
            'num_seqs': self.num_agents,
            'value_size': self.value_size,
            'normalize_value': self.normalize_value,
            'normalize_input': self.normalize_input,
        }
        self.model = self.network.build(config)
        self.model.to(self.device)
        self.model.eval()

    def get_action(self, obs, is_determenistic=False):
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs': obs,
            'rnn_states': self.states
        }
        with torch.no_grad():
            res_dict = self.model(input_dict)
        mu = res_dict['mus']
        action = res_dict['actions']
        self.states = res_dict['rnn_states']
        if is_determenistic:
            current_action = mu
        else:
            current_action = action

        if self.clip_actions:
            return rescale_actions(self.actions_low, self.actions_high, torch.clamp(current_action, -1.0, 1.0))
        else:
            return current_action

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.model.load_state_dict(checkpoint['model'])
        if self.normalize_input and 'running_mean_std' in checkpoint:
            self.model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])


# %%
class VineRobotControlModel(nn.Module):
    def __init__(self, config_path, checkpoint_path, x_range, u_range):
        super().__init__()
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path

        # Action space scaling
        self.rail_force_min, self.rail_force_max = x_range
        self.u_min, self.u_max = u_range

        # Create rl_games player with config and checkpoint
        with open(config_path, "rb") as f:
            cfg = pickle.load(f)
        self.rl_games_player = PpoPlayerContinuous(cfg['params'])
        _restore(self.rl_games_player, {"checkpoint": checkpoint_path})

    def get_action(self, q, qd, tip_pos, tip_vel, target_pos, target_vel):
        # Inputs all have shape (xi,), where xi is the specific vector size
        # Assumes inputs are torch tensors, and inputs and model are on the same device
        # Returns torch tensor on same device
        obs = torch.cat([q, qd, tip_pos, tip_vel, target_pos, target_vel])[None, ...].to(q.device)  # (1, sum(xi))
        action = self.forward(obs)[0]  # (1, action_dim)  => (action_dim,)

        if torch.numel(action) == 1:
            return self.rescale(action, self.u_min, self.u_max)
        elif torch.numel(action) == 2:
            action[0] = self.rescale(action[0], self.rail_force_min, self.rail_force_max)
            action[1] = self.rescale(action[1], self.u_min, self.u_max)
        return action

    def forward(self, obs):
        return self.rl_games_player.get_action(obs)

    def rescale(self, x, low, high):
        return (x + 1) * (high - low) / 2 + low


# %%
# Create VineRobotControlModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vine_robot_control_model = VineRobotControlModel(
    ABS_CONFIG_PATH, ABS_CHECKPOINT_PATH, RAIL_FORCE_RANGE, U_RANGE).to(device)

# %%
print(f"vine_robot_control_model: {vine_robot_control_model}")
print(f"vine_robot_control_model.rl_games_player: {vine_robot_control_model.rl_games_player}")

# %%
# Test on inputs
q = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0]).to(device)
qd = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0]).to(device)
tip_pos = torch.tensor([0.0, 0.0, 0.0]).to(device)
tip_vel = torch.tensor([0.0, 0.0, 0.0]).to(device)
target_pos = torch.tensor([0.0, 0.0, 0.0]).to(device)
target_vel = torch.tensor([0.0, 0.0, 0.0]).to(device)

action = vine_robot_control_model.get_action(q, qd, tip_pos, tip_vel, target_pos, target_vel)
x = action[0]
u = action[1]
print(f"With inputs q = {q}, qd = {qd}, tip_pos = {tip_pos}, tip_vel = {tip_vel}, target_pos = {target_pos}, target_vel = {target_vel}, the model outputs x = {x} and u = {u}")

# %%
vine_robot_control_model.nn_model

# %%

# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import torch
import math

from isaacgym import gymutil, gymtorch, gymapi
from .base.vec_task import VecTask
import wandb

# CONSTANTS (RARELY CHANGE)
NUM_STATES = 13  # xyz, quat, v_xyz, w_xyz
NUM_XYZ = 3
LENGTH_RAIL = 0.8
LENGTH_PER_LINK = 0.0885
N_REVOLUTE_DOFS = 5
N_PRESSURE_ACTIONS = 1
START_POS_IDX, END_POS_IDX = 0, 3
START_QUAT_IDX, END_QUAT_IDX = 3, 7
START_LIN_VEL_IDX, END_LIN_VEL_IDX = 7, 10
START_ANG_VEL_IDX, END_ANG_VEL_IDX = 10, 13

# PARAMETERS (OFTEN CHANGE)
USE_MOVING_BASE = False

U_MIN, U_MAX = -0.1, 3.0
RAIL_FORCE_SCALE = 1000.0

# Observations
NO_VEL_IN_OBS = False

# Rewards
REWARD_NAMES = ["Dense", "Const Negative", "Position Success",
                "Velocity Success", "Velocity", "Rail Force Control", "U Control"]
DENSE_REWARD_WEIGHT = 0.0
CONST_NEGATIVE_REWARD_WEIGHT = 0.0
POSITION_SUCCESS_REWARD_WEIGHT = 0.0
VELOCITY_SUCCESS_REWARD_WEIGHT = 0.0
VELOCITY_REWARD_WEIGHT = 1.0
RAIL_FORCE_CONTROL_REWARD_WEIGHT = 0.1
U_CONTROL_REWARD_WEIGHT = 0.1
REWARD_WEIGHTS = [DENSE_REWARD_WEIGHT, CONST_NEGATIVE_REWARD_WEIGHT, POSITION_SUCCESS_REWARD_WEIGHT,
                  VELOCITY_SUCCESS_REWARD_WEIGHT, VELOCITY_REWARD_WEIGHT, RAIL_FORCE_CONTROL_REWARD_WEIGHT, U_CONTROL_REWARD_WEIGHT]

N_PRISMATIC_DOFS = 1 if USE_MOVING_BASE else 0
INIT_QUAT = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
INIT_X, INIT_Y, INIT_Z = 0.0, 0.0, 1.5

MAX_EFFECTIVE_ANGLE = math.radians(45)
VINE_LENGTH = LENGTH_PER_LINK * N_REVOLUTE_DOFS

TARGET_POS_MIN_X, TARGET_POS_MAX_X = 0.0, 0.0  # Ignored dimension
# TARGET_POS_MIN_Y, TARGET_POS_MAX_Y = -LENGTH_RAIL/2, LENGTH_RAIL/2  # Set to length of rail
TARGET_POS_MIN_Y, TARGET_POS_MAX_Y = (-math.sin(MAX_EFFECTIVE_ANGLE)*VINE_LENGTH,
                                      math.sin(MAX_EFFECTIVE_ANGLE) * VINE_LENGTH)
TARGET_POS_MIN_Z, TARGET_POS_MAX_Z = INIT_Z - VINE_LENGTH, INIT_Z - math.cos(MAX_EFFECTIVE_ANGLE) * VINE_LENGTH

DOF_MODE = "FORCE"  # "FORCE" OR "POSITION"
RANDOMIZE_DOF_INIT = False
RANDOMIZE_TARGETS = True
PD_TARGET_ALL_JOINTS = False

# GLOBALS
USE_WANDB = True


def print_if(text="", should_print=False):
    if should_print:
        print(text)


class Vine5LinkMovingBase(VecTask):
    """
    State:
      * 6 Joint positions (5 revolute, 1 prismatic)
    Goal:
      * 1 Target Pos
    Observation:
      * 6 joint positions
      * 6 joint velocities
      * 3 tip position
      * 3 tip velocity
      * 3 target position
      * 3 target velocity
    Action:
      * 1 for rail_force prismatic joint
      * 1 for u pressure
    Reward:
      * -Dist to target
    Environment:
      * Random target position
      * Random start position
    """

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        # Store cfg file and read in parameters
        self.cfg = cfg
        self.max_episode_length = self.cfg["env"]["maxEpisodeLength"]

        # Must set this before continuing
        if NO_VEL_IN_OBS:
            self.cfg["env"]["numObservations"] = N_REVOLUTE_DOFS + N_PRISMATIC_DOFS + NUM_XYZ + NUM_XYZ
        else:
            self.cfg["env"]["numObservations"] = 2 * (N_REVOLUTE_DOFS + N_PRISMATIC_DOFS + NUM_XYZ + NUM_XYZ)

        if PD_TARGET_ALL_JOINTS:
            self.cfg["env"]["numActions"] = N_REVOLUTE_DOFS + N_PRISMATIC_DOFS
        else:
            self.cfg["env"]["numActions"] = N_PRESSURE_ACTIONS + N_PRISMATIC_DOFS

        self.subscribe_to_keyboard_events()

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id,
                         headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        self.initialize_state_tensors()
        self.target_positions = self.sample_target_positions(self.num_envs)
        self.target_velocities = self.sample_target_velocities(self.num_envs)
        self.A = None  # Cache this matrix
        self.reward_weights = torch.tensor([REWARD_WEIGHTS], device=self.device)

        # Setup viewer camera
        index_to_view = int(0.1 * self.num_envs)
        tip_pos = self.tip_positions[index_to_view]
        cam_target = gymapi.Vec3(tip_pos[0], tip_pos[1], INIT_Z)
        cam_pos = cam_target + gymapi.Vec3(2.0, 0.0, 0.0)
        self.gym.viewer_camera_look_at(self.viewer, self.envs[index_to_view], cam_pos, cam_target)

        self.wandb_dict = {}

    def initialize_state_tensors(self):
        # Store dof state tensor, and get pos and vel
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        # Store rigid body state tensor
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state_tensor)

        # For now, only care about last link to get tip location
        # rigid_body_names = self.gym.get_asset_rigid_body_dict(self.vine_asset)
        rigid_body_state_by_env = self.rigid_body_state.view(
            self.num_envs, self.num_rigid_bodies, NUM_STATES)

        self.link_positions = rigid_body_state_by_env[:, :, START_POS_IDX:END_POS_IDX]
        self.tip_positions = rigid_body_state_by_env[:, -1, START_POS_IDX:END_POS_IDX]
        self.link_velocities = rigid_body_state_by_env[:, :, START_LIN_VEL_IDX:END_LIN_VEL_IDX]
        self.tip_velocities = rigid_body_state_by_env[:, -1, START_LIN_VEL_IDX:END_LIN_VEL_IDX]

    def refresh_state_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

    def log_wandb_dict(self):
        # Only wandb log if working
        global USE_WANDB
        if USE_WANDB:
            try:
                wandb.log(self.wandb_dict)
            except wandb.errors.Error:
                print("Wandb not initialized, no longer trying to log")
                USE_WANDB = False
            self.wandb_dict = {}

    ##### KEYBOARD EVENT SUBSCRIPTIONS START #####
    def subscribe_to_keyboard_events(self):
        # Need to populate self.event_action_to_key and self.event_action_to_function

        self.event_action_to_key = {
            "RESET": gymapi.KEY_R,
            "PAUSE": gymapi.KEY_P,
            "PRINT_DEBUG": gymapi.KEY_D,
            "PRINT_DEBUG_IDX_UP": gymapi.KEY_K,
            "PRINT_DEBUG_IDX_DOWN": gymapi.KEY_J,
            "MOVE_LEFT": gymapi.KEY_LEFT,
            "MOVE_RIGHT": gymapi.KEY_RIGHT,
            "MAX_PRESSURE": gymapi.KEY_UP,
            "MIN_PRESSURE": gymapi.KEY_DOWN,
        }
        self.event_action_to_function = {
            "RESET": self._reset_callback,
            "PAUSE": self._pause_callback,
            "PRINT_DEBUG": self._print_debug_callback,
            "PRINT_DEBUG_IDX_UP": self._print_debug_idx_up_callback,
            "PRINT_DEBUG_IDX_DOWN": self._print_debug_idx_down_callback,
            "MOVE_LEFT": self._move_left_callback,
            "MOVE_RIGHT": self._move_right_callback,
            "MAX_PRESSURE": self._max_pressure_callback,
            "MIN_PRESSURE": self._min_pressure_callback,
        }
        # Create state variables
        self.PRINT_DEBUG = False
        self.PRINT_DEBUG_IDX = 0
        self.MOVE_LEFT_COUNTER = 0
        self.MOVE_RIGHT_COUNTER = 0
        self.MAX_PRESSURE_COUNTER = 0
        self.MIN_PRESSURE_COUNTER = 0

        assert (sorted(list(self.event_action_to_key.keys())) == sorted(list(self.event_action_to_function.keys())))

    def _reset_callback(self):
        print("RESETTING")
        all_env_ids = torch.ones_like(self.reset_buf).nonzero(as_tuple=False).squeeze(-1)
        self.reset_idx(all_env_ids)

    def _pause_callback(self):
        print("PAUSING")
        import time
        time.sleep(1)

    def _print_debug_callback(self):
        self.PRINT_DEBUG = not self.PRINT_DEBUG
        print(f"self.PRINT_DEBUG = {self.PRINT_DEBUG}")

    def _print_debug_idx_up_callback(self):
        self.PRINT_DEBUG_IDX += 1
        if self.PRINT_DEBUG_IDX >= self.num_envs:
            self.PRINT_DEBUG_IDX = self.num_envs - 1
        print(f"self.PRINT_DEBUG_IDX = {self.PRINT_DEBUG_IDX}")

    def _print_debug_idx_down_callback(self):
        self.PRINT_DEBUG_IDX -= 1
        if self.PRINT_DEBUG_IDX < 0:
            self.PRINT_DEBUG_IDX = 0
        print(f"self.PRINT_DEBUG_IDX = {self.PRINT_DEBUG_IDX}")

    def _move_left_callback(self):
        self.MOVE_LEFT_COUNTER = 100
        self.MOVE_RIGHT_COUNTER = 0
        print(f"self.MOVE_LEFT = {self.MOVE_LEFT_COUNTER}")

    def _move_right_callback(self):
        self.MOVE_RIGHT_COUNTER = 100
        self.MOVE_LEFT_COUNTER = 0
        print(f"self.MOVE_RIGHT = {self.MOVE_RIGHT_COUNTER}")

    def _max_pressure_callback(self):
        self.MAX_PRESSURE_COUNTER = 100
        self.MIN_PRESSURE_COUNTER = 0
        print(f"self.MAX_PRESSURE = {self.MAX_PRESSURE_COUNTER}")

    def _min_pressure_callback(self):
        self.MIN_PRESSURE_COUNTER = 100
        self.MAX_PRESSURE_COUNTER = 0
        print(f"self.MIN_PRESSURE = {self.MIN_PRESSURE_COUNTER}")

    ##### KEYBOARD EVENT SUBSCRIPTIONS END #####

    def create_sim(self):
        # set the up axis to be z-up given that assets are y-up by default
        self.up_axis = self.cfg["sim"]["up_axis"]

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        # set the normal force to be z dimension
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0) if self.up_axis == 'z' else gymapi.Vec3(0.0, 1.0, 0.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        # define plane on which environments are initialized
        lower = gymapi.Vec3(0.5 * -spacing, -spacing,
                            0.0) if self.up_axis == 'z' else gymapi.Vec3(0.5 * -spacing, 0.0, -spacing)
        upper = gymapi.Vec3(0.5 * spacing, spacing, spacing)

        # Find asset file
        vine_asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        vine_asset_file = "urdf/Vine5LinkMovingBase.urdf" if USE_MOVING_BASE else "urdf/Vine5LinkFixedBase.urdf"

        vine_asset_path = os.path.join(vine_asset_root, vine_asset_file)
        vine_asset_root = os.path.dirname(vine_asset_path)
        vine_asset_file = os.path.basename(vine_asset_path)

        # Create vine asset
        vine_asset_options = gymapi.AssetOptions()
        vine_asset_options.fix_base_link = True  # Fixed base for vine
        self.vine_asset = self.gym.load_asset(self.sim, vine_asset_root, vine_asset_file, vine_asset_options)

        # Store useful variables
        self.num_dof = self.gym.get_asset_dof_count(self.vine_asset)
        self.num_rigid_bodies = self.gym.get_asset_rigid_body_count(self.vine_asset)

        # Sanity check
        dof_types = [self.gym.get_asset_dof_type(self.vine_asset, i)
                     for i in range(self.gym.get_asset_dof_count(self.vine_asset))]
        num_revolute_dofs = len([dof_type for dof_type in dof_types if dof_type == gymapi.DofType.DOF_ROTATION])
        num_prismatic_dofs = len([dof_type for dof_type in dof_types if dof_type == gymapi.DofType.DOF_TRANSLATION])
        assert (num_revolute_dofs + num_prismatic_dofs == self.num_dof)
        assert (num_revolute_dofs == N_REVOLUTE_DOFS)
        assert (num_prismatic_dofs == N_PRISMATIC_DOFS)

        # Split into revolute and prismatic
        dof_names = [self.gym.get_asset_dof_name(self.vine_asset, i)
                     for i in range(self.gym.get_asset_dof_count(self.vine_asset))]
        dof_dict = self.gym.get_asset_dof_dict(self.vine_asset)
        revolute_dof_names = [dof_name for dof_name, dof_type in zip(
            dof_names, dof_types) if dof_type == gymapi.DofType.DOF_ROTATION]
        prismatic_dof_names = [dof_name for dof_name, dof_type in zip(
            dof_names, dof_types) if dof_type == gymapi.DofType.DOF_TRANSLATION]
        self.revolute_dof_indices = sorted([dof_dict[name] for name in revolute_dof_names])
        self.prismatic_dof_indices = sorted([dof_dict[name] for name in prismatic_dof_names])

        # Sanity check ordering of indices
        if N_PRISMATIC_DOFS == 1:
            assert (self.prismatic_dof_indices == [0])
            assert (self.revolute_dof_indices == [i+1 for i in range(N_REVOLUTE_DOFS)])
        elif N_PRISMATIC_DOFS == 0:
            assert (self.prismatic_dof_indices == [])
            assert (self.revolute_dof_indices == [i for i in range(N_REVOLUTE_DOFS)])
        else:
            raise ValueError(f"Can't have N_PRISMATIC_DOFS = {N_PRISMATIC_DOFS}")

        # Store limits
        self.dof_props = self.gym.get_asset_dof_properties(self.vine_asset)
        self.dof_lowers = torch.from_numpy(self.dof_props["lower"]).to(self.device)
        self.dof_uppers = torch.from_numpy(self.dof_props["upper"]).to(self.device)
        self.revolute_dof_lowers = self.dof_lowers[self.revolute_dof_indices]
        self.revolute_dof_uppers = self.dof_uppers[self.revolute_dof_indices]
        self.prismatic_dof_lowers = self.dof_lowers[self.prismatic_dof_indices]
        self.prismatic_dof_uppers = self.dof_uppers[self.prismatic_dof_indices]

        # Set initial actor poses
        vine_init_pose = gymapi.Transform()
        assert (self.up_axis == 'z')
        vine_init_pose.p.x = INIT_X
        vine_init_pose.p.y = INIT_Y
        vine_init_pose.p.z = INIT_Z
        vine_init_pose.r = INIT_QUAT

        self.vine_handles = []
        self.envs = []
        self.object_handles = []
        for i in range(num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            # Different collision_groups so that different envs don't interact
            # collision_filter = 0 for enabled self-collision, collision_filter > 0 disable self-collisions
            collision_group, collision_filter, segmentation_id = i, 0, 0

            # Create vine robots
            vine_handle = self.gym.create_actor(
                env_ptr, self.vine_asset, vine_init_pose, "vine", group=collision_group, filter=collision_filter, segmentationId=segmentation_id)

            # Set dof properties
            dof_props = self.gym.get_actor_dof_properties(env_ptr, vine_handle)

            if DOF_MODE == "FORCE":
                dof_props['driveMode'][:] = gymapi.DOF_MODE_EFFORT
            elif DOF_MODE == "POSITION":
                dof_props['driveMode'][j] = gymapi.DOF_MODE_POS
            else:
                raise ValueError(f"Invalid DOF_MODE = {DOF_MODE}")

            self.gym.set_actor_dof_properties(env_ptr, vine_handle, dof_props)

            self.envs.append(env_ptr)
            self.vine_handles.append(vine_handle)

        PRINT_ASSET_INFO = False
        if PRINT_ASSET_INFO:
            self._print_asset_info(self.vine_asset)

    def _print_asset_info(self, asset):
        """
        self.num_dof = 6
        DOF 0
          Name:     'slider_to_cart'
          Type:     Translation
          Properties:  (True, -0.35, 0.35, 0, 0.5, 1., 0., 0., 0., 0.)
        DOF 1
          Name:     'cart_to_link_0'
          Type:     Rotation
          Properties:  (True, -0.52, 0.52, 0, 3.4e+38, 3.4e+38, 0., 0., 0., 0.)
        DOF 2
          Name:     'link_0_to_link_1'
          Type:     Rotation
          Properties:  (True, -0.52, 0.52, 0, 3.4e+38, 3.4e+38, 0., 0., 0., 0.)
        DOF 3
          Name:     'link_1_to_link_2'
          Type:     Rotation
          Properties:  (True, -0.52, 0.52, 0, 3.4e+38, 3.4e+38, 0., 0., 0., 0.)
        DOF 4
          Name:     'link_2_to_link_3'
          Type:     Rotation
          Properties:  (True, -0.52, 0.52, 0, 3.4e+38, 3.4e+38, 0., 0., 0., 0.)
        DOF 5
          Name:     'link_3_to_link_4'
          Type:     Rotation
          Properties:  (True, -0.52, 0.52, 0, 3.4e+38, 3.4e+38, 0., 0., 0., 0.)

        self.num_rigid_bodies = 7
        rigid_body_dict = {'cart': 1, 'link_0': 2, 'link_1': 3, 'link_2': 4, 'link_3': 5, 'link_4': 6, 'slider': 0}
        joint_dict = {'cart_to_link_0': 1, 'link_0_to_link_1': 2, 'link_1_to_link_2': 3, 'link_2_to_link_3': 4, 'link_3_to_link_4': 5, 'slider_to_cart': 0}
        dof_dict = {'cart_to_link_0': 1, 'link_0_to_link_1': 2, 'link_1_to_link_2': 3, 'link_2_to_link_3': 4, 'link_3_to_link_4': 5, 'slider_to_cart': 0}

        Box(-1.0, 1.0, (2,), float32) Box(-inf, inf, (12,), float32)
        """
        # Acquire variables
        dof_names = self.gym.get_asset_dof_names(asset)
        dof_props = self.gym.get_asset_dof_properties(asset)
        dof_types = [self.gym.get_asset_dof_type(asset, i) for i in range(self.num_dof)]
        dof_type_strings = [self.gym.get_dof_type_string(dof_type) for dof_type in dof_types]

        rigid_body_dict = self.gym.get_asset_rigid_body_dict(asset)
        joint_dict = self.gym.get_asset_joint_dict(asset)
        dof_dict = self.gym.get_asset_dof_dict(asset)

        print(f"self.num_dof = {self.num_dof}")
        for i, (dof_name, dof_prop, dof_type_string) in enumerate(zip(dof_names, dof_props, dof_type_strings)):
            print("DOF %d" % i)
            print("  Name:     '%s'" % dof_name)
            print("  Type:     %s" % dof_type_string)
            print("  Properties:  %r" % dof_prop)
        print()
        print(f"self.num_rigid_bodies = {self.num_rigid_bodies}")
        print(f"rigid_body_dict = {rigid_body_dict}")
        print(f"joint_dict = {joint_dict}")
        print(f"dof_dict = {dof_dict}")
        print()

    def compute_reward(self):
        # retrieve environment observations from buffer
        tip_positions = self.obs_buf[:, -6:-3]
        target_positions = self.obs_buf[:, -3:]
        dist_tip_to_target = torch.linalg.norm(tip_positions - target_positions, dim=-1)

        SUCCESS_DIST = 0.1
        target_reached = dist_tip_to_target < SUCCESS_DIST

        self.wandb_dict.update({
            "dist_tip_to_target": dist_tip_to_target.mean().item(),
            "abs_tip_y": tip_positions[:, 1].abs().mean().item(),
            "tip_z": tip_positions[:, 2].mean().item(),
            "max_abs_tip_y": tip_positions[:, 1].abs().max().item(),
            "max_tip_z": tip_positions[:, 2].max().item(),
            "tip_velocities": torch.norm(self.tip_velocities, dim=-1).mean().item(),
            "rail_force": torch.norm(self.rail_force, dim=-1).mean().item(),
            "u": torch.norm(self.u, dim=-1).mean().item(),
            "target_reached": target_reached.float().mean().item(),
            "target_reached_max": target_reached.float().max().item(),
            "tip_velocities_max": torch.norm(self.tip_velocities, dim=-1).max().item(),
        })

        self.rew_buf[:], reward_matrix = compute_reward_jit(
            dist_tip_to_target, target_reached, self.tip_velocities, self.target_velocities, self.rail_force, self.u, self.reward_weights, REWARD_NAMES
        )

        for i, reward_name in enumerate(REWARD_NAMES):
            self.wandb_dict.update({
                f"Mean {reward_name} Reward": reward_matrix[:, i].mean().item(),
                f"Max {reward_name} Reward": reward_matrix[:, i].max().item()
            })

        self.reset_buf[:] = compute_reset_jit(self.reset_buf, self.progress_buf,
                                              self.max_episode_length, target_reached)

    def compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        # Refresh tensors
        self.refresh_state_tensors()

        # Populate obs_buf
        # tensors_to_add elements must all be (num_envs, X)
        if NO_VEL_IN_OBS:
            tensors_to_concat = [self.dof_pos, self.tip_positions, self.target_positions]
        else:
            tensors_to_concat = [self.dof_pos, self.dof_vel, self.tip_positions, self.tip_velocities, self.target_positions, self.target_velocities]
        self.obs_buf[:] = torch.cat(tensors_to_concat, dim=-1)

        return self.obs_buf

    def reset_idx(self, env_ids):
        if RANDOMIZE_DOF_INIT:
            num_revolute_joints = len(self.revolute_dof_lowers)
            for i in range(num_revolute_joints):
                self.dof_pos[env_ids, self.revolute_dof_indices[i]] = torch.FloatTensor(len(env_ids)).uniform_(
                    self.revolute_dof_lowers[i], self.revolute_dof_uppers[i]).to(self.device)

            num_prismatic_joints = len(self.prismatic_dof_lowers)
            for i in range(num_prismatic_joints):
                self.dof_pos[env_ids, self.prismatic_dof_indices[i]] = torch.FloatTensor(len(env_ids)).uniform_(
                    self.prismatic_dof_lowers[i], self.prismatic_dof_uppers[i]).to(self.device)
        else:
            self.dof_pos[env_ids, :] = 0

        # Set dof velocities to 0
        self.dof_pos[env_ids, :] = 0.0
        self.dof_vel[env_ids, :] = 0.0

        # Update dofs
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        if len(env_ids_int32) == self.num_envs:
            self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))
        else:
            self.gym.set_dof_state_tensor_indexed(self.sim,
                                                  gymtorch.unwrap_tensor(self.dof_state),
                                                  gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        # New target positions
        self.target_positions[env_ids, :] = self.sample_target_positions(len(env_ids))
        self.target_velocities[env_ids, :] = self.sample_target_velocities(len(env_ids))

    def sample_target_positions(self, num_envs):
        target_positions = torch.zeros(num_envs, NUM_XYZ, device=self.device)
        if RANDOMIZE_TARGETS:
            target_positions[:, 0] = torch.FloatTensor(num_envs).uniform_(
                TARGET_POS_MIN_X, TARGET_POS_MAX_X).to(self.device)
            target_positions[:, 1] = torch.FloatTensor(num_envs).uniform_(
                TARGET_POS_MIN_Y, TARGET_POS_MAX_Y).to(self.device)
            target_positions[:, 2] = torch.FloatTensor(num_envs).uniform_(
                TARGET_POS_MIN_Z, TARGET_POS_MAX_Z).to(self.device)
        else:
            target_positions[:, 1] = TARGET_POS_MAX_Y
            target_positions[:, 2] = TARGET_POS_MIN_Z

        return target_positions

    def sample_target_velocities(self, num_envs):
        # TODO
        return torch.zeros(num_envs, NUM_XYZ, device=self.device)

    def pre_physics_step(self, actions):
        self.raw_actions = actions.clone().to(self.device)

        if DOF_MODE == "FORCE":
            if PD_TARGET_ALL_JOINTS:
                REVOLUTE_FORCE_SCALING = 1.0
                PRISMATIC_FORCE_SCALING = 1000.0
                dof_efforts = torch.zeros(self.num_envs, self.num_dof, device=self.device)

                # Revolute
                for idx in self.revolute_dof_indices:
                    dof_efforts[:, idx] = self.raw_actions[:, idx] * REVOLUTE_FORCE_SCALING
                for idx in self.prismatic_dof_indices:
                    dof_efforts[:, idx] = self.raw_actions[:, idx] * PRISMATIC_FORCE_SCALING
                self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(dof_efforts))
            else:
                dof_efforts = torch.zeros(self.num_envs, self.num_dof, device=self.device)

                # Break apart actions and states
                if N_PRISMATIC_DOFS == 1:
                    self.rail_force = rescale_to_rail_force(self.raw_actions[:, 0:1])  # (num_envs, 1)
                    self.u = rescale_to_u(self.raw_actions[:, 1:2])  # (num_envs, 1)
                    q = self.dof_pos[:, 1:]  # (num_envs, 5)
                    qd = self.dof_vel[:, 1:]  # (num_envs, 5)
                elif N_PRISMATIC_DOFS == 0:
                    self.rail_force = torch.zeros_like(self.raw_actions[:, 0:1], device=self.device)  # (num_envs, 1)
                    self.u = rescale_to_u(self.raw_actions)  # (num_envs, 1)
                    q = self.dof_pos[:]  # (num_envs, 5)
                    qd = self.dof_vel[:]  # (num_envs, 5)
                else:
                    raise ValueError(f"Can't have N_PRISMATIC_DOFS = {N_PRISMATIC_DOFS}")

                # Manual intervention
                if self.MOVE_LEFT_COUNTER > 0:
                    self.rail_force[:] = -RAIL_FORCE_SCALE
                    self.MOVE_LEFT_COUNTER -= 1
                if self.MOVE_RIGHT_COUNTER > 0:
                    self.rail_force[:] = RAIL_FORCE_SCALE
                    self.MOVE_RIGHT_COUNTER -= 1

                if self.MAX_PRESSURE_COUNTER > 0:
                    self.u[:] = U_MAX
                    self.MAX_PRESSURE_COUNTER -= 1
                if self.MIN_PRESSURE_COUNTER > 0:
                    self.u[:] = U_MIN
                    self.MIN_PRESSURE_COUNTER -= 1

                if self.A is None:
                    # torque = - Kq - Cqd - b - Bu;
                    #        = - [K C diag(b) diag(B)] @ [q; qd; ones(5), u*ones(5)]
                    #        = - A @ x
                    K = torch.diag(torch.tensor([1.0822678619473745, 1.3960597815085283,
                                                 0.7728716674414156, 0.566602254820747, 0.20000000042282678], device=self.device))
                    C = torch.diag(torch.tensor([0.010098832804688505, 0.008001446516454621,
                                                 0.01352315902253585, 0.021895211325047674, 0.017533205699630634], device=self.device))
                    b = torch.tensor([-0.002961879962361915, -0.019149230853283454, -0.01339719175569314, -
                                     0.011436913019114144, -0.0031035566743229624], device=self.device)
                    B = torch.tensor([-0.02525783894248118, -0.06298872026151316, -0.049676622868418834, -
                                     0.029474741498381096, -0.015412936470522515], device=self.device)
                    A1 = torch.cat([K, C, torch.diag(b), torch.diag(B)], dim=-1)  # (5, 20)
                    self.A = A1[None, ...].repeat_interleave(self.num_envs, dim=0)  # (num_envs, 5, 20)

                x = torch.cat([q, qd, torch.ones(self.num_envs, 5, device=self.device), self.u *
                              torch.ones(self.num_envs, 5, device=self.device)], dim=1)[..., None]  # (num_envs, 20, 1)
                torques = -torch.matmul(self.A, x).squeeze().cpu()  # (num_envs, 5, 1) => (num_envs, 5)

                # Set efforts
                if N_PRISMATIC_DOFS == 1:
                    dof_efforts[:, 0:1] = self.rail_force
                    dof_efforts[:, 1:] = torques
                elif N_PRISMATIC_DOFS == 0:
                    dof_efforts[:, :] = torques
                self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(dof_efforts))

        elif DOF_MODE == "POSITION":
            raise ValueError(f"Unable to run with {DOF_MODE}")
        else:
            raise ValueError(f"Invalid DOF_MODE = {DOF_MODE}")

    def post_physics_step(self):
        self.progress_buf += 1

        # Reset
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        # Compute observations and reward
        self.compute_observations()
        self.compute_reward()

        # Log info
        self.log_wandb_dict()

        # Draw debug info
        if self.viewer and self.enable_viewer_sync:
            # Create spheres
            visualization_sphere_radius = 0.05
            visualization_sphere_green = gymutil.WireframeSphereGeometry(
                visualization_sphere_radius, 3, 3, color=(0, 1, 0))

            self.gym.clear_lines(self.viewer)
            for i in range(self.num_envs):
                # Draw target
                target_position = self.target_positions[i]
                sphere_pose = gymapi.Transform(gymapi.Vec3(
                    target_position[0], target_position[1], target_position[2]), r=None)
                gymutil.draw_lines(visualization_sphere_green, self.gym, self.viewer, self.envs[i], sphere_pose)


def rescale_to_u(u):
    return (u + 1.0) / 2.0 * (U_MAX-U_MIN) + U_MIN


def rescale_to_rail_force(rail_force):
    return rail_force * RAIL_FORCE_SCALE

#####################################################################
### =========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_reward_jit(dist_to_target, target_reached, tip_velocities, target_velocities, rail_force, u, reward_weights, REWARD_NAMES):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, List[str]) -> Tuple[Tensor, Tensor]
    # reward = sum(w_i * r_i) with various reward function r_i and weights w_i

    # dense_reward = -dist_to_target [Try to reach target]
    # const_negative_reward = -1 [Punish for not succeeding]
    # position_success_reward = REWARD_BONUS if dist_to_target < SUCCESS_DISTANCE else 0 [Succeed if close enough]
    # velocity_success_reward = -norm(tip_velocity - desired_tip_velocity) if dist_to_target < SUCCESS_DISTANCE else 0 [Succeed if close enough and moving at the right speed]
    # velocity_reward = norm(tip_velocity) [Try to move fast]
    # rail_force_control_reward = -norm(rail_force) [Punish for using too much actuation]
    # u_control_reward = -norm(rail_force) [Punish for using too much actuation]
    N_REWARDS = torch.numel(reward_weights)
    N_ENVS = dist_to_target.shape[0]

    REWARD_BONUS = 1000.0

    # Brittle: Ensure reward order matches
    reward_matrix = torch.zeros(N_ENVS, N_REWARDS, device=dist_to_target.device)
    for i, reward_name in enumerate(REWARD_NAMES):
        if reward_name == "Dense":
            reward_matrix[:, i] -= dist_to_target
        elif reward_name == "Const Negative":
            reward_matrix[:, i] -= 1
        elif reward_name == "Position Success":
            reward_matrix[:, i] += torch.where(target_reached, REWARD_BONUS, 0.0)
        elif reward_name == "Velocity Success":
            reward_matrix[:, i] -= torch.where(target_reached, torch.norm(tip_velocities - target_velocities, dim=-1).double(), 0.0)
        elif reward_name == "Velocity":
            reward_matrix[:, i] += torch.norm(tip_velocities, dim=-1)
        elif reward_name == "Rail Force Control":
            reward_matrix[:, i] -= torch.norm(rail_force, dim=-1)
        elif reward_name == "U Control":
            reward_matrix[:, i] -= torch.norm(u, dim=-1)
        else:
            raise ValueError(f"Invalid reward name: {reward_name}")

    total_reward = torch.sum(reward_matrix * reward_weights, dim=-1)

    return total_reward, reward_matrix


def compute_reset_jit(reset_buf, progress_buf, max_episode_length, target_reached):
    # type: (Tensor, Tensor, float, Tensor) -> Tensor
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(target_reached, torch.ones_like(reset), reset)
    return reset

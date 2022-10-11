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

from isaacgym import gymutil, gymtorch, gymapi
from .base.vec_task import VecTask

# CONSTANTS
NUM_STATES = 13  # xyz, quat, v_xyz, w_xyz
NUM_XYZ = 3
HORIZONTAL_PLANE_QUAT = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
VERTICAL_PLANE_QUAT = gymapi.Quat(0.707, 0.0, 0.0, 0.707)

# PARAMETERS
INIT_X, INIT_Y, INIT_Z = 0.0, 0.0, 1.5
INIT_QUAT = VERTICAL_PLANE_QUAT
TARGET_POS_MIN, TARGET_POS_MAX = -0.7, 0.7
DOF_MODE = "POSITION"  # "FORCE" OR "POSITION"
N_REVOLUTE_DOFS = 3
RANDOMIZE_REVOLUTES = False
RANDOMIZE_PRISMATICS = False
JOINT_BUFFER = 0.9

# TODO: Investigate if self collision checks work (probably not)


class Vine(VecTask):
    """
    Let N = 3 revolute joints

    State:
      * 6 Joint positions (3 revolute, 3 prismatic)
    Goal:
      * 1 Target Pos
    Observation:
      * 6 Cos/Sin of Revolute Joint Positions (3 revolute)
      * 3 Prismatic Joint Positions
      * 3 tip position
      * 3 target position
    Action:
      * 3 for revolute joint angles (only applies actions to angles that have length)
      * 1 desired full length
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
        self.cfg["env"]["numObservations"] = 15
        self.cfg["env"]["numActions"] = N_REVOLUTE_DOFS + 1
        self.subscribe_to_keyboard_events()

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id,
                         headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        self.initialize_state_tensors()
        self.target_positions = self.sample_target_positions(self.num_envs)

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
        rigid_body_state_by_env = self.rigid_body_state.view(self.num_envs, self.num_rigid_bodies, NUM_STATES)
        self.tip_positions = rigid_body_state_by_env[:, -1, 0:3]

    def refresh_state_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

    ##### KEYBOARD EVENT SUBSCRIPTIONS START #####
    def subscribe_to_keyboard_events(self):
        # Need to populate self.event_action_to_key and self.event_action_to_function

        self.event_action_to_key = {
            "RESET": gymapi.KEY_R,
            "PAUSE": gymapi.KEY_P,
            "ELONGATE": gymapi.KEY_W,
            "SHORTEN": gymapi.KEY_S,
            "TURN_LEFT": gymapi.KEY_LEFT,
            "TURN_RIGHT": gymapi.KEY_RIGHT,
            "NEXT_TURN_IDX": gymapi.KEY_UP,
            "PREV_TURN_IDX": gymapi.KEY_DOWN,
        }
        self.event_action_to_function = {
            "RESET": self._reset_callback,
            "PAUSE": self._pause_callback,
            "ELONGATE": self._elongate_callback,
            "SHORTEN": self._shorten_callback,
            "TURN_LEFT": self._turn_left_callback,
            "TURN_RIGHT": self._turn_right_callback,
            "NEXT_TURN_IDX": self._next_turn_idx_callback,
            "PREV_TURN_IDX": self._prev_turn_idx_callback,
        }
        # Create state variables
        self.FORCE_ELONGATE = 0
        self.FORCE_SHORTEN = 0
        self.FORCE_TURN_LEFT = 0
        self.FORCE_TURN_RIGHT = 0
        self.FORCE_TURN_IDX = 0

        assert(sorted(list(self.event_action_to_key.keys())) == sorted(list(self.event_action_to_function.keys())))

    def _reset_callback(self):
        print("RESETTING")
        all_env_ids = torch.ones_like(self.reset_buf).nonzero(as_tuple=False).squeeze(-1)
        self.reset_idx(all_env_ids)

    def _pause_callback(self):
        print("PAUSING")
        import time
        time.sleep(1)

    def _elongate_callback(self):
        print("ELONGATING")
        self.FORCE_ELONGATE = 60
        self.FORCE_SHORTEN = 0

    def _shorten_callback(self):
        print("SHORTENING")
        self.FORCE_SHORTEN = 60
        self.FORCE_ELONGATE = 0

    def _turn_left_callback(self):
        print(f"TURNING_LEFT for idx = {self.FORCE_TURN_IDX}")
        self.FORCE_TURN_LEFT = 60
        self.FORCE_TURN_RIGHT = 0

    def _turn_right_callback(self):
        print(f"TURNING_RIGHT for idx = {self.FORCE_TURN_IDX}")
        self.FORCE_TURN_RIGHT = 60
        self.FORCE_TURN_LEFT = 0

    def _next_turn_idx_callback(self):
        print(f"INCREASING TURN IDX")
        self.FORCE_TURN_IDX += 1
        if self.FORCE_TURN_IDX >= N_REVOLUTE_DOFS:
            self.FORCE_TURN_IDX = N_REVOLUTE_DOFS - 1
        print(f"Now: self.FORCE_TURN_IDX = {self.FORCE_TURN_IDX}")

    def _prev_turn_idx_callback(self):
        print(f"DECREASING TURN IDX")
        self.FORCE_TURN_IDX -= 1
        if self.FORCE_TURN_IDX < 0:
            self.FORCE_TURN_IDX = 0
        print(f"NOW: self.FORCE_TURN_IDX = {self.FORCE_TURN_IDX}")
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
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        asset_file = "urdf/vine.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      self.cfg["env"]["asset"].get("assetRoot", asset_root))
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        # Create asset
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True  # Fixed base for vine
        self.vine_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        # Store useful variables
        self.num_dof = self.gym.get_asset_dof_count(self.vine_asset)
        self.num_rigid_bodies = self.gym.get_asset_rigid_body_count(self.vine_asset)

        # Sanity check
        dof_dict = self.gym.get_asset_dof_dict(self.vine_asset)
        num_revolute_dofs = len([name for name, _ in dof_dict.items() if "revolute" in name])
        num_prismatic_dofs = len([name for name, _ in dof_dict.items() if "prismatic" in name])
        assert(num_revolute_dofs == num_prismatic_dofs)
        assert(num_revolute_dofs + num_prismatic_dofs == self.num_dof)
        assert(num_revolute_dofs == N_REVOLUTE_DOFS)

        # Split into revolute and prismatic
        revolute_dof_names = [name for name, _ in dof_dict.items() if "revolute" in name]
        prismatic_dof_names = [name for name, _ in dof_dict.items() if "prismatic" in name]
        self.revolute_dof_indices = sorted([dof_dict[name] for name in revolute_dof_names])
        self.prismatic_dof_indices = sorted([dof_dict[name] for name in prismatic_dof_names])

        # Store limits
        self.dof_props = self.gym.get_asset_dof_properties(self.vine_asset)
        self.dof_lowers = torch.from_numpy(self.dof_props["lower"]).to(self.device)
        self.dof_uppers = torch.from_numpy(self.dof_props["upper"]).to(self.device)
        self.revolute_dof_lowers = self.dof_lowers[self.revolute_dof_indices]
        self.revolute_dof_uppers = self.dof_uppers[self.revolute_dof_indices]
        self.prismatic_dof_lowers = self.dof_lowers[self.prismatic_dof_indices]
        self.prismatic_dof_uppers = self.dof_uppers[self.prismatic_dof_indices]

        # Set initial actor poses
        pose = gymapi.Transform()
        assert(self.up_axis == 'z')
        pose.p.x = INIT_X
        pose.p.y = INIT_Y
        pose.p.z = INIT_Z
        pose.r = INIT_QUAT

        self.vine_handles = []
        self.envs = []
        for i in range(num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            collision_group, collision_filter, segmentation_id = i, 1, 0
            vine_handle = self.gym.create_actor(
                env_ptr, self.vine_asset, pose, "vine", group=collision_group, filter=collision_filter, segmentationId=segmentation_id)

            # Set dof properties
            dof_props = self.gym.get_actor_dof_properties(env_ptr, vine_handle)

            for j in range(self.gym.get_asset_dof_count(self.vine_asset)):
                dof_type = self.gym.get_asset_dof_type(self.vine_asset, j)

                # Dof type specific params
                if dof_type == gymapi.DofType.DOF_ROTATION:
                    if j == 0:  # First revolute is different
                        dof_props['stiffness'][:] = 10.0  # TODO: Tune
                        dof_props['damping'][:] = 1.0  # TODO: Tune
                    else:
                        dof_props['stiffness'][:] = 10.0  # TODO: Tune
                        dof_props['damping'][:] = 1.0  # TODO: Tune
                elif dof_type == gymapi.DofType.DOF_TRANSLATION:
                    dof_props['stiffness'][:] = 100.0  # TODO: Tune
                    dof_props['damping'][:] = 1.0  # TODO: Tune
                else:
                    raise ValueError(f"Invalid dof_type = {dof_type}")

                if DOF_MODE == "FORCE":
                    dof_props['driveMode'][j] = gymapi.DOF_MODE_EFFORT
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
        Name:     'revolute_joint_0'
        Type:     Rotation
        Properties:  (True, -3.15, 3.15, 0, 3., 10., 0., 0., 0., 0.)
        DOF 1
        Name:     'prismatic_joint_0'
        Type:     Translation
        Properties:  (True, 0., 0.5, 0, 0.2, 200., 0., 0., 0., 0.)
        DOF 2
        Name:     'revolute_joint_1'
        Type:     Rotation
        Properties:  (True, -3.15, 3.15, 0, 3., 10., 0., 0., 0., 0.)
        DOF 3
        Name:     'prismatic_joint_1'
        Type:     Translation
        Properties:  (True, 0., 0.5, 0, 0.2, 200., 0., 0., 0., 0.)
        DOF 4
        Name:     'revolute_joint_2'
        Type:     Rotation
        Properties:  (True, -3.15, 3.15, 0, 3., 10., 0., 0., 0., 0.)
        DOF 5
        Name:     'prismatic_joint_2'
        Type:     Translation
        Properties:  (True, 0., 0.5, 0, 0.2, 200., 0., 0., 0., 0.)

        num_rigid_bodies = 7
        rigid_body_dict = {'base_link': 0, 'base_virtual_link': 1, 'link_0': 2, 'link_1': 4, 'link_2': 6, 'virtual_link_0': 3, 'virtual_link_1': 5}
        joint_dict = {'prismatic_joint_0': 1, 'prismatic_joint_1': 3, 'prismatic_joint_2': 5, 'revolute_joint_0': 0, 'revolute_joint_1': 2, 'revolute_joint_2': 4}
        dof_dict = {'prismatic_joint_0': 1, 'prismatic_joint_1': 3, 'prismatic_joint_2': 5, 'revolute_joint_0': 0, 'revolute_joint_1': 2, 'revolute_joint_2': 4}
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

        self.rew_buf[:], self.reset_buf[:] = compute_vine_reward(
            dist_tip_to_target, self.reset_buf, self.progress_buf, self.max_episode_length
        )

    def compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        # Refresh tensors
        self.refresh_state_tensors()

        # Split into revolute and prismatic
        revolute_dof_pos = self.dof_pos[:, self.revolute_dof_indices]
        prismatic_dof_pos = self.dof_pos[:, self.prismatic_dof_indices]

        # Populate obs_buf
        num_revolute_dofs, num_prismatic_dofs = self.num_dof // 2, self.num_dof // 2
        self.obs_buf[env_ids, 0:num_revolute_dofs] = torch.cos(revolute_dof_pos[env_ids, :])
        self.obs_buf[env_ids, num_revolute_dofs:(2 * num_revolute_dofs)] = torch.sin(revolute_dof_pos[env_ids, :])
        self.obs_buf[env_ids, (2 * num_revolute_dofs):(2 * num_revolute_dofs +
                                                       num_prismatic_dofs)] = prismatic_dof_pos[env_ids, :]
        self.obs_buf[env_ids, -6:-3] = self.tip_positions
        self.obs_buf[env_ids, -3:] = self.target_positions

        return self.obs_buf

    def reset_idx(self, env_ids):
        if RANDOMIZE_PRISMATICS:
            # Sample random initial lengths
            min_length, max_length = sum(self.prismatic_dof_lowers), sum(self.prismatic_dof_uppers)
            initial_lengths = torch.FloatTensor(len(env_ids)).uniform_(min_length, max_length).to(self.device)

            # Set initial prismatic dof positions to match sampled initial lengths
            current_lengths = initial_lengths.clone()
            num_prismatic_joints = len(self.prismatic_dof_lowers)
            for i in range(num_prismatic_joints):
                # If current_length not reached, then extend to full
                # Else fill in with current_length
                self.dof_pos[env_ids, self.prismatic_dof_indices[i]] = torch.where(current_lengths >= self.prismatic_dof_uppers[i],
                                                                                   self.prismatic_dof_uppers[i],
                                                                                   current_lengths)
                # Decrement current_lengths and ensure it says doesn't go below 0
                current_lengths -= self.prismatic_dof_uppers[i]
                current_lengths = torch.where(current_lengths < 0.0, 0.0, current_lengths)
        else:
            num_prismatic_joints = len(self.prismatic_dof_lowers)
            for i in range(num_prismatic_joints):
                self.dof_pos[env_ids, self.prismatic_dof_indices[i]] = 0

        if RANDOMIZE_REVOLUTES:
            # Set randomized initial revolute dof positions
            num_revolute_joints = len(self.revolute_dof_lowers)
            for i in range(num_revolute_joints):
                self.dof_pos[env_ids, self.revolute_dof_indices[i]] = torch.FloatTensor(len(env_ids)).uniform_(
                    self.revolute_dof_lowers[i], self.revolute_dof_uppers[i]).to(self.device)

                # Set to 0.0 if length at this index is 0.0
                self.dof_pos[env_ids, self.revolute_dof_indices[i]] = torch.where(self.dof_pos[env_ids, self.prismatic_dof_indices[i]] == 0.0,
                                                                                  0.0,
                                                                                  self.dof_pos[env_ids, self.revolute_dof_indices[i]])
        else:
            num_revolute_joints = len(self.revolute_dof_lowers)
            for i in range(num_revolute_joints):
                self.dof_pos[env_ids, self.revolute_dof_indices[i]] = 0

        # Set dof velocities to 0
        self.dof_vel[env_ids, :] = 0.0

        # Update dofs
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        # New target positions
        self.target_positions[env_ids, :] = self.sample_target_positions(len(env_ids))

    def sample_target_positions(self, num_envs):
        if INIT_QUAT == VERTICAL_PLANE_QUAT:
            target_positions = torch.FloatTensor(num_envs, NUM_XYZ).uniform_(
                TARGET_POS_MIN, TARGET_POS_MAX).to(self.device)
            target_positions[:, 1] = 0
            target_positions[:, 2] = torch.FloatTensor(num_envs).uniform_(0, 2*INIT_Z).to(self.device)
        elif INIT_QUAT == HORIZONTAL_PLANE_QUAT:
            target_positions = torch.FloatTensor(num_envs, NUM_XYZ).uniform_(
                TARGET_POS_MIN, TARGET_POS_MAX).to(self.device)
            target_positions[:, 2] = INIT_Z
        else:
            raise ValueError(f"Invalid INIT_QUAT = {INIT_QUAT}")
        return target_positions

    def pre_physics_step(self, actions):
        self.raw_actions = actions.clone().to(self.device)

        # Handle forced commands from keyboard
        if self.FORCE_ELONGATE > 0:
            self.raw_actions[:, -1] = 1.0
            self.FORCE_ELONGATE -= 1
        if self.FORCE_SHORTEN > 0:
            self.raw_actions[:, -1] = -1.0
            self.FORCE_SHORTEN -= 1
        if self.FORCE_TURN_LEFT > 0:
            self.raw_actions[:, self.FORCE_TURN_IDX] = 1.0
            self.FORCE_TURN_LEFT -= 1
        if self.FORCE_TURN_RIGHT > 0:
            self.raw_actions[:, self.FORCE_TURN_IDX] = -1.0
            self.FORCE_TURN_RIGHT -= 1

        # Break into revolute and prismatic action
        revolute_raw_actions, prismatic_raw_actions = self.raw_actions[:, :-1], self.raw_actions[:, -1]

        # Compute prismatic indexes (smallest index i such that prismatic_joint_i < prismatic_joint_limit_i, with buffer)
        # And remainder_lengths (length of prismatic_joint_i)
        prismatic_dof_pos = self.dof_pos[:, self.prismatic_dof_indices]
        num_prismatic_joints = len(self.prismatic_dof_lowers)
        prismatic_indexes = torch.ones(self.num_envs, dtype=torch.int32, device=self.device) * -1
        for i in range(num_prismatic_joints):
            JOINT_BUFFER = 0.9
            prismatic_indexes[(prismatic_dof_pos[:, i] < JOINT_BUFFER * self.prismatic_dof_uppers[i])
                              & (prismatic_indexes == -1)] = i
        # If goes to end, then bring it back to before end
        prismatic_indexes[prismatic_indexes == -1] = num_prismatic_joints - 1

        # TODO: Handle edge cases
        #   * full extension, then try to grow (do nothing) [SHOULD BE HANDLED]
        #   * full extension, then try to shrink (shrink) [NOT HANDLED]
        #   * full retraction, then try to grow (grow)  [SHOULD BE HANDLED]
        #   * full retraction, then try to shrink (do nothing)  [SHOULD BE HANDLED]
        #   * try retracting on empty link (need to switch to previous)  [NOT HANDLED]
        #   * try growing on full link (need to switch to next)  [SHOULD BE HANDLED]

        if DOF_MODE == "FORCE":
            # TODO: IF PREVIOUS LENGTH ALREADY FULL, DO I STILL NEED TO APPLY FORCE? Assume yes
            REVOLUTE_FORCE_SCALING = 1.0
            PRISMATIC_FORCE_SCALING = 1.0
            dof_efforts = torch.zeros(self.num_envs, self.num_dof, device=self.device)

            # Revolute
            dof_efforts[:, self.revolute_dof_indices] = revolute_raw_actions * REVOLUTE_FORCE_SCALING
            for i in range(num_prismatic_joints):
                dof_efforts[:, self.revolute_dof_indices[i]] = torch.where(
                    i <= prismatic_indexes, dof_efforts[:, self.revolute_dof_indices[i]], 0)  # TODO: Make it go back to middle

            # Prismatic
            for i in range(num_prismatic_joints):
                dof_efforts[:, self.prismatic_dof_indices[i]] = prismatic_raw_actions * REVOLUTE_FORCE_SCALING

                dof_efforts[:, self.prismatic_dof_indices[i]] = torch.where(i < prismatic_indexes, PRISMATIC_FORCE_SCALING,
                                                                            torch.where(i > prismatic_indexes, -PRISMATIC_FORCE_SCALING,
                                                                                        PRISMATIC_FORCE_SCALING * prismatic_raw_actions)
                                                                            )
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(dof_efforts))
        elif DOF_MODE == "POSITION":
            # Populate revolute targets: Angle is 0 if beyond prismatic joint
            revolute_actions = (revolute_raw_actions + 1.) / 2. * (self.revolute_dof_uppers -
                                                                   self.revolute_dof_lowers) + self.revolute_dof_lowers
            for i in range(num_prismatic_joints):
                revolute_actions[:, i] = torch.where(i <= prismatic_indexes, revolute_actions[:, i], 0)

            # Populate prismatic targets
            # Compute difference between desired_lengths and current_lengths to see if we want to grow or shrink
            current_lengths = torch.sum(prismatic_dof_pos, dim=1).to(self.device)
            desired_lengths = (prismatic_raw_actions + 1.) / 2. * torch.sum(self.prismatic_dof_uppers)
            difference_lengths = desired_lengths - current_lengths  # +ve if grow, -ve if shrink
            prismatic_actions = torch.zeros(self.num_envs, num_prismatic_joints, device=self.device)

            # Compute remainder_lengths, which is length at the prismatic_index
            remainder_lengths = torch.zeros(self.num_envs, device=self.device)
            for i in range(num_prismatic_joints):
                remainder_lengths[prismatic_indexes == i] = prismatic_dof_pos[prismatic_indexes == i, i]

            # Check if we are at a boundary at which we should go to next or prev joint index
            # New index allowed to be out of bounds, because we only compare indexes to these values (not use them directly)
            go_to_next_joint = (difference_lengths > 0) & (remainder_lengths > JOINT_BUFFER * self.prismatic_dof_uppers[prismatic_indexes.long()])
            go_to_prev_joint = (difference_lengths < 0) & (remainder_lengths < (1 - JOINT_BUFFER) * self.prismatic_dof_uppers[prismatic_indexes.long()] + self.prismatic_dof_lowers[prismatic_indexes.long()])
            modified_prismatic_indexes = torch.where(go_to_next_joint, prismatic_indexes + 1,
                                                     torch.where(go_to_prev_joint, prismatic_indexes - 1,
                                                                 prismatic_indexes))

            # If below prismatic_index, go to max length
            # If above prismatic_index, go to min length
            # If at prismatic_index, go to desired length (clamped to valid range)
            for i in range(num_prismatic_joints):
                prismatic_actions[:, i] = torch.where(i < modified_prismatic_indexes, self.prismatic_dof_uppers[i],
                                                      torch.where(i > modified_prismatic_indexes, self.prismatic_dof_lowers[i],
                                                                  torch.clamp(prismatic_dof_pos[:, i] + difference_lengths,
                                                                              min=self.prismatic_dof_lowers[i],
                                                                              max=self.prismatic_dof_uppers[i])))

            # Fill in position targets
            position_targets = torch.zeros(self.num_envs, self.num_dof, device=self.device)
            position_targets[:, self.revolute_dof_indices] = revolute_actions
            position_targets[:, self.prismatic_dof_indices] = prismatic_actions

            # Apply targets
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(position_targets))
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

        # Draw debug info
        if self.viewer and self.enable_viewer_sync:
            # Create spheres
            visualization_sphere_radius = 0.05
            visualization_sphere_green = gymutil.WireframeSphereGeometry(
                visualization_sphere_radius, 3, 3, color=(0, 1, 0))

            # Draw target
            self.gym.clear_lines(self.viewer)
            for i in range(self.num_envs):
                target_position = self.target_positions[i]
                sphere_pose = gymapi.Transform(gymapi.Vec3(
                    target_position[0], target_position[1], target_position[2]), r=None)
                gymutil.draw_lines(visualization_sphere_green, self.gym, self.viewer, self.envs[i], sphere_pose)


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_vine_reward(dist_to_target, reset_buf, progress_buf, max_episode_length):
    # type: (Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]

    # reward is punishing dist_to_target
    reward = -dist_to_target  # TODO: Improve with reward shaping, eg. reduce control action or length
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    return reward, reset

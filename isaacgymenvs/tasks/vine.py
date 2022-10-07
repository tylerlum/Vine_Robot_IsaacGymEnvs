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

NUM_STATES = 13  # xyz, quat, v_xyz, w_xyz
NUM_XYZ = 3
Z = 1.0
TARGET_POS_MIN, TARGET_POS_MAX = -0.5, 0.5
DOF_MODE = "FORCE"  # "FORCE" OR "POSITION"
# TODO: Self collision check, parameters of movement, mass, diameter


class Vine(VecTask):
    """
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
      * 3 for angles (only applies actions to angles that have length)
      * 1 for lengthen/retract the closest prismatic joint that is not at 0 or full length
    Reward:
      * Dist to target
    Environment:
      * Fixed target
      * Fixed start position
    """

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        # Store cfg file and read in parameters
        self.cfg = cfg
        self.max_episode_length = self.cfg["env"]["maxEpisodeLength"]

        # Must set this before continuing
        self.cfg["env"]["numObservations"] = 15

        self.cfg["env"]["numActions"] = 4  # # revolute dofs + 1

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
        # TODO: Can get other rigid body info
        # rigid_body_names = self.gym.get_asset_rigid_body_dict(self.vine_asset)
        rigid_body_state_by_env = self.rigid_body_state.view(self.num_envs, self.num_rigid_bodies, NUM_STATES)
        self.tip_positions = rigid_body_state_by_env[:, -1, 0:3]

    def refresh_state_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

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
        # TODO: Change pose to hanging
        pose = gymapi.Transform()
        if self.up_axis == 'z':
            pose.p.z = Z
            # asset is rotated z-up by default, no additional rotations needed
            pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        else:
            pose.p.y = Z
            pose.r = gymapi.Quat(-np.sqrt(2)/2, 0.0, 0.0, np.sqrt(2)/2)

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

            if DOF_MODE == "FORCE":
                dof_props['driveMode'][:] = gymapi.DOF_MODE_EFFORT
            elif DOF_MODE == "POSITION":
                dof_props['driveMode'][:] = gymapi.DOF_MODE_POS
            else:
                raise ValueError(f"Invalid DOF_MODE = {DOF_MODE}")
            dof_props['stiffness'][:] = 1.0  # TODO
            dof_props['damping'][:] = 1.0  # TODO
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
        # Set initial lengths
        min_length, max_length = sum(self.prismatic_dof_lowers), sum(self.prismatic_dof_uppers)
        initial_lengths = torch.FloatTensor(len(env_ids)).uniform_(min_length, max_length).to(self.device)

        # Compute prismatic indexes (smallest index i such that prismatic_joint_i < prismatic_joint_limit_i)
        # And remainder_lengths (length of prismatic_joint_i)
        current_lengths = initial_lengths.clone()
        remainder_lengths = initial_lengths.clone()
        num_prismatic_joints = len(self.prismatic_dof_lowers)
        prismatic_indexes = torch.ones(len(env_ids), dtype=torch.int32, device=self.device) * num_prismatic_joints
        for i in range(num_prismatic_joints):
            prev_lengths = current_lengths.clone()  # TODO: Optimize?
            current_lengths -= self.prismatic_dof_uppers[i]
            remainder_lengths[(current_lengths < 0) & (prev_lengths >= 0)
                              ] = prev_lengths[(current_lengths < 0) & (prev_lengths >= 0)]
            prismatic_indexes[(current_lengths < 0) & (prev_lengths >= 0)] = i

        # Set randomized initial revolute dof positions
        num_revolute_joints = len(self.revolute_dof_lowers)
        for i in range(num_revolute_joints):
            self.dof_pos[env_ids, self.revolute_dof_indices[i]] = torch.FloatTensor(len(env_ids)).uniform_(
                self.revolute_dof_lowers[i], self.revolute_dof_uppers[i]).to(self.device)

        # Set initial prismatic dof positions to match sampled initial lengths
        for i in range(num_prismatic_joints):
            self.dof_pos[env_ids, self.prismatic_dof_indices[i]] = torch.where(i < prismatic_indexes, self.prismatic_dof_uppers[i],
                                                                               torch.where(i > prismatic_indexes, self.prismatic_dof_lowers[i],
                                                                                           remainder_lengths))
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
        target_positions = torch.FloatTensor(num_envs, NUM_XYZ).uniform_(TARGET_POS_MIN, TARGET_POS_MAX).to(self.device)
        target_positions[:, 2] = Z
        return target_positions

    def pre_physics_step(self, actions):
        self.raw_actions = actions.clone().to(self.device)

        # Break into revolute and prismatic action
        revolute_raw_actions, prismatic_raw_actions = self.raw_actions[:, :-1], self.raw_actions[:, -1]

        # Compute prismatic indexes (smallest index i such that prismatic_joint_i < prismatic_joint_limit_i)
        # And remainder_lengths (length of prismatic_joint_i)
        prismatic_dof_pos = self.dof_pos[:, self.prismatic_dof_indices]
        num_prismatic_joints = len(self.prismatic_dof_lowers)
        prismatic_indexes = torch.ones(self.num_envs, dtype=torch.int32, device=self.device) * num_prismatic_joints
        for i in range(num_prismatic_joints):
            prismatic_indexes[(prismatic_dof_pos[:, i] < self.prismatic_dof_uppers[i])
                              & (prismatic_indexes == num_prismatic_joints)] = i

        # TODO: Handle edge cases
        #   * full extension, then try to grow (do nothing)
        #   * full extension, then try to shrink (shrink)
        #   * full retraction, then try to grow (grow)
        #   * full retraction, then try to shrink (do nothing)
        #   * try retracting on empty link (need to switch to previous)
        #   * try growing on full link (need to switch to next)

        if DOF_MODE == "FORCE":
            # TODO: IF PREVIOUS LENGTH ALREADY FULL, DO I STILL NEED TO APPLY FORCE? Assume yes
            REVOLUTE_FORCE_SCALING = 1
            PRISMATIC_FORCE_SCALING = 1
            dof_efforts = torch.zeros(self.num_envs, self.num_dof, device=self.device)
            dof_efforts[:, self.revolute_dof_indices] = revolute_raw_actions * REVOLUTE_FORCE_SCALING
            for i in range(num_prismatic_joints):
                dof_efforts[:, self.prismatic_dof_indices[i]] = prismatic_raw_actions * REVOLUTE_FORCE_SCALING

                dof_efforts[:, self.prismatic_dof_indices[i]] = torch.where(i < prismatic_indexes, PRISMATIC_FORCE_SCALING,
                                                                            torch.where(i > prismatic_indexes, -PRISMATIC_FORCE_SCALING,
                                                                                        PRISMATIC_FORCE_SCALING * prismatic_raw_actions)
                                                                            )
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(dof_efforts))
        elif DOF_MODE == "POSITION":
            position_targets = torch.zeros(self.num_envs, self.num_dof, device=self.device)

            # Populate revolute targets
            revolute_actions = (revolute_raw_actions + 1) / 2 * (self.revolute_dof_uppers -
                                                                 self.revolute_dof_lowers) + self.revolute_dof_lowers
            position_targets[:, self.revolute_dof_indices] = revolute_actions

            # Populate prismatic targets
            current_lengths = torch.sum(prismatic_dof_pos, dim=1).to(self.device)
            desired_lengths = prismatic_raw_actions * torch.sum(self.prismatic_dof_uppers)
            difference_lengths = desired_lengths - current_lengths  # +ve if grow, -ve if shrink
            for i in range(num_prismatic_joints):
                position_targets[:, self.prismatic_dof_indices[i]] = torch.where(i < prismatic_indexes, self.prismatic_dof_uppers[i],
                                                                                 torch.where(i > prismatic_indexes, self.prismatic_dof_lowers[i],
                                                                                 torch.where(difference_lengths > 0,
                                                                                             torch.min(
                                                                                                 difference_lengths, self.prismatic_dof_uppers[i]),
                                                                                             torch.max(difference_lengths, self.prismatic_dof_lowers[i]))))

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
            visualization_sphere_radius = 0.1
            visualization_sphere_red = gymutil.WireframeSphereGeometry(
                visualization_sphere_radius, 3, 3, color=(1, 0, 0))

            # Draw target
            self.gym.clear_lines(self.viewer)
            for i in range(self.num_envs):
                target_position = self.target_positions[i]
                sphere_pose = gymapi.Transform(gymapi.Vec3(
                    target_position[0], target_position[1], target_position[2]), r=None)
                gymutil.draw_lines(visualization_sphere_red, self.gym, self.viewer, self.envs[i], sphere_pose)


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_vine_reward(dist_to_target, reset_buf, progress_buf, max_episode_length):
    # type: (Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]

    # reward is punishing dist_to_target
    reward = -dist_to_target  # TODO
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    return reward, reset

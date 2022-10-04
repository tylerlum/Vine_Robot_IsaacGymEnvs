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
# TODO: Self collision check, parameters of movement, mass, diameter


class Vine(VecTask):
    """
    State:
      * 6 Joint positions (3 revolute, 3 prismatic)
      * 1 Dist to target
    Observation:
      * 6 Cos/Sin of Revolute Joint Positions (3 revolute)
      * 3 Prismatic Joint Positions
      * 1 Dist to target
    Action:
      * 1 for lengthen/retract the closest prismatic joint that is not at 0 or full length
      * 3 for angles (only applies actions to angles that have length)
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

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id,
                         headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        self.initialize_state_tensors()

    def initialize_state_tensors(self):
        # Store dof state tensor, and get pos and vel
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

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
        self.revolute_dof_pos = dof_pos[:, self.revolute_dof_indices]
        self.revolute_dof_vel = dof_vel[:, self.revolute_dof_indices]
        self.prismatic_dof_pos = dof_pos[:, self.prismatic_dof_indices]
        self.prismatic_dof_vel = dof_vel[:, self.prismatic_dof_indices]

        # Store rigid body state tensor
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state_tensor)

        # For now, only care about last link to get tip location
        # TODO: Can get other rigid body info
        # rigid_body_names = self.gym.get_asset_rigid_body_dict(self.vine_asset)
        rigid_body_state_by_env = self.rigid_body_state.view(self.num_envs, self.num_rigid_bodies, NUM_STATES)
        self.last_link_rigid_body_state = rigid_body_state_by_env[:, -1, :]
        self.tip_pos = self.last_link_rigid_body_state[:, 0:3]

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

        # Set initial actor poses
        pose = gymapi.Transform()
        if self.up_axis == 'z':
            pose.p.z = 2.0
            # asset is rotated z-up by default, no additional rotations needed
            pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        else:
            pose.p.y = 2.0
            pose.r = gymapi.Quat(-np.sqrt(2)/2, 0.0, 0.0, np.sqrt(2)/2)

        self.vine_handles = []
        self.envs = []
        for i in range(num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            collision_group = i
            collision_filter = 1
            segmentation_id = 0
            vine_handle = self.gym.create_actor(
                env_ptr, self.vine_asset, pose, "vine", group=collision_group, filter=collision_filter, segmentationId=segmentation_id)

            # Set dof properties
            dof_props = self.gym.get_actor_dof_properties(env_ptr, vine_handle)
            dof_props['driveMode'][:] = gymapi.DOF_MODE_POS
            dof_props['stiffness'][:] = 0.0  # TODO
            dof_props['damping'][:] = 0.0  # TODO
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

        num_rigid_bodies = self.gym.get_asset_rigid_body_count(asset)
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
        print(f"num_rigid_bodies = {num_rigid_bodies}")
        print(f"rigid_body_dict = {rigid_body_dict}")
        print(f"joint_dict = {joint_dict}")
        print(f"dof_dict = {dof_dict}")
        print()

    def compute_reward(self):
        # retrieve environment observations from buffer
        dist_to_target = self.obs_buf[:, 9]

        self.rew_buf[:], self.reset_buf[:] = compute_vine_reward(
            dist_to_target, self.reset_buf, self.progress_buf, self.max_episode_length
        )

    def compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        # Refresh tensors
        self.refresh_state_tensors()

        # Compute dist_tip_to_target
        TARGET = torch.tensor([1, 2, 3]).to(self.device)  # TODO
        dist_tip_to_target = torch.linalg.norm(self.tip_pos[env_ids] - TARGET, dim=-1)

        # Populate obs_buf
        self.obs_buf[env_ids, 0:self.num_dof//2] = torch.cos(self.revolute_dof_pos[env_ids, :])
        self.obs_buf[env_ids, self.num_dof//2:self.num_dof] = torch.sin(self.revolute_dof_pos[env_ids, :])
        self.obs_buf[env_ids, self.num_dof:self.num_dof+self.num_dof//2] = self.prismatic_dof_pos[env_ids, :]
        self.obs_buf[env_ids, -1] = dist_tip_to_target

        return self.obs_buf

    def reset_idx(self, env_ids):
        # TODO: Reset init pos
        self.revolute_dof_pos[env_ids, :] = 0.0
        self.revolute_dof_vel[env_ids, :] = 0.0
        self.prismatic_dof_pos[env_ids, :] = 0.0
        self.prismatic_dof_vel[env_ids, :] = 0.0

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        actions_tensor = torch.zeros((self.num_envs, self.num_dof), device=self.device, dtype=torch.float)
        # TODO: Use actions

        position_targets = gymtorch.unwrap_tensor(actions_tensor)
        self.gym.set_dof_position_target_tensor(self.sim, position_targets)

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward()

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_vine_reward(dist_to_target, reset_buf, progress_buf, max_episode_length):
    # type: (Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]

    # reward is punishing dist_to_target
    reward = 1.0 - dist_to_target  # TODO
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    return reward, reset

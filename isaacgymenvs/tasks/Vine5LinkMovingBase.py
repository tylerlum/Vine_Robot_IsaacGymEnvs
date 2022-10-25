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
NORMAL_INIT_XYZ = (0.0, 0.0, 1.5)
NORMAL_QUAT = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

# PARAMETERS
INIT_X, INIT_Y, INIT_Z = NORMAL_INIT_XYZ
INIT_QUAT = NORMAL_QUAT
TARGET_POS_MIN_X, TARGET_POS_MAX_X = 0.0, 0.0
TARGET_POS_MIN_Y, TARGET_POS_MAX_Y = -2.0, 2.0
TARGET_POS_MIN_Z, TARGET_POS_MAX_Z = 0.0, 3.0
DOF_MODE = "FORCE"  # "FORCE" OR "POSITION"
N_REVOLUTE_DOFS = 6
N_PRISMATIC_DOFS = 1
RANDOMIZE_DOF_INIT = True
RANDOMIZE_TARGETS = True

# TODO: Investigate if self collision checks work (probably not)


def print_if(text="", should_print=False):
    if should_print:
        print(text)


class Vine5LinkMovingBase(VecTask):
    """
    State:
      * 7 Joint positions (6 revolute, 1 prismatic)
    Goal:
      * 1 Target Pos
    Observation:
      * 7 joint positions
      * 3 tip position
      * 3 target position
    Action:
      * 7 for joint forces/torques OR
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
        self.cfg["env"]["numObservations"] = N_REVOLUTE_DOFS + N_PRISMATIC_DOFS + NUM_XYZ + NUM_XYZ
        self.cfg["env"]["numActions"] = N_REVOLUTE_DOFS + N_PRISMATIC_DOFS
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
        rigid_body_state_by_env = self.rigid_body_state.view(
            self.num_envs, self.num_rigid_bodies, NUM_STATES)
        self.link_positions = rigid_body_state_by_env[:, :, 0:3]
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
            "PRINT_DEBUG": gymapi.KEY_D,
            "PRINT_DEBUG_IDX_UP": gymapi.KEY_K,
            "PRINT_DEBUG_IDX_DOWN": gymapi.KEY_J,
        }
        self.event_action_to_function = {
            "RESET": self._reset_callback,
            "PAUSE": self._pause_callback,
            "PRINT_DEBUG": self._print_debug_callback,
            "PRINT_DEBUG_IDX_UP": self._print_debug_idx_up_callback,
            "PRINT_DEBUG_IDX_DOWN": self._print_debug_idx_down_callback,
        }
        # Create state variables
        self.PRINT_DEBUG = False
        self.PRINT_DEBUG_IDX = 0

        assert(sorted(list(self.event_action_to_key.keys())) == sorted(list(self.event_action_to_function.keys())))

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
        vine_asset_file = "urdf/Vine5LinkMovingBase.urdf"

        if "asset" in self.cfg["env"]:
            vine_asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           self.cfg["env"]["asset"].get("assetRoot", vine_asset_root))
            vine_asset_file = self.cfg["env"]["asset"].get("assetFileName", vine_asset_file)

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
        assert(num_revolute_dofs + num_prismatic_dofs == self.num_dof)
        assert(num_revolute_dofs == N_REVOLUTE_DOFS)
        assert(num_prismatic_dofs == N_PRISMATIC_DOFS)

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
        assert(self.up_axis == 'z')
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

            for j in range(self.gym.get_asset_dof_count(self.vine_asset)):
                dof_type = self.gym.get_asset_dof_type(self.vine_asset, j)

                # Dof type specific params
                if dof_type == gymapi.DofType.DOF_ROTATION:
                    if j == 0:  # First revolute is different
                        dof_props['stiffness'][j] = 10.0  # TODO: Tune
                        dof_props['damping'][j] = 1.0  # TODO: Tune
                    else:
                        dof_props['stiffness'][j] = 10.0  # TODO: Tune
                        dof_props['damping'][j] = 1.0  # TODO: Tune
                elif dof_type == gymapi.DofType.DOF_TRANSLATION:
                    dof_props['stiffness'][j] = 100.0  # TODO: Tune
                    dof_props['damping'][j] = 1.0   # TODO: Tune
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

        PRINT_ASSET_INFO = True
        if PRINT_ASSET_INFO:
            self._print_asset_info(self.vine_asset)

    def _print_asset_info(self, asset):
        """
        self.num_dof = 7
        DOF 0
        Name:     'slider_to_cart'
        Type:     Translation
        Properties:  (True, -4., 4., 0, 8., 10000., 0., 0., 0., 0.)
        DOF 1
        Name:     'cart_to_link_0'
        Type:     Rotation
        Properties:  (True, -0.52, 0.52, 0, 8., 10000., 0., 0., 0., 0.)
        DOF 2
        Name:     'link_0_to_link_1'
        Type:     Rotation
        Properties:  (True, -0.52, 0.52, 0, 8., 10000., 0., 0., 0., 0.)
        DOF 3
        Name:     'link_1_to_link_2'
        Type:     Rotation
        Properties:  (True, -0.52, 0.52, 0, 8., 10000., 0., 0., 0., 0.)
        DOF 4
        Name:     'link_2_to_link_3'
        Type:     Rotation
        Properties:  (True, -0.52, 0.52, 0, 8., 10000., 0., 0., 0., 0.)
        DOF 5
        Name:     'link_3_to_link_4'
        Type:     Rotation
        Properties:  (True, -0.52, 0.52, 0, 8., 10000., 0., 0., 0., 0.)
        DOF 6
        Name:     'link_4_to_link_5'
        Type:     Rotation
        Properties:  (True, -0.52, 0.52, 0, 8., 10000., 0., 0., 0., 0.)
        self.num_rigid_bodies = 8
        rigid_body_dict = {'cart': 1, 'link_0': 2, 'link_1': 3, 'link_2': 4, 'link_3': 5, 'link_4': 6, 'link_5': 7, 'slider': 0}
        joint_dict = {'cart_to_link_0': 1, 'link_0_to_link_1': 2, 'link_1_to_link_2': 3, 'link_2_to_link_3': 4, 'link_3_to_link_4': 5, 'link_4_to_link_5': 6, 'slider_to_cart': 0}
    dof_dict = {'cart_to_link_0': 1, 'link_0_to_link_1': 2, 'link_1_to_link_2': 3, 'link_2_to_link_3': 4, 'link_3_to_link_4': 5, 'link_4_to_link_5': 6, 'slider_to_cart': 0}
    Box(-1.0, 1.0, (7,), float32) Box(-inf, inf, (13,), float32)
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

        # Populate obs_buf
        self.obs_buf[env_ids, 0:(N_REVOLUTE_DOFS+N_PRISMATIC_DOFS)] = self.dof_pos[env_ids]
        self.obs_buf[env_ids, -6:-3] = self.tip_positions
        self.obs_buf[env_ids, -3:] = self.target_positions

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

    def pre_physics_step(self, actions):
        self.raw_actions = actions.clone().to(self.device)

        if DOF_MODE == "FORCE":
            REVOLUTE_FORCE_SCALING = 1.0
            PRISMATIC_FORCE_SCALING = 1000.0
            dof_efforts = torch.zeros(self.num_envs, self.num_dof, device=self.device)

            # Revolute
            for i in self.revolute_dof_indices:
                dof_efforts[:, i] = self.raw_actions[:, i] * REVOLUTE_FORCE_SCALING
            for i in self.prismatic_dof_indices:
                dof_efforts[:, i] = self.raw_actions[:, i] * PRISMATIC_FORCE_SCALING
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(dof_efforts))
        elif DOF_MODE == "POSITION":
            POSITION_SCALING = 1.0
            position_targets = torch.zeros(self.num_envs, self.num_dof, device=self.device)
            position_targets[:] = self.raw_actions * POSITION_SCALING

            if self.PRINT_DEBUG:
                print(f"self.PRINT_DEBUG_IDX = {self.PRINT_DEBUG_IDX}")
                print(f"self.raw_actions = {self.raw_actions[self.PRINT_DEBUG_IDX]}")
                print(f"position_targets = {position_targets[self.PRINT_DEBUG_IDX]}")
                print()

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

            self.gym.clear_lines(self.viewer)
            for i in range(self.num_envs):
                # Draw target
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

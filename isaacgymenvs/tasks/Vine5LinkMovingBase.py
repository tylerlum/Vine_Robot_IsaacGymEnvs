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
import datetime
from enum import Enum
import logging
import time

from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import to_torch, quat_from_angle_axis
from .base.vec_task import VecTask
import wandb


# Increase pipe size to make the problem easier
PIPE_ADDITIONAL_SCALING = 1.05

# CONSTANTS (RARELY CHANGE)
NUM_STATES = 13  # xyz, quat, v_xyz, w_xyz
XYZ_LIST = ['x', 'y', 'z']
NUM_XYZ = len(XYZ_LIST)
NUM_OBJECT_INFO = 2  # target depth, angle
NUM_RGBA = 4
LENGTH_RAIL = 0.8
N_REVOLUTE_DOFS = 5
N_PRESSURE_ACTIONS = 1
START_POS_IDX, END_POS_IDX = 0, 3
START_QUAT_IDX, END_QUAT_IDX = 3, 7
START_LIN_VEL_IDX, END_LIN_VEL_IDX = 7, 10
START_ANG_VEL_IDX, END_ANG_VEL_IDX = 10, 13

DOF_MODE = gymapi.DOF_MODE_EFFORT

# PARAMETERS (OFTEN CHANGE)
USE_MOVING_BASE = True


class ObservationType(Enum):
    POS_ONLY = "POS_ONLY"
    POS_AND_VEL = "POS_AND_VEL"
    POS_AND_FD_VEL = "POS_AND_FD_VEL"
    POS_AND_PREV_POS = "POS_AND_PREV_POS"
    POS_AND_FD_VEL_AND_OBJ_INFO = "POS_AND_FD_VEL_AND_OBJ_INFO"
    TIP_AND_CART_AND_OBJ_INFO = "TIP_AND_CART_AND_OBJ_INFO"
    NO_FD_TIP_AND_CART_AND_OBJ_INFO = "NO_FD_TIP_AND_CART_AND_OBJ_INFO"


# Rewards
# Brittle: Ensure reward order matches
REWARD_NAMES = ["Position", "Const Negative", "Position Success",
                "Velocity Success", "Velocity", "Rail Velocity Control",
                "FPAM Control", "Rail Velocity Change", "FPAM Change", "Rail Limit",
                "Cart Y", "Tip Y", "Contact Force"]

N_PRISMATIC_DOFS = 1 if USE_MOVING_BASE else 0
INIT_QUAT = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
INIT_X, INIT_Y, INIT_Z = 0.0, 0.0, 1.0

# PIPE_RADIUS = 0.065 * PIPE_ADDITIONAL_SCALING
PIPE_RADIUS = 0.07 * PIPE_ADDITIONAL_SCALING


class Vine5LinkMovingBase(VecTask):
    """
    Observation:
      POS_ONLY
        * Joint positions
        * Tip position
        * Target position
        * current p_fpam
        * previous rail velocity command
      POS_AND_VEL
        * Joint positions
        * Joint velocities
        * Tip position
        * Tip velocity
        * Target position
        * Target velocity
        * current p_fpam
        * previous rail velocity command
      POS_AND_FD_VEL
        * Joint positions
        * Joint velocities (finite difference)
        * Tip position
        * Tip velocity (finite difference)
        * Target position
        * Target velocity (finite difference)
        * current p_fpam
        * previous rail velocity command
      POS_AND_PREV_POS
        * Joint positions
        * Prev joint positions
        * Tip position
        * Prev tip position
        * Target position
        * Prev target position
        * current p_fpam
        * previous rail velocity command
    Action:
      * 1 for u_rail_velocity prismatic joint
      * 1 for u_fpam pressure
    Reward:
      * Weighted sum of many rewards
    Environment:
      * Random target position
      * Random start position
    """

    ##### INITIALIZATION START #####
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        # Store cfg file and read in parameters
        self.cfg = cfg
        self.log_dir = os.path.join('runs', cfg["name"])
        logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                            datefmt='%Y-%m-%d:%H:%M:%S',
                            level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)
        self.time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.max_episode_length = self.cfg["env"]["maxEpisodeLength"]

        # Randomization
        self.vine_randomize = self.cfg["task"]["vine_randomize"]

        # Must set this before continuing
        observation_type = ObservationType[self.cfg["env"]["OBSERVATION_TYPE"]]
        if observation_type == ObservationType.POS_ONLY:
            self.cfg["env"]["numObservations"] = (
                N_REVOLUTE_DOFS + N_PRISMATIC_DOFS + NUM_XYZ + NUM_XYZ + N_PRESSURE_ACTIONS + N_PRISMATIC_DOFS
            )
        elif observation_type in [ObservationType.TIP_AND_CART_AND_OBJ_INFO, ObservationType.NO_FD_TIP_AND_CART_AND_OBJ_INFO]:
            self.cfg["env"]["numObservations"] = (
                2 * (N_PRISMATIC_DOFS + NUM_XYZ + NUM_XYZ) +
                N_PRESSURE_ACTIONS + N_PRISMATIC_DOFS
            )
            self.cfg["env"]["numObservations"] += NUM_OBJECT_INFO
        else:
            self.cfg["env"]["numObservations"] = (
                2 * (N_REVOLUTE_DOFS + N_PRISMATIC_DOFS + NUM_XYZ + NUM_XYZ) +
                N_PRESSURE_ACTIONS + N_PRISMATIC_DOFS
            )
            if observation_type == ObservationType.POS_AND_FD_VEL_AND_OBJ_INFO:
                # Add more observations for object info
                self.cfg["env"]["numObservations"] += NUM_OBJECT_INFO
        self.cfg["env"]["numActions"] = N_PRESSURE_ACTIONS + N_PRISMATIC_DOFS

        self.subscribe_to_keyboard_events()

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id,
                         headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        self.initialize_state_tensors()
        self.target_positions = self.sample_target_positions(self.num_envs)
        self.target_velocities = self.sample_target_velocities(self.num_envs)

        # Rewards
        self.aggregated_rew_buf = torch.zeros_like(self.rew_buf, device=self.device, dtype=self.rew_buf.dtype)

        # Set up reward weights that match REWARD_NAMES
        reward_name_to_weight_dict = {
            "Position": self.cfg['env']['POSITION_REWARD_WEIGHT'],
            "Const Negative": self.cfg['env']['CONST_NEGATIVE_REWARD_WEIGHT'],
            "Position Success": self.cfg['env']['POSITION_SUCCESS_REWARD_WEIGHT'],
            "Velocity Success": self.cfg['env']['VELOCITY_SUCCESS_REWARD_WEIGHT'],
            "Velocity": self.cfg['env']['VELOCITY_REWARD_WEIGHT'],
            "Rail Velocity Control": self.cfg['env']['U_RAIL_VELOCITY_CONTROL_REWARD_WEIGHT'],
            "FPAM Control": self.cfg['env']['U_FPAM_CONTROL_REWARD_WEIGHT'],
            "Rail Velocity Change": self.cfg['env']['RAIL_VELOCITY_CHANGE_REWARD_WEIGHT'],
            "FPAM Change": self.cfg['env']['U_FPAM_CHANGE_REWARD_WEIGHT'],
            "Rail Limit": self.cfg['env']['RAIL_LIMIT_REWARD_WEIGHT'],
            "Cart Y": self.cfg['env']['CART_Y_REWARD_WEIGHT'],
            "Tip Y": self.cfg['env']['TIP_Y_REWARD_WEIGHT'],
            "Contact Force": self.cfg['env']['CONTACT_FORCE_REWARD_WEIGHT'],
        }
        assert (set(REWARD_NAMES) == set(reward_name_to_weight_dict.keys()))
        reward_weights = [reward_name_to_weight_dict[name] for name in REWARD_NAMES]
        self.reward_weights = torch.tensor([reward_weights], device=self.device)

        # Setup viewer camera
        self.index_to_view = int(0.1 * self.num_envs)
        tip_pos = self.tip_positions[self.index_to_view]
        cam_target = gymapi.Vec3(tip_pos[0], tip_pos[1], INIT_Z + 0.09)
        cam_pos = cam_target + gymapi.Vec3(1.0, 0.0, 0.0)
        self.gym.viewer_camera_look_at(self.viewer, self.envs[self.index_to_view], cam_pos, cam_target)

        # Setup camera for taking pictures
        self.camera_properties = gymapi.CameraProperties()
        if self.cfg['env']['USE_NICE_VISUALS']:
            self.camera_properties.width = self.camera_properties.width  # Save storage space
            self.camera_properties.height = self.camera_properties.height  # Save storage space
        else:
            self.camera_properties.width = self.camera_properties.width // 4  # Save storage space
            self.camera_properties.height = self.camera_properties.height // 4  # Save storage space
        self.camera_handle = self.gym.create_camera_sensor(self.envs[self.index_to_view], self.camera_properties)
        self.video_frames = []
        self.num_video_frames = 100
        self.capture_video_every = 1
        self.num_steps = 0
        self.gym.set_camera_location(self.camera_handle, self.envs[self.index_to_view], cam_pos, cam_target)

        # Perform smoothing of actions
        self.smoothed_u_fpam = torch.zeros(self.num_envs, N_PRESSURE_ACTIONS, device=self.device)

        # Dt
        self.dt = self.cfg["sim"]["dt"]
        self.control_dt = self.dt * self.control_freq_inv

        # Keep track of prevs
        self.prev_dof_pos = self.dof_pos.clone()
        self.prev_tip_positions = self.tip_positions.clone()
        self.prev_u_rail_velocity = torch.zeros(self.num_envs, N_PRISMATIC_DOFS, device=self.device)
        self.prev_cart_vel_error = torch.zeros(self.num_envs, 1, device=self.device)
        self.prev_cart_vel = torch.zeros(self.num_envs, 1, device=self.device)

        # Keep track of object info
        self.object_info = torch.zeros(self.num_envs, NUM_OBJECT_INFO, device=self.device)

        # Observation scaling
        self.obs_scaling = torch.ones(self.cfg["env"]["numObservations"], device=self.device)
        if self.cfg["env"]["SCALE_OBSERVATIONS"]:
            observation_type = ObservationType[self.cfg["env"]["OBSERVATION_TYPE"]]
            # BRITTLE: Need to ensure these are reasonable and adjust as observations change
            if observation_type == ObservationType.POS_AND_FD_VEL_AND_OBJ_INFO:
                self.obs_scaling[:] = to_torch([0.12, 0.269, 0.148, 0.249, 0.148, 0.344,
                                                0.67, 2.22, 1.47, 1.14, 0.903, 0.716,
                                                0.0656, 0.238, 0.0656,
                                                0.732, 2.0, 0.732,
                                                0.02, 0.0235, 0.02,
                                                0.732, 2.0, 0.732,
                                                0.845,
                                                0.86,
                                                0.0385,
                                                0.5], dtype=torch.float, device=self.device)
            elif observation_type in [ObservationType.TIP_AND_CART_AND_OBJ_INFO, ObservationType.NO_FD_TIP_AND_CART_AND_OBJ_INFO]:
                self.obs_scaling[:] = to_torch([0.12,  # 0.269, 0.148, 0.249, 0.148, 0.344,
                                                0.67,  # 2.22, 1.47, 1.14, 0.903, 0.716,
                                                0.0656, 0.238, 0.0656,
                                                0.732, 2.0, 0.732,
                                                0.02, 0.0235, 0.02,
                                                0.732, 2.0, 0.732,
                                                0.845,
                                                0.86,
                                                0.0385,
                                                0.5], dtype=torch.float, device=self.device)
            else:
                raise NotImplementedError(f"Observation scaling not implemented for {observation_type}")

        # Log and cache
        self.use_wandb = True
        self.wandb_dict = {}
        self.histogram_observation_data_list = []
        self.histogram_actions_list = []
        self.A = None  # Cache this matrix

        # Hacky solution to contact forces
        self.shelf_contact_force_norms = []
        self.pipe_contact_force_norms = []

        if len(self.cfg['env']['MAT_FILE']) > 0:
            self.mat = self.read_mat_file(self.cfg['env']['MAT_FILE'])
        else:
            self.mat = None

        self.start_time = time.time()

        # Action history for delay
        first_u_rail_velocity = torch.zeros(self.num_envs, N_PRISMATIC_DOFS, device=self.device)
        first_u_fpam = torch.zeros(self.num_envs, N_PRESSURE_ACTIONS, device=self.device)
        self.actions_history = [(first_u_rail_velocity, first_u_fpam) for _ in range(self.cfg["env"]["ACTION_DELAY"])]
        if len(self.cfg['env']['U_TRAJ_MAT_FILE']) > 0:
            print("WARNING - turning off randomize dof init and appending to actions history")
            print(f"USING MAT FILE {self.cfg['env']['U_TRAJ_MAT_FILE']}")
            self.cfg['env']['RANDOMIZE_DOF_INIT'] = False

            # load traj from mat file
            mat = self.read_mat_file(self.cfg['env']['U_TRAJ_MAT_FILE'])
            U_traj = mat['raw_RL_commands']
            n_actions, n_timesteps = U_traj.shape
            assert (n_actions == 2)

            # append onto actions history
            print(f"WARNING: Actions history will use {n_timesteps} timesteps")
            for i in range(n_timesteps):
                u_rail_velocity = torch.zeros(self.num_envs, N_PRISMATIC_DOFS, device=self.device) + U_traj[0, i]
                u_fpam = torch.zeros(self.num_envs, N_PRESSURE_ACTIONS, device=self.device) + U_traj[1, i]
                self.actions_history.append((u_rail_velocity, u_fpam))

    def read_mat_file(self, filename):
        # TODO: Unused right now
        import scipy.io
        mat = scipy.io.loadmat(filename)
        return mat

    def initialize_state_tensors(self):
        # Store dof state tensor, and get pos and vel
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        # Store root states
        root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_state = gymtorch.wrap_tensor(root_state_tensor)

        # Store rigid body state tensor
        # rigid_body_names = self.gym.get_asset_rigid_body_dict(self.vine_asset)
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state_tensor)

        # Store contact force tensor
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.contact_force = gymtorch.wrap_tensor(contact_force_tensor).view(
            self.num_envs, -1, 3)  # shape: num_envs, num_bodies, xyz axis

        arbitrary_idx = 0  # Any index in list would work

        link_names = ["link_0", "link_1", "link_2", "link_3", "link_4"]
        self.link_indices = torch.zeros(len(link_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, link_name in enumerate(link_names):
            link_idx = self.gym.find_actor_rigid_body_index(
                self.envs[arbitrary_idx], self.vine_handles[arbitrary_idx], link_name, gymapi.DOMAIN_ENV)
            self.link_indices[i] = link_idx

        if self.cfg["env"]["CREATE_SHELF"]:
            shelf_link_names = ["shelf_link"]
            self.shelf_link_indices = torch.zeros(
                len(shelf_link_names), dtype=torch.long, device=self.device, requires_grad=False)
            for i, shelf_link_name in enumerate(shelf_link_names):
                shelf_link_idx = self.gym.find_actor_rigid_body_index(
                    self.envs[arbitrary_idx], self.shelf_handles[arbitrary_idx], shelf_link_name, gymapi.DOMAIN_ENV)
                self.shelf_link_indices[i] = shelf_link_idx

        if self.cfg["env"]["CREATE_PIPE"]:
            pipe_link_names = ["base_link"]
            self.pipe_link_indices = torch.zeros(len(pipe_link_names), dtype=torch.long,
                                                 device=self.device, requires_grad=False)
            for i, pipe_link_name in enumerate(pipe_link_names):
                pipe_link_idx = self.gym.find_actor_rigid_body_index(
                    self.envs[arbitrary_idx], self.pipe_handles[arbitrary_idx], pipe_link_name, gymapi.DOMAIN_ENV)
                self.pipe_link_indices[i] = pipe_link_idx

        # Get tip and cart information
        tip_idx = self.gym.find_actor_rigid_body_index(
            self.envs[arbitrary_idx], self.vine_handles[arbitrary_idx], "tip", gymapi.DOMAIN_ENV)
        cart_idx = self.gym.find_actor_rigid_body_index(
            self.envs[arbitrary_idx], self.vine_handles[arbitrary_idx], "cart", gymapi.DOMAIN_ENV)

        rigid_body_state_by_env = self.rigid_body_state.view(
            self.num_envs, -1, NUM_STATES)

        self.link_positions = rigid_body_state_by_env[:, :, START_POS_IDX:END_POS_IDX]
        self.tip_positions = self.link_positions[:, tip_idx]
        self.cart_positions = self.link_positions[:, cart_idx]

        self.link_velocities = rigid_body_state_by_env[:, :, START_LIN_VEL_IDX:END_LIN_VEL_IDX]
        self.tip_velocities = self.link_velocities[:, tip_idx]
        self.cart_velocities = self.link_velocities[:, cart_idx]

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

        # Create objects
        self.pipe_asset = self.get_obstacle_asset(
            "urdf/pipe/urdf/pipe.urdf", vhacd_enabled=True)  # Mesh needs convex decomposition
        self.shelf_asset = self.get_obstacle_asset("urdf/shelf/urdf/shelf.urdf")
        self.custom_shelf_asset = self.get_obstacle_asset("urdf/shelf/urdf/custom_shelf.urdf")
        self.sushi_shelf_asset = self.get_obstacle_asset("urdf/sushi_shelf/urdf/sushi_shelf.urdf", vhacd_enabled=True)
        self.shelf_super_market1_asset = self.get_obstacle_asset(
            "urdf/shelf_super_market1/urdf/shelf_super_market1.urdf")
        self.shelf_super_market2_asset = self.get_obstacle_asset(
            "urdf/shelf_super_market2/urdf/shelf_super_market2.urdf")

        # Create vine asset and store useful variables
        self.vine_asset = self.get_vine_asset()
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
        vine_init_pose.p.z = INIT_Z + 0.09
        vine_init_pose.r = INIT_QUAT

        self.envs = []
        self.vine_handles = []
        self.shelf_handles = []
        self.pipe_handles = []

        self.vine_indices = []
        self.shelf_indices = []
        self.pipe_indices = []
        for i in range(num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            self.envs.append(env_ptr)

            # Different collision_groups so that different envs don't interact
            # collision_filter = 0 for enabled self-collision, collision_filter > 0 disable self-collisions
            collision_group, collision_filter, segmentation_id = i, 0, 0

            # Create shelf
            if self.cfg['env']['CREATE_SHELF']:
                shelf_init_pose = gymapi.Transform()
                shelf_init_pose.p.y = 0.2
                shelf_init_pose.p.z = 0.0
                shelf_handle = self.gym.create_actor(env_ptr, self.custom_shelf_asset, shelf_init_pose, "shelf",
                                                     group=collision_group, filter=collision_filter, segmentationId=segmentation_id + 1)
                shelf_scale = 1.0
                self.gym.set_actor_scale(env_ptr, shelf_handle, shelf_scale)
                self.shelf_indices.append(self.gym.get_actor_index(env_ptr, shelf_handle, gymapi.DOMAIN_SIM))

                self.set_friction(env_ptr=env_ptr, object_handle=shelf_handle, friction_coefficient=0.0)
                self.shelf_handles.append(shelf_handle)

            # Create pipe
            if self.cfg['env']['CREATE_PIPE']:
                pipe_init_pose = gymapi.Transform()
                pipe_init_pose.p.y = -0.4
                pipe_init_pose.p.z = 0.50
                pipe_handle = self.gym.create_actor(env_ptr, self.pipe_asset, pipe_init_pose, "pipe",
                                                    group=collision_group, filter=collision_filter, segmentationId=segmentation_id + 2)
                pipe_scale = 0.001 * PIPE_ADDITIONAL_SCALING
                self.gym.set_actor_scale(env_ptr, pipe_handle, pipe_scale)
                self.pipe_indices.append(self.gym.get_actor_index(env_ptr, pipe_handle, gymapi.DOMAIN_SIM))

                self.set_friction(env_ptr=env_ptr, object_handle=pipe_handle, friction_coefficient=0.0)
                self.pipe_handles.append(pipe_handle)

            # Create vine robots
            vine_handle = self.gym.create_actor(env_ptr, self.vine_asset, vine_init_pose, "vine",
                                                group=collision_group, filter=collision_filter, segmentationId=segmentation_id)
            self.vine_indices.append(self.gym.get_actor_index(env_ptr, vine_handle, gymapi.DOMAIN_SIM))
            self.vine_handles.append(vine_handle)
            self.set_friction(env_ptr=env_ptr, object_handle=vine_handle, friction_coefficient=0.0)

            # Set dof properties
            vine_dof_props = self.gym.get_actor_dof_properties(env_ptr, vine_handle)
            vine_dof_props['driveMode'].fill(DOF_MODE)
            vine_dof_props['damping'].fill(self.cfg['env']['DAMPING'])
            for j in range(self.gym.get_asset_dof_count(self.vine_asset)):
                dof_type = self.gym.get_asset_dof_type(self.vine_asset, j)
                if dof_type not in [gymapi.DofType.DOF_ROTATION, gymapi.DofType.DOF_TRANSLATION]:
                    raise ValueError(f"Invalid dof_type = {dof_type}")

                # Prismatic joint should have no stiffness
                vine_dof_props['stiffness'][j] = self.cfg['env']['STIFFNESS'] if dof_type == gymapi.DofType.DOF_ROTATION else 0.0

            self.gym.set_actor_dof_properties(env_ptr, vine_handle, vine_dof_props)

        self.shelf_indices = to_torch(self.shelf_indices, dtype=torch.long, device=self.device)
        self.pipe_indices = to_torch(self.pipe_indices, dtype=torch.long, device=self.device)
        self.vine_indices = to_torch(self.vine_indices, dtype=torch.long, device=self.device)

        self._print_asset_info(self.vine_asset)

    def get_obstacle_asset(self, asset_file, fix_base_link=True, vhacd_enabled=False):
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")

        obstacle_asset_options = gymapi.AssetOptions()
        obstacle_asset_options.fix_base_link = fix_base_link  # Fixed base for obstacles
        obstacle_asset_options.vhacd_enabled = vhacd_enabled  # Convex decomposition for meshes
        if vhacd_enabled:
            # Numbers copied from isaacgym docs
            obstacle_asset_options.vhacd_params.resolution = 3000000
            obstacle_asset_options.vhacd_params.max_convex_hulls = 16
            obstacle_asset_options.vhacd_params.max_num_vertices_per_ch = 64
        obstacle_asset = self.gym.load_asset(self.sim, asset_root, asset_file, obstacle_asset_options)
        return obstacle_asset

    def set_friction(self, env_ptr, object_handle, friction_coefficient):
        # Set rigid shape properties
        object_rigid_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle)
        # assert(len(object_rigid_shape_props) == self.gym.get_asset_rigid_shape_count(object_asset))  # Sanity check
        for j in range(len(object_rigid_shape_props)):
            object_rigid_shape_props[j].friction = friction_coefficient
        self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle, object_rigid_shape_props)

    def get_vine_asset(self):
        # Find asset file
        vine_asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        vine_asset_file = "urdf/Vine5LinkMovingBase.urdf" if USE_MOVING_BASE else "urdf/Vine5LinkFixedBase.urdf"

        vine_asset_path = os.path.join(vine_asset_root, vine_asset_file)
        vine_asset_root = os.path.dirname(vine_asset_path)
        vine_asset_file = os.path.basename(vine_asset_path)

        # Create vine asset
        vine_asset_options = gymapi.AssetOptions()
        vine_asset_options.fix_base_link = True  # Fixed base for vine
        vine_asset = self.gym.load_asset(self.sim, vine_asset_root, vine_asset_file, vine_asset_options)
        return vine_asset

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

        self.logger.debug(f"self.num_dof = {self.num_dof}")
        for i, (dof_name, dof_prop, dof_type_string) in enumerate(zip(dof_names, dof_props, dof_type_strings)):
            self.logger.debug("DOF %d" % i)
            self.logger.debug("  Name:     '%s'" % dof_name)
            self.logger.debug("  Type:     %s" % dof_type_string)
            self.logger.debug("  Properties:  %r" % dof_prop)
        self.logger.debug("")
        self.logger.debug(f"self.num_rigid_bodies = {self.num_rigid_bodies}")
        self.logger.debug(f"rigid_body_dict = {rigid_body_dict}")
        self.logger.debug(f"joint_dict = {joint_dict}")
        self.logger.debug(f"dof_dict = {dof_dict}")
        self.logger.debug("")
    ##### INITIALIZATION END #####

    ##### WANDB LOGGING START #####
    def log_wandb_dict(self):
        # Only wandb log if working
        if not self.use_wandb:
            return
        try:
            wandb.log(self.wandb_dict)
        except wandb.errors.Error:
            self.logger.warning("Wandb not initialized, no longer trying to log")
            self.use_wandb = False
        self.wandb_dict = {}

    def save_cfg_file_to_wandb(self):
        if not self.use_wandb:
            return

        # BRITTLE: Depends on filename structure
        import re
        all_logdir_files = os.listdir(self.log_dir)
        pattern = re.compile("[a-zA-Z0-9_-]+_rlg_config_dict.pkl")
        cfg_file = sorted([f for f in all_logdir_files if pattern.match(f)])[-1]

        # Sanity check: Check that datetimes are close
        try:
            datetime1_split = re.split('-|_', cfg_file)[:6]
            datetime2_split = re.split('-|_', self.time_str)[:6]
            datetime1 = datetime.datetime(*[int(x) for x in datetime1_split])
            datetime2 = datetime.datetime(*[int(x) for x in datetime2_split])
            time_diff = (datetime2 - datetime1).total_seconds()
            if abs(time_diff) > 10:
                raise ValueError()
            cfg_file_path = os.path.join(self.log_dir, cfg_file)
            self.logger.info(f"Saving cfg file to wandb: {cfg_file_path}")
            wandb.save(cfg_file_path)
        except:
            self.logger.warning("WARNING: Could not save cfg file to wandb")

    def save_model_to_wandb(self):
        if not self.use_wandb:
            return

        if self.num_steps % 100 != 99:
            return

        models_dir = os.path.join(self.log_dir, "nn")
        for model_file in os.listdir(models_dir):
            model_file_path = os.path.join(models_dir, model_file)
            if model_file_path.endswith(".pth") and os.path.getctime(model_file_path) > self.start_time:
                self.logger.info(f"Saving model to wandb: {model_file_path}")
                try:
                    wandb.save(model_file_path)
                except wandb.errors.Error:
                    self.logger.warning("Wandb not initialized, no longer trying to log")
                    self.use_wandb = False
    ##### WANDB LOGGING END #####

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
            "HISTOGRAM": gymapi.KEY_H,
            "VIDEO": gymapi.KEY_C,
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
            "HISTOGRAM": self._histogram_callback,
            "VIDEO": self._video_callback,
        }
        # Create state variables
        self.PRINT_DEBUG = False
        self.PRINT_DEBUG_IDX = 0
        self.MOVE_LEFT_COUNTER = 0
        self.MOVE_RIGHT_COUNTER = 0
        self.MAX_PRESSURE_COUNTER = 0
        self.MIN_PRESSURE_COUNTER = 0
        self.create_histogram_command_from_keyboard_press = False
        self.create_video_command_from_keyboard_press = False

        assert (sorted(list(self.event_action_to_key.keys())) == sorted(list(self.event_action_to_function.keys())))

    def _reset_callback(self):
        self.logger.info("RESETTING")
        all_env_ids = torch.ones_like(self.reset_buf).nonzero(as_tuple=False).squeeze(-1)
        self.reset_idx(all_env_ids)

    def _pause_callback(self):
        self.logger.info("PAUSING")
        time.sleep(1)

    def _print_debug_callback(self):
        self.PRINT_DEBUG = not self.PRINT_DEBUG
        self.logger.info(f"self.PRINT_DEBUG = {self.PRINT_DEBUG}")

    def _print_debug_idx_up_callback(self):
        self.PRINT_DEBUG_IDX += 1
        if self.PRINT_DEBUG_IDX >= self.num_envs:
            self.PRINT_DEBUG_IDX = self.num_envs - 1
        self.logger.info(f"self.PRINT_DEBUG_IDX = {self.PRINT_DEBUG_IDX}")

    def _print_debug_idx_down_callback(self):
        self.PRINT_DEBUG_IDX -= 1
        if self.PRINT_DEBUG_IDX < 0:
            self.PRINT_DEBUG_IDX = 0
        self.logger.info(f"self.PRINT_DEBUG_IDX = {self.PRINT_DEBUG_IDX}")

    def _move_left_callback(self):
        self.MOVE_LEFT_COUNTER = 10
        self.MOVE_RIGHT_COUNTER = 0
        self.logger.info(f"self.MOVE_LEFT_COUNTER = {self.MOVE_LEFT_COUNTER}")

    def _move_right_callback(self):
        self.MOVE_RIGHT_COUNTER = 10
        self.MOVE_LEFT_COUNTER = 0
        self.logger.info(f"self.MOVE_RIGHT_COUNTER = {self.MOVE_RIGHT_COUNTER}")

    def _max_pressure_callback(self):
        self.MAX_PRESSURE_COUNTER = 10
        self.MIN_PRESSURE_COUNTER = 0
        self.logger.info(f"self.MAX_PRESSURE_COUNTER = {self.MAX_PRESSURE_COUNTER}")

    def _min_pressure_callback(self):
        self.MIN_PRESSURE_COUNTER = 10
        self.MAX_PRESSURE_COUNTER = 0
        self.logger.info(f"self.MIN_PRESSURE_COUNTER = {self.MIN_PRESSURE_COUNTER}")

    def _histogram_callback(self):
        self.create_histogram_command_from_keyboard_press = True
        self.histogram_observation_data_list = []
        self.logger.info(
            f"self.create_histogram_command_from_keyboard_press = {self.create_histogram_command_from_keyboard_press}")

    def _video_callback(self):
        self.create_video_command_from_keyboard_press = True
        self.logger.info(
            f"self.create_video_command_from_keyboard_press = {self.create_video_command_from_keyboard_press}")

    ##### KEYBOARD EVENT SUBSCRIPTIONS END #####

    ##### RESET START #####
    def reset_idx(self, env_ids):
        if self.cfg['env']['RANDOMIZE_DOF_INIT']:
            num_revolute_joints = len(self.revolute_dof_lowers)
            for i in range(num_revolute_joints):
                min_angle = max(self.revolute_dof_lowers[i], -math.radians(10))
                max_angle = min(self.revolute_dof_uppers[i], math.radians(10))
                self.dof_pos[env_ids, self.revolute_dof_indices[i]] = torch.FloatTensor(
                    len(env_ids)).uniform_(min_angle, max_angle).to(self.device)

            num_prismatic_joints = len(self.prismatic_dof_lowers)
            for i in range(num_prismatic_joints):
                min_dist = max(self.prismatic_dof_lowers[i], self.cfg['env']['RANDOM_INIT_CART_MIN_Y'])
                max_dist = min(self.prismatic_dof_uppers[i], self.cfg['env']['RANDOM_INIT_CART_MAX_Y'])
                self.dof_pos[env_ids, self.prismatic_dof_indices[i]] = torch.FloatTensor(
                    len(env_ids)).uniform_(min_dist, max_dist).to(self.device)
        else:
            self.dof_pos[env_ids, :] = 0

        # Set dof velocities to 0
        self.dof_vel[env_ids, :] = 0.0
        self.prev_dof_pos[env_ids, :] = self.dof_pos[env_ids, :].clone()

        # TODO: Need to reset prev_tip_positions as well? Need to do forward kinematics
        self.prev_tip_positions[env_ids] = self.tip_positions[env_ids].clone()
        self.prev_u_rail_velocity[env_ids] = torch.zeros(len(env_ids), N_PRISMATIC_DOFS, device=self.device)
        self.prev_cart_vel_error[env_ids] = torch.zeros(len(env_ids), 1, device=self.device)

        # Update dofs
        vine_indices = self.vine_indices[env_ids].to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(vine_indices), len(vine_indices))

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.rew_buf[env_ids] = 0
        self.aggregated_rew_buf[env_ids] = 0

        # New target positions
        self.target_positions[env_ids, :] = self.sample_target_positions(len(env_ids))
        self.target_velocities[env_ids, :] = self.sample_target_velocities(len(env_ids))

        # Reset action history
        for i in range(self.cfg["env"]["ACTION_DELAY"]):
            history_u_rail_velocity, history_u_fpam = self.actions_history[i]
            history_u_rail_velocity[env_ids, :] = 0
            history_u_fpam[env_ids, :] = 0

        if self.cfg['env']['CREATE_SHELF']:
            # Shelf dimensions
            half_shelf_length_y = 0.4 / 2
            shelf_thickness = 0.01

            # How deep we want the target to be
            shelf_depth_target = torch.FloatTensor(len(env_ids)).uniform_(
                self.cfg['env']['MIN_TARGET_DEPTH_IN_OBSTACLE'], self.cfg['env']['MAX_TARGET_DEPTH_IN_OBSTACLE']).to(self.device)

            shelf_pos_offset = torch.zeros(len(env_ids), 3, device=self.device)
            shelf_pos_offset[:, 1] -= half_shelf_length_y
            shelf_pos_offset[:, 1] += shelf_depth_target
            shelf_pos_offset[:, 2] -= shelf_thickness
            shelf_pos = self.target_positions[env_ids, :] + shelf_pos_offset

            self.root_state[self.shelf_indices[env_ids], START_POS_IDX:END_POS_IDX] = shelf_pos
            self.root_state[self.shelf_indices[env_ids], START_LIN_VEL_IDX:END_LIN_VEL_IDX] = 0
            self.root_state[self.shelf_indices[env_ids], START_ANG_VEL_IDX:END_ANG_VEL_IDX] = 0

            shelf_indices = self.shelf_indices[env_ids].to(dtype=torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(
                self.root_state), gymtorch.unwrap_tensor(shelf_indices), len(shelf_indices))

            self.object_info[env_ids, 0] = shelf_depth_target

        if self.cfg['env']["CREATE_PIPE"]:
            # Our goal is the set the pipe pose, give the target position
            # The pipe orientation, theta, should vary based on the target z
            #
            # When theta = 0, the pipe is vertical, with opening facing down
            # When theta = 90, the pipe is horizontal, with opening facing right
            # When theta = 180, the pipe is vertical, with opening facing up
            # effective_z is the distance from the vine base to the target
            # When effective_z is large, it is close to the ground, so theta should be close to 180
            # When effective_z is small, it is far from the ground, so theta should be close to 90
            #
            # We fit a polynomial function to data, theta_prime = f(effective_z)
            # Where f is a cubic that outputs in degrees
            effective_z = INIT_Z - self.target_positions[env_ids, 2]
            polynomial_coefficients = 1.0e+04 * np.array([1.3199, -1.2276, 0.4045, -0.0447])
            theta_prime = torch.deg2rad(to_torch(np.polyval(p=polynomial_coefficients,
                                        x=effective_z.cpu()), device=self.device))
            theta = theta_prime + torch.deg2rad(torch.tensor(90.0, device=self.device))

            x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((len(env_ids), 1))
            orientation = quat_from_angle_axis(theta, x_unit_tensor)

            pipe_target_entrance_depth = torch.FloatTensor(len(env_ids)).uniform_(
                self.cfg['env']['MIN_TARGET_DEPTH_IN_OBSTACLE'], self.cfg['env']['MAX_TARGET_DEPTH_IN_OBSTACLE']).to(self.device)
            pipe_pos_offset_x = to_torch([-PIPE_RADIUS], dtype=torch.float,
                                         device=self.device).repeat((len(env_ids), 1))
            pipe_pos_offset_y = pipe_target_entrance_depth * torch.cos(theta_prime)
            pipe_pos_offset_z = pipe_target_entrance_depth * torch.sin(theta_prime)
            pipe_pos_offset_y -= -PIPE_RADIUS * torch.sin(theta_prime)
            pipe_pos_offset_z -= PIPE_RADIUS * torch.cos(theta_prime)
            pipe_pos_offset = torch.cat([pipe_pos_offset_x, pipe_pos_offset_y.unsqueeze(-1),
                                        pipe_pos_offset_z.unsqueeze(-1)], dim=-1)
            pipe_pos = self.target_positions[env_ids, :] + pipe_pos_offset
            self.root_state[self.pipe_indices[env_ids], START_POS_IDX:END_POS_IDX] = pipe_pos

            self.root_state[self.pipe_indices[env_ids], START_QUAT_IDX:END_QUAT_IDX] = orientation
            self.root_state[self.pipe_indices[env_ids], START_LIN_VEL_IDX:END_LIN_VEL_IDX] = 0
            self.root_state[self.pipe_indices[env_ids], START_ANG_VEL_IDX:END_ANG_VEL_IDX] = 0

            pipe_indices = self.pipe_indices[env_ids].to(dtype=torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(
                self.root_state), gymtorch.unwrap_tensor(pipe_indices), len(pipe_indices))

            self.object_info[env_ids, 0] = pipe_target_entrance_depth
            self.object_info[env_ids, 1] = theta_prime

    def sample_target_positions(self, num_envs):
        target_positions = torch.zeros(num_envs, NUM_XYZ, device=self.device)
        # TODO: move this to init for efficiency
        # IMPORTANT: Tune these angles depending the task, affects the range of target positions

        TARGET_POS_MIN_X, TARGET_POS_MAX_X = 0.0, 0.0  # Ignored dimension
        if USE_MOVING_BASE:
            # TARGET_POS_MIN_Y, TARGET_POS_MAX_Y = -LENGTH_RAIL/2, LENGTH_RAIL/2  # Set to length of rail
            # TODO: Tune the Y limits of target position depending on task and pipe dims/orientation
            TARGET_POS_MIN_Y, TARGET_POS_MAX_Y = self.cfg['env']['MIN_TARGET_Y'], self.cfg['env']['MAX_TARGET_Y']
        else:
            raise NotImplementedError("Not implemented for non-moving base")
        TARGET_POS_MIN_Z, TARGET_POS_MAX_Z = self.cfg['env']['MIN_TARGET_Z'], self.cfg['env']['MAX_TARGET_Z']

        if self.cfg['env']['RANDOMIZE_TARGETS']:
            # TODO Find the best way to set targets

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
    ##### RESET END #####

    ##### PRE PHYSICS STEP START #####
    def pre_physics_step(self, actions):
        if self.mat is not None:
            self.overwrite_with_mat()

        # Compute high level actions
        self.raw_actions = actions.clone().to(self.device)

        # Add noise to actions before scaling
        if self.vine_randomize:
            action_noise = self.cfg["task"]["randomization_parameters"]["ACTION_NOISE_STD"] * \
                torch.randn_like(self.raw_actions)
            self.raw_actions += action_noise

        # Store newest action, use oldest action
        newest_u_rail_velocity, newest_u_fpam = self.raw_actions_to_actions(self.raw_actions)
        self.actions_history.append((newest_u_rail_velocity, newest_u_fpam))
        self.u_rail_velocity, self.u_fpam = self.actions_history.pop(0)

        self.manual_intervention()
        self.smoothed_u_fpam = self.u_fpam_to_smoothed_u_fpam(self.u_fpam, self.smoothed_u_fpam)

        # Store prevs
        self.prev_dof_pos = self.dof_pos.clone()
        self.prev_tip_positions = self.tip_positions.clone()
        self.prev_u_rail_velocity = self.u_rail_velocity.clone()

    def overwrite_with_mat(self):
        # Get dof positions from mat file
        all_cart_pos = self.mat['cart_pos']  # (1, num_steps)
        all_Q = self.mat['Q']  # (5, num_steps)
        _, total_num_timesteps = all_cart_pos.shape
        assert (all_cart_pos.shape == (1, total_num_timesteps))
        assert (all_Q.shape == (5, total_num_timesteps))

        index = self.num_steps % total_num_timesteps
        self.logger.info(f"Currently at {index} / {total_num_timesteps}")
        cart_pos = self.mat['cart_pos'][:, index]  # (1,)
        Q = self.mat['Q'][:, index]  # (5,)

        # Set dof_pos and dof_vel
        self.dof_pos[:, :1] = to_torch(cart_pos, device=self.device)[
            None, ...].repeat_interleave(self.num_envs, dim=0)  # (num_envs, 1)
        self.dof_pos[:, 1:] = to_torch(Q, device=self.device)[None, ...].repeat_interleave(
            self.num_envs, dim=0)  # (num_envs, 5)
        self.dof_vel[:, :] = 0.0

        vine_indices = self.vine_indices.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(vine_indices), len(vine_indices))

        # Get target
        self.target_positions[:] = to_torch(self.mat['moving_target_pos'][:, index], device=self.device)[
            None, ...].repeat_interleave(self.num_envs, dim=0)  # (num_envs, 3)
        self.target_velocities[:] = to_torch(self.mat['target_vel'].squeeze(), device=self.device)[
            None, ...].repeat_interleave(self.num_envs, dim=0)  # (num_envs, 3)
        self.tip_positions = to_torch(self.mat['tip_pos'][:, index], device=self.device)[
            None, ...].repeat_interleave(self.num_envs, dim=0)  # (num_envs, 3)
        self.tip_velocities = to_torch(self.mat['tip_vel'][:, index], device=self.device)[
            None, ...].repeat_interleave(self.num_envs, dim=0)  # (num_envs, 3)

        return

    def raw_actions_to_actions(self, raw_actions):
        # Break apart actions and states
        if N_PRISMATIC_DOFS == 1:
            u_rail_velocity = rescale_to_u_rail_velocity(
                raw_actions[:, 0:1], self.cfg['env']['RAIL_VELOCITY_SCALE'])  # (num_envs, 1)
            u_fpam = rescale_to_u(raw_actions[:, 1:2], self.cfg['env']['FPAM_MIN'],
                                  self.cfg['env']['FPAM_MAX'])  # (num_envs, 1)
        elif N_PRISMATIC_DOFS == 0:
            u_rail_velocity = torch.zeros_like(raw_actions[:, 0:1], device=raw_actions.device)  # (num_envs, 1)
            u_fpam = rescale_to_u(raw_actions, self.cfg['env']['FPAM_MIN'])  # (num_envs, 1)
        else:
            raise ValueError(f"Can't have N_PRISMATIC_DOFS = {N_PRISMATIC_DOFS}")

        return u_rail_velocity, u_fpam

    def u_fpam_to_smoothed_u_fpam(self, u_fpam, smoothed_u_fpam):
        # Compute smoothed u_fpam
        alphas = torch.where(u_fpam > smoothed_u_fpam,
                             self.cfg['env']['SMOOTHING_ALPHA_INFLATE'], self.cfg['env']['SMOOTHING_ALPHA_DEFLATE'])

        smoothed_u_fpam = alphas * smoothed_u_fpam + (1 - alphas) * u_fpam
        return smoothed_u_fpam

    def manual_intervention(self):
        # Manual intervention
        if self.MOVE_LEFT_COUNTER > 0:
            self.u_rail_velocity[:] = -self.cfg['env']['RAIL_VELOCITY_SCALE']
            self.MOVE_LEFT_COUNTER -= 1
        if self.MOVE_RIGHT_COUNTER > 0:
            self.u_rail_velocity[:] = self.cfg['env']['RAIL_VELOCITY_SCALE']
            self.MOVE_RIGHT_COUNTER -= 1

        if self.MAX_PRESSURE_COUNTER > 0:
            self.u_fpam[:] = self.cfg['env']['FPAM_MAX']
            self.MAX_PRESSURE_COUNTER -= 1
        if self.MIN_PRESSURE_COUNTER > 0:
            self.u_fpam[:] = self.cfg['env']['FPAM_MIN']
            self.MIN_PRESSURE_COUNTER -= 1

        if self.cfg['env']['FORCE_U_FPAM']:
            self.u_fpam[:] = 0.0
        if self.cfg['env']['FORCE_U_RAIL_VELOCITY']:
            self.u_rail_velocity[:] = 0.0

    def compute_and_set_dof_actuation_force_tensor(self):
        dof_efforts = torch.zeros(self.num_envs, self.num_dof, device=self.device)

        if N_PRISMATIC_DOFS == 1:
            q = self.dof_pos[:, 1:]  # (num_envs, 5)
            qd = self.dof_vel[:, 1:]  # (num_envs, 5)
        elif N_PRISMATIC_DOFS == 0:
            q = self.dof_pos[:]  # (num_envs, 5)
            qd = self.dof_vel[:]  # (num_envs, 5)
        else:
            raise ValueError(f"Can't have N_PRISMATIC_DOFS = {N_PRISMATIC_DOFS}")

        # Compute torques
        if self.A is None:
            # torque = - Kq - Cqd - b - Bu;
            #        = - [K C diag(b) diag(B)] @ [q; qd; ones(5), u_fpam*ones(5)]
            #        = - A @ x
            K = torch.diag(torch.tensor([0.8385, 1.5400, 1.5109, 1.2887, 0.4347], device=self.device))
            C = torch.diag(torch.tensor([0.0178, 0.0304, 0.0528, 0.0367, 0.0223], device=self.device))
            b = torch.tensor([0.0007, 0.0062, 0.0402, 0.0160, 0.0133], device=self.device)
            B = torch.tensor([0.0247, 0.0616, 0.0779, 0.0498, 0.0268], device=self.device)

            A1 = torch.cat([K, C, torch.diag(b), torch.diag(B)], dim=-1)  # (5, 20)
            self.A = A1[None, ...].repeat_interleave(self.num_envs, dim=0)  # (num_envs, 5, 20)

        if self.vine_randomize:
            A = self.A * torch.FloatTensor(*self.A.shape).uniform_(self.cfg['task']['randomization_parameters']['DYNAMICS_SCALING_MIN'],
                                                                   self.cfg['task']['randomization_parameters']['DYNAMICS_SCALING_MAX']).to(self.A.device)
        else:
            A = self.A

        u_fpam_to_use = self.smoothed_u_fpam if self.cfg['env']['USE_SMOOTHED_FPAM'] else self.u_fpam
        x = torch.cat([q, qd, torch.ones(self.num_envs, N_REVOLUTE_DOFS, device=self.device), u_fpam_to_use *
                       torch.ones(self.num_envs, N_REVOLUTE_DOFS, device=self.device)], dim=1)[..., None]  # (num_envs, 20, 1)
        torques = -torch.matmul(A, x).squeeze().cpu()  # (num_envs, 5, 1) => (num_envs, 5)

        # Compute rail force
        # Previous approach:
        # * given v and v_target, we compute v_target
        # * set force = P * V_MAX

        cart_vel_y = self.cart_velocities[:, 1:2]  # (num_envs, 1)
        cart_vel_error = self.u_rail_velocity - cart_vel_y

        # compute force for acceleration tracking to be used when velocity error is large
        # baseline force is bang-bang control
        RAIL_ACCELERATION = self.cfg['env']['RAIL_ACCELERATION']
        APPROX_MASS = 0.5
        RAIL_FORCE_MAX = RAIL_ACCELERATION * APPROX_MASS
        rail_force_minmax = torch.where(cart_vel_error > 0, torch.tensor(
            RAIL_FORCE_MAX, device=self.device), torch.tensor(-RAIL_FORCE_MAX, device=self.device))

        # fine tune with P control on acceleration
        if self.cfg['env']['USE_CART_ACCEL_TRACKING']:
            # print("Using cart accel tracking")
            accel = (cart_vel_y - self.prev_cart_vel) / self.dt

            accel_target = torch.where(cart_vel_error > 0, torch.tensor(
                RAIL_ACCELERATION, device=self.device), torch.tensor(-RAIL_ACCELERATION, device=self.device))

            # Add noise to the rail acceleration target
            if self.vine_randomize:
                accel_target *= (
                    torch.FloatTensor(*accel_target.shape).uniform_(self.cfg['task']['randomization_parameters']['ACCEL_TARGET_SCALING_MIN'],
                                                                    self.cfg['task']['randomization_parameters']['ACCEL_TARGET_SCALING_MAX']).to(accel_target.device)
                )
            COURSE_P_GAIN = .30
            COURSE_D_GAIN = .01
            adjustment = COURSE_P_GAIN * (accel_target - accel)

            rail_force_minmax += adjustment

            # print(f"accel: {accel[0,0]}\cart_vel_error: {cart_vel_error[0,0]}\tadjustment: {adjustment[0,0]}")

        # compute force for velocity tracking to be used when velocity error is small
        rail_force_pid = self.cfg['env']['RAIL_P_GAIN'] * cart_vel_error + \
            self.cfg['env']['RAIL_D_GAIN'] * (cart_vel_error - self.prev_cart_vel_error)

        # choose between velocity and acceleration tracking for each environment
        self.rail_force = torch.where(torch.abs(cart_vel_error) > 0.1, rail_force_minmax, rail_force_pid)

        self.prev_cart_vel_error = cart_vel_error.detach().clone()
        self.prev_cart_vel = cart_vel_y.detach().clone()

        # Set efforts
        if N_PRISMATIC_DOFS == 1:
            dof_efforts[:, 0:1] = self.rail_force
            dof_efforts[:, 1:] = torques
        elif N_PRISMATIC_DOFS == 0:
            dof_efforts[:, :] = torques
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(dof_efforts))
    ##### PRE PHYSICS STEP END #####

    ##### POST PHYSICS STEP START #####
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
            visualization_sphere_radius = self.cfg['env']['SUCCESS_DIST']

            if self.cfg['env']['USE_NICE_VISUALS']:
                num_lats, num_lons = 100, 100
            else:
                num_lats, num_lons = 3, 3

            visualization_sphere_green = gymutil.WireframeSphereGeometry(
                radius=visualization_sphere_radius, num_lats=num_lats, num_lons=num_lons, color=(0.17647059, 0.63137255, 0.13333333))

            self.gym.clear_lines(self.viewer)
            # Draw target
            for i in range(self.num_envs):
                target_position = self.target_positions[i]
                sphere_pose = gymapi.Transform(gymapi.Vec3(
                    target_position[0], target_position[1], target_position[2]), r=None)
                gymutil.draw_lines(visualization_sphere_green, self.gym, self.viewer, self.envs[i], sphere_pose)

            # Draw episode progress
            for i in range(self.num_envs):
                if self.cfg['env']['USE_NICE_VISUALS']:
                    break

                # For now, draw only one env to save time
                if i != self.index_to_view:
                    continue

                left_most_pos = gymapi.Vec3(0, -LENGTH_RAIL/2, INIT_Z + 0.2)
                right_most_pos = gymapi.Vec3(0, LENGTH_RAIL/2, INIT_Z + 0.2)
                fraction_complete = self.progress_buf[i] / self.max_episode_length

                pos1_vec3 = left_most_pos
                pos2_vec3 = left_most_pos + gymapi.Vec3(0, fraction_complete * (right_most_pos.y - left_most_pos.y), 0)
                green_color = gymapi.Vec3(0.1, 0.9, 0.1)
                gymutil.draw_line(pos1_vec3, pos2_vec3, green_color, self.gym, self.viewer, self.envs[i])

            # Draw rail soft limits
            for i in range(self.num_envs):
                if self.cfg['env']['USE_NICE_VISUALS']:
                    break

                # For now, draw only one env to save time
                if i != self.index_to_view:
                    continue

                half_line_length = 0.1
                center = gymapi.Vec3(0, 0, INIT_Z)
                left_line_bottom = center + gymapi.Vec3(0, -self.cfg['env']['RAIL_SOFT_LIMIT'], -half_line_length)
                left_line_top = left_line_bottom + gymapi.Vec3(0, 0, 2 * half_line_length)
                right_line_bottom = center + gymapi.Vec3(0, self.cfg['env']['RAIL_SOFT_LIMIT'], -half_line_length)
                right_line_top = right_line_bottom + gymapi.Vec3(0, 0, 2 * half_line_length)

                red_color = gymapi.Vec3(0.9, 0.1, 0.1)
                gymutil.draw_line(left_line_bottom, left_line_top, red_color, self.gym, self.viewer, self.envs[i])
                gymutil.draw_line(right_line_bottom, right_line_top, red_color, self.gym, self.viewer, self.envs[i])

        # Create video
        should_start_video_capture = (self.num_steps % self.capture_video_every ==
                                      0) or self.create_video_command_from_keyboard_press
        video_capture_in_progress = len(self.video_frames) > 0
        if self.cfg['env']['CAPTURE_VIDEO'] and (should_start_video_capture or video_capture_in_progress):
            self.create_video_command_from_keyboard_press = False
            if not video_capture_in_progress:
                self.logger.info("-" * 100)
                self.logger.info("Starting to capture video frames...")
                self.logger.info("-" * 100)
                self.enable_viewer_sync_before = self.enable_viewer_sync

            # Store image
            self.enable_viewer_sync = True
            self.gym.render_all_camera_sensors(self.sim)
            color_image = self.gym.get_camera_image(self.sim, self.envs[self.index_to_view], self.camera_handle, gymapi.IMAGE_COLOR).reshape(
                self.camera_properties.height, self.camera_properties.width, NUM_RGBA)
            self.video_frames.append(color_image)

            if len(self.video_frames) == self.num_video_frames:
                # Save to file and wandb
                video_filename = f"{self.time_str}_video_{self.num_steps}.gif"
                video_path = os.path.join(self.log_dir, video_filename)
                self.logger.info("-" * 100)
                self.logger.info(f"Saving video to {video_path}...")

                if not self.enable_viewer_sync_before:
                    self.video_frames.pop(0)  # Remove first frame because it was not synced

                import imageio
                imageio.mimsave(video_path, self.video_frames)
                self.wandb_dict["video"] = wandb.Video(video_path, fps=1./self.control_dt)
                self.logger.info("DONE")
                self.logger.info("-" * 100)

                # Reset variables
                self.video_frames = []
                self.enable_viewer_sync = self.enable_viewer_sync_before

        self.num_steps += 1

        # Log info
        self.log_wandb_dict()

        # Save model
        self.save_model_to_wandb()
        if self.num_steps == 1:
            self.save_cfg_file_to_wandb()

    def compute_reward(self):
        dist_tip_to_target = torch.linalg.norm(self.tip_positions - self.target_positions, dim=-1)

        # Target reached
        ONLY_CARE_ABOUT_Y_TARGET = False
        if ONLY_CARE_ABOUT_Y_TARGET:
            tip_y = self.tip_positions[:, 1]
            target_y = self.target_positions[:, 1]
            target_reached = tip_y < target_y  # More negative in y dir BRITTLE
        else:
            target_reached = dist_tip_to_target < self.cfg['env']['SUCCESS_DIST']

        # Limit hit
        cart_y = self.cart_positions[:, 1]
        limit_hit = torch.logical_or(cart_y > self.cfg['env']['RAIL_SOFT_LIMIT'],
                                     cart_y < -self.cfg['env']['RAIL_SOFT_LIMIT'])

        # Tip y exceeds target y
        tip_y = self.tip_positions[:, 1]
        tip_limit_hit = tip_y < self.target_positions[:, 1]

        # Get contact forces
        if self.cfg["env"]["CREATE_SHELF"]:
            assert (len(self.shelf_contact_force_norms) > 0)
            contact_force_norms = torch.stack(self.shelf_contact_force_norms, dim=0)  # (control_freq_inv, num_envs)
            contact_force_norm = torch.mean(contact_force_norms, dim=0)  # (num_envs)
            nonzero_contact_force = contact_force_norm > 0
        else:
            # No contact forces stuff for pipe for now
            contact_force_norm = torch.zeros(self.num_envs, device=self.device)
            nonzero_contact_force = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.wandb_dict.update({
            "dist_tip_to_target": dist_tip_to_target.mean().item(),
            "target_reached": target_reached.float().mean().item(),
            "limit_hit": limit_hit.float().mean().item(),
            "tip_limit_hit": tip_limit_hit.float().mean().item(),
            "abs_tip_y": self.tip_positions[:, 1].abs().mean().item(),
            "tip_z": self.tip_positions[:, 2].mean().item(),
            "max_abs_tip_y": self.tip_positions[:, 1].abs().max().item(),
            "max_tip_z": self.tip_positions[:, 2].max().item(),
            "tip_velocities": torch.norm(self.tip_velocities, dim=-1).mean().item(),
            "tip_velocities_max": torch.norm(self.tip_velocities, dim=-1).max().item(),
            "u_rail_velocity": torch.norm(self.u_rail_velocity, dim=-1).mean().item(),
            "prev_u_rail_velocity": torch.norm(self.prev_u_rail_velocity, dim=-1).mean().item(),
            "rail_force": torch.norm(self.rail_force, dim=-1).mean().item(),
            "u_fpam": torch.norm(self.u_fpam, dim=-1).mean().item(),
            "smoothed_u_fpam": torch.norm(self.smoothed_u_fpam, dim=-1).mean().item(),
            "tip_target_velocity_difference": torch.norm(self.tip_velocities - self.target_velocities, dim=-1).mean().item(),
            "progress_buf": self.progress_buf.float().mean().item(),
            "contact_forces": contact_force_norm.mean().item(),
            "nonzero_contact_force": nonzero_contact_force.float().mean().item(),
        })

        self.rew_buf[:], reward_matrix, weighted_reward_matrix = compute_reward_jit(
            dist_to_target=dist_tip_to_target, target_reached=target_reached, tip_velocities=self.tip_velocities,
            target_velocities=self.target_velocities, u_rail_velocity=self.u_rail_velocity, u_fpam=self.u_fpam, prev_u_rail_velocity=self.prev_u_rail_velocity,
            smoothed_u_fpam=self.smoothed_u_fpam, limit_hit=limit_hit, tip_limit_hit=tip_limit_hit, cart_y=cart_y,
            contact_force_norm=contact_force_norm, reward_weights=self.reward_weights, reward_names=REWARD_NAMES
        )
        self.aggregated_rew_buf += self.rew_buf

        self.wandb_dict.update({
            "Aggregated Reward": self.aggregated_rew_buf.mean().item(),
            "Aggregated Reward 1 Std Up": self.aggregated_rew_buf.mean().item() + self.aggregated_rew_buf.std().item(),
            "Aggregated Reward 1 Std Down": self.aggregated_rew_buf.mean().item() - self.aggregated_rew_buf.std().item(),
        })

        # Log input and output
        for i, idx in enumerate(self.prismatic_dof_indices):
            self.wandb_dict[f"prismatic_q{i} at self.index_to_view"] = self.dof_pos[self.index_to_view, idx]
            self.wandb_dict[f"prismatic_qd{i} at self.index_to_view"] = self.dof_vel[self.index_to_view, idx]
            self.wandb_dict[f"prismatic_finite_diff_qd{i} at self.index_to_view"] = self.finite_difference_dof_vel[self.index_to_view, idx]

        for i, idx in enumerate(self.revolute_dof_indices):
            self.wandb_dict[f"q{i} at self.index_to_view"] = self.dof_pos[self.index_to_view, idx]
            self.wandb_dict[f"qd{i} at self.index_to_view"] = self.dof_vel[self.index_to_view, idx]
            self.wandb_dict[f"finite_diff_qd{i} at self.index_to_view"] = self.finite_difference_dof_vel[self.index_to_view, idx]

        for i, dir in enumerate(XYZ_LIST):
            self.wandb_dict[f"tip_vel_{dir} at self.index_to_view"] = self.tip_velocities[self.index_to_view, i]
            self.wandb_dict[f"cart_vel_{dir} at self.index_to_view"] = self.cart_velocities[self.index_to_view, i]
            self.wandb_dict[f"target_vel_{dir} at self.index_to_view"] = self.target_velocities[self.index_to_view, i]
            self.wandb_dict[f"finite_diff_tip_vel_{dir} at self.index_to_view"] = self.finite_difference_tip_velocities[self.index_to_view, i]

            self.wandb_dict[f"tip_pos_{dir} at self.index_to_view"] = self.tip_positions[self.index_to_view, i]
            self.wandb_dict[f"cart_pos_{dir} at self.index_to_view"] = self.cart_positions[self.index_to_view, i]
            self.wandb_dict[f"target_pos_{dir} at self.index_to_view"] = self.target_positions[self.index_to_view, i]

        self.wandb_dict["u_fpam at self.index_to_view"] = self.u_fpam[self.index_to_view]
        self.wandb_dict["smoothed u_fpam at self.index_to_view"] = self.smoothed_u_fpam[self.index_to_view]
        self.wandb_dict["u_rail_velocity at self.index_to_view"] = self.u_rail_velocity[self.index_to_view]
        self.wandb_dict["rail_force at self.index_to_view"] = self.rail_force[self.index_to_view]
        self.wandb_dict["contact_force at self.index_to_view"] = contact_force_norm[self.index_to_view]
        self.wandb_dict["nonzero_contact_force at self.index_to_view"] = nonzero_contact_force[self.index_to_view]

        for i, reward_name in enumerate(REWARD_NAMES):
            self.wandb_dict.update({
                f"Mean {reward_name} Reward": reward_matrix[:, i].mean().item(),
                f"Max {reward_name} Reward": reward_matrix[:, i].max().item(),
                f"Weighted Mean {reward_name} Reward": weighted_reward_matrix[:, i].mean().item(),
                f"Weighted Max {reward_name} Reward": weighted_reward_matrix[:, i].max().item(),
                f"Mean Total Reward": self.rew_buf.mean().item(),
                f"Max Total Reward": self.rew_buf.max().item(),
            })

        self.reset_buf[:] = compute_reset_jit(
            reset_buf=self.reset_buf, progress_buf=self.progress_buf,
            max_episode_length=self.max_episode_length, target_reached=target_reached, limit_hit=limit_hit, tip_limit_hit=tip_limit_hit,
            nonzero_contact_force=nonzero_contact_force,
            use_target_reached_reset=self.cfg['env']['USE_TARGET_REACHED_RESET'],
            use_tip_limit_hit_reset=self.cfg['env']['USE_TIP_LIMIT_HIT_RESET'],
            use_nonzero_contact_force_reset=self.cfg['env']['USE_NONZERO_CONTACT_FORCE_RESET'],
        )

    def refresh_state_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        # Refresh tensors
        self.refresh_state_tensors()

        # Finite difference to get velocities
        self.finite_difference_dof_vel = (self.dof_pos - self.prev_dof_pos) / self.control_dt
        self.finite_difference_tip_velocities = (self.tip_positions - self.prev_tip_positions) / self.control_dt

        # Populate obs_buf
        # tensors_to_add elements must all be (num_envs, X)
        observation_type = ObservationType[self.cfg["env"]["OBSERVATION_TYPE"]]

        if observation_type == ObservationType.POS_ONLY:
            tensors_to_concat = [self.dof_pos, self.tip_positions, self.target_positions,
                                 self.smoothed_u_fpam, self.prev_u_rail_velocity]
        elif observation_type == ObservationType.POS_AND_VEL:
            tensors_to_concat = [self.dof_pos, self.dof_vel, self.tip_positions, self.tip_velocities,
                                 self.target_positions, self.target_velocities,
                                 self.smoothed_u_fpam, self.prev_u_rail_velocity]
        elif observation_type == ObservationType.POS_AND_FD_VEL:
            tensors_to_concat = [self.dof_pos, self.finite_difference_dof_vel, self.tip_positions, self.finite_difference_tip_velocities,
                                 self.target_positions, self.target_velocities,
                                 self.smoothed_u_fpam, self.prev_u_rail_velocity]
        elif observation_type == ObservationType.POS_AND_PREV_POS:
            tensors_to_concat = [self.dof_pos, self.prev_dof_pos, self.tip_positions, self.prev_tip_positions,
                                 self.target_positions, self.target_velocities,
                                 self.smoothed_u_fpam, self.prev_u_rail_velocity]
        elif observation_type == ObservationType.POS_AND_FD_VEL_AND_OBJ_INFO:
            tensors_to_concat = [self.dof_pos, self.finite_difference_dof_vel, self.tip_positions, self.finite_difference_tip_velocities,
                                 self.target_positions, self.target_velocities,
                                 self.smoothed_u_fpam, self.prev_u_rail_velocity,
                                 self.object_info]
        elif observation_type == ObservationType.TIP_AND_CART_AND_OBJ_INFO:
            tensors_to_concat = [self.dof_pos[:, :1], self.finite_difference_dof_vel[:, :1], self.tip_positions, self.finite_difference_tip_velocities,
                                 self.target_positions, self.target_velocities,
                                 self.smoothed_u_fpam, self.prev_u_rail_velocity,
                                 self.object_info]
        elif observation_type == ObservationType.NO_FD_TIP_AND_CART_AND_OBJ_INFO:
            tensors_to_concat = [self.dof_pos[:, :1], self.dof_vel[:, :1], self.tip_positions, self.tip_velocities,
                                 self.target_positions, self.target_velocities,
                                 self.smoothed_u_fpam, self.prev_u_rail_velocity,
                                 self.object_info]
        else:
            raise NotImplementedError(f"Observation type {observation_type} not implemented.")

        self.obs_buf[:] = torch.cat(tensors_to_concat, dim=-1)

        # Scale observations
        self.obs_buf = self.obs_buf / self.obs_scaling

        # Add obs noise
        if self.vine_randomize:
            obs_noise = self.cfg["task"]["randomization_parameters"]["OBSERVATION_NOISE_STD"] * \
                torch.randn_like(self.obs_buf)
            self.obs_buf += obs_noise

        if self.cfg['env']['CREATE_HISTOGRAMS_PERIODICALLY'] or self.create_histogram_command_from_keyboard_press:
            if len(self.histogram_observation_data_list) == 0:
                self.logger.info("-" * 100)
                self.logger.info("Starting to store observation data for histogram...")
                self.logger.info("-" * 100)

            # Store observations (from all envs or just one)
            HISTOGRAM_USING_ALL_ENVS = False
            if HISTOGRAM_USING_ALL_ENVS:
                new_data = [self.obs_buf[i, :].cpu().numpy().tolist() for i in range(self.obs_buf.shape[0])
                            ]  # list of lists (inner list has length num_obs)
            else:
                new_data = [self.obs_buf[self.index_to_view, :].cpu().numpy().tolist()]

            # self.histogram_observation_data_list is a list of lists (outer list has length num_rows)
            self.histogram_observation_data_list += new_data
            self.histogram_actions_list += [self.raw_actions[self.index_to_view, :].cpu().numpy().tolist()]  # HACK

            if len(self.histogram_observation_data_list) == 100 * len(new_data):
                self.logger.info("-" * 100)
                self.logger.info(f"Creating histogram at self.num_steps {self.num_steps}...")

                # Save histogram data to file
                SAVE_HISTOGRAM_DATA_TO_FILE_AND_EXIT = False
                if SAVE_HISTOGRAM_DATA_TO_FILE_AND_EXIT:
                    import pickle
                    filename = f"{self.time_str}_histogram_data_{self.num_steps}.pkl"
                    print(f"Saving to {filename}")
                    with open(filename, "wb") as f:
                        pickle.dump(self.histogram_observation_data_list, f)
                    print(f"Done saving to {filename}")
                    filename = f"{self.time_str}_action_data_{self.num_steps}.pkl"
                    print(f"Saving to {filename}")
                    with open(filename, "wb") as f:
                        pickle.dump(self.histogram_actions_list, f)
                    print(f"Done saving to {filename}")
                    exit()

                # BRITTLE: Depends on observations above
                observation_names = [*[f"joint_pos_{i}" for i in range(self.num_dof)],
                                     *[f"joint_vel_{i}" for i in range(self.num_dof)],
                                     *[f"tip_pos_{i}" for i in XYZ_LIST],
                                     *[f"tip_vel_{i}" for i in XYZ_LIST],
                                     *[f"target_pos_{i}" for i in XYZ_LIST],
                                     *[f"target_vel_{i}" for i in XYZ_LIST],
                                     "smoothed_u_fpam", "prev_u_rail_vel",
                                     "target_depth", "target_angle"]
                # Each entry is a row in the table
                table = wandb.Table(data=self.histogram_observation_data_list, columns=observation_names)
                ALL_HISTOGRAMS = True
                names_to_plot = observation_names if ALL_HISTOGRAMS else [
                    "tip_pos_y", "tip_pos_z", "tip_vel_y", "tip_vel_z"]
                histograms_dict = {f'{name}_histogram {self.num_steps}': wandb.plot.histogram(
                    table, name, title=f"{name} Histogram {self.num_steps}") for name in names_to_plot}
                wandb.log(histograms_dict)
                self.logger.info("DONE")
                self.logger.info("-" * 100)

                # Reset
                self.histogram_observation_data_list = []
                self.create_histogram_command_from_keyboard_press = False

        return self.obs_buf
    ##### POST PHYSICS STEP END #####


def rescale_to_u(u_fpam, min, max):
    return (u_fpam + 1.0) / 2.0 * (max - min) + min


def rescale_to_u_rail_velocity(u_rail_velocity, scale):
    return u_rail_velocity * scale

#######################################################################
### =========================jit functions========================= ###
#######################################################################


@torch.jit.script
def compute_reward_jit(dist_to_target, target_reached, tip_velocities,
                       target_velocities, u_rail_velocity, u_fpam, prev_u_rail_velocity,
                       smoothed_u_fpam, limit_hit, tip_limit_hit, cart_y, contact_force_norm,
                       reward_weights, reward_names):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, List[str]) -> Tuple[Tensor, Tensor, Tensor]
    # reward = sum(w_i * r_i) with various reward function r_i and weights w_i

    # position_reward = -dist_to_target [Try to reach target]
    # const_negative_reward = -1 [Punish for not succeeding]
    # position_success_reward = REWARD_BONUS if dist_to_target < SUCCESS_DISTANCE else 0 [Succeed if close enough]
    # velocity_success_reward = -norm(tip_velocity - desired_tip_velocity) if dist_to_target < SUCCESS_DISTANCE else 0 [Succeed if close enough and moving at the right speed]
    # velocity_reward = norm(tip_velocity) [Try to move fast]
    # u_rail_velocity_control_reward = -norm(u_rail_velocity) [Punish for using too much actuation]
    # u_control_reward = -norm(u_fpam) [Punish for using too much actuation]
    # u_rail_velocity_change_reward = -norm(u_rail_velocity - prev_u_rail_velocity) [Punish for changing u_rail_velocity]
    # u_change_reward = -norm(u_fpam - smoothed_u_fpam) [Punish for changing u_fpam]
    # rail_limit_reward = RAIL_LIMIT_PUNISHMENT if limit_hit else 0 [Punish for hitting rail limits]
    # cart_y_reward = -abs(cart_y) [Punish for getting near rail limits]
    # tip_y_reward = TIP_LIMIT_PUNISHMENT if tip_limit_hit else 0 [Punish for exceeding target]
    # contact_force_reward = contact_force_norm if contact_force_norm > THRESH else 0 [Punish for large contact]
    N_REWARDS = torch.numel(reward_weights)
    N_ENVS = dist_to_target.shape[0]

    REWARD_BONUS = 1000.0
    RAIL_LIMIT_PUNISHMENT = -100.0
    TIP_LIMIT_PUNISHMENT = -100.0
    CONTACT_FORCE_THRESHOLD = 0.0

    # Brittle: Ensure reward order matches
    reward_matrix = torch.zeros(N_ENVS, N_REWARDS, device=dist_to_target.device)
    for i, reward_name in enumerate(reward_names):
        if reward_name == "Position":
            reward_matrix[:, i] -= dist_to_target
        elif reward_name == "Const Negative":
            reward_matrix[:, i] -= 1
        elif reward_name == "Position Success":
            reward_matrix[:, i] += torch.where(target_reached, REWARD_BONUS, 0.0)
        elif reward_name == "Velocity Success":
            reward_matrix[:, i] -= torch.where(target_reached,
                                               torch.norm(tip_velocities - target_velocities, dim=-1).double(),
                                               0.0)
        elif reward_name == "Velocity":
            reward_matrix[:, i] += torch.norm(tip_velocities, dim=-1)
        elif reward_name == "Rail Velocity Control":
            reward_matrix[:, i] -= torch.norm(u_rail_velocity, dim=-1)
        elif reward_name == "FPAM Control":
            reward_matrix[:, i] -= torch.norm(u_fpam, dim=-1)
        elif reward_name == "Rail Velocity Change":
            reward_matrix[:, i] -= torch.norm(u_rail_velocity - prev_u_rail_velocity, dim=-1)
        elif reward_name == "FPAM Change":
            reward_matrix[:, i] -= torch.norm(u_fpam - smoothed_u_fpam, dim=-1)
        elif reward_name == "Rail Limit":
            reward_matrix[:, i] += torch.where(limit_hit, RAIL_LIMIT_PUNISHMENT, 0.0)
        elif reward_name == "Cart Y":
            # reward_matrix[:, i] -= torch.abs(cart_y)
            # TODO Generalize this if we keep
            reward_matrix[:, i] += torch.where(torch.abs(cart_y) > 0.2, RAIL_LIMIT_PUNISHMENT/10, 0.0)
        elif reward_name == "Tip Y":
            reward_matrix[:, i] += torch.where(tip_limit_hit, TIP_LIMIT_PUNISHMENT, 0.0)
        elif reward_name == "Contact Force":
            force_norms = contact_force_norm.double()
            reward_matrix[:, i] -= torch.where(force_norms > CONTACT_FORCE_THRESHOLD, force_norms, 0.0)
        else:
            raise ValueError(f"Invalid reward name: {reward_name}")

    weighted_reward_matrix = reward_matrix * reward_weights
    total_reward = torch.sum(weighted_reward_matrix, dim=-1)

    return total_reward, reward_matrix, weighted_reward_matrix


def compute_reset_jit(reset_buf, progress_buf, max_episode_length, target_reached,
                      limit_hit, tip_limit_hit, nonzero_contact_force,
                      use_target_reached_reset, use_tip_limit_hit_reset, use_nonzero_contact_force_reset):
    # type: (Tensor, Tensor, float, Tensor, Tensor, Tensor, Tensor, bool, bool, bool) -> Tensor
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    reset_for_target_reached = torch.logical_and(target_reached, torch.tensor(
        [use_target_reached_reset], device=target_reached.device))
    reset = torch.where(reset_for_target_reached, torch.ones_like(reset), reset)

    reset_for_tip_limit_hit = torch.logical_and(tip_limit_hit, torch.tensor(
        [use_tip_limit_hit_reset], device=tip_limit_hit.device))
    reset = torch.where(reset_for_tip_limit_hit, torch.ones_like(reset), reset)
    reset = torch.where(limit_hit, torch.ones_like(reset), reset)

    reset_for_nonzero_contact_force = torch.logical_and(nonzero_contact_force, torch.tensor(
        [use_nonzero_contact_force_reset], device=nonzero_contact_force.device))
    reset = torch.where(reset_for_nonzero_contact_force, torch.ones_like(reset), reset)
    return reset

# buer_env.py

import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class BuerEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, device="cuda"):
        self.device = torch.device(device)

        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]

        self.simulate_action_latency = True
        self.dt = 0.02
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=env_cfg["substeps"]),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(1.5, 1.5, 1.0),
                camera_lookat=(0.0, 0.0, 0.3),
            ),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        # add ground
        self.ground = self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # add robot
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="models/buer_test.urdf",
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )

        # build
        self.scene.build(n_envs=num_envs)

        # names to indices
        self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in self.env_cfg["dof_names"]]

        # PD control parameters
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motor_dofs)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motor_dofs)

        # prepare reward functions
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # initialize buffers
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(self.num_envs, 1)
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor([self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]], device=self.device)
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.default_dof_pos = torch.tensor([self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["dof_names"]], device=self.device)
        self.extras = dict()

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), self.device)

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
        self.scene.step()

        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat))
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        # world-frame velocities:
        # self.base_lin_vel[:] = self.robot.get_vel()
        # self.base_ang_vel[:] = self.robot.get_ang()

        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)

        envs_idx = ((self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0).nonzero(as_tuple=False).flatten())
        self._resample_commands(envs_idx)

        self.reset_buf = self.episode_length_buf > self.max_episode_length

        # Fall detection: check if the robot is too tilted
        base_up_vector = gs.utils.geom.transform_by_quat(
            torch.tensor([0.0, 0.0, 1.0], device=self.device).expand_as(self.base_pos),
            self.base_quat
        )
        self.reset_buf |= (base_up_vector[:, 2].abs() > self.env_cfg["termination_z_threshold"])

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0
        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"],       # 3
                self.projected_gravity,                              # 3
                self.commands * self.commands_scale,                 # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"], # 15
                self.dof_vel * self.obs_scales["dof_vel"],           # 15
                self.actions,                                        # 15
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # Reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.robot.set_dofs_position(position=self.dof_pos[envs_idx], dofs_idx_local=self.motor_dofs, zero_velocity=True, envs_idx=envs_idx)

        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat

        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx)

        # Reset other buffers
        self.last_actions[envs_idx] = 0.
        self.last_dof_vel[envs_idx] = 0.
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        #Randomization
        self.randomize_friction()
        self.randomize_pd_gains()
        # self.randomize_armature()

        # Fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None

    def _reward_tracking_lin_vel_x(self):
        lin_vel_error_x = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        return torch.exp(-lin_vel_error_x / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_lin_vel_y(self):
        lin_vel_error_y = torch.square(self.commands[:, 1] - self.base_lin_vel[:, 1])
        return torch.exp(-lin_vel_error_y / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])
    
    def _reward_similar_to_default(self):
    # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_orientation(self):
        base_up_vector = gs.utils.geom.transform_by_quat(
            torch.tensor([0.0, 0.0, 1.0], device=self.device).expand_as(self.base_pos),
            self.base_quat
        )
        horizontal_projection_sq = torch.sum(torch.square(base_up_vector[:, :2]), dim=1) # Larger is better
        return horizontal_projection_sq
    
    # ------------ randomization ----------------
    def randomize_link_properties(self):
        # scale mass of links
        mass_scale = 0.9 + 0.2 * torch.rand((self.robot.n_links,), device=self.device)  # 0.9〜1.1
        self.robot.set_links_inertial_mass(mass_scale)

    def randomize_com_shift(self):
        # shift COM positions
        num_links = self.robot.n_links
        link_indices = list(range(num_links))
        com_shift = 0.01 * torch.randn((self.num_envs, num_links, 3), device=self.device)  # +-0.01m=+-10mm
        self.robot.set_COM_shift(com_shift, link_indices)

    def randomize_friction(self):
        # frictions between the ground and robots
        # friction = 0.5 + torch.rand(1).item() # 0.5~1.5
        friction = 0.1 + 1.6*torch.rand(1).item() # 0.1~1.7

        self.robot.set_friction(friction)
        self.ground.set_friction(friction)

    def randomize_pd_gains(self):
        # pd gains of the joint control
        num_dofs = self.robot.n_dofs
        kp_min, kp_max = 400.0, 600.0
        kv_min, kv_max = 20.0, 40.0
        kp = torch.rand(num_dofs, device=self.device) * (kp_max - kp_min) + kp_min
        kv = torch.rand(num_dofs, device=self.device) * (kv_max - kv_min) + kv_min
        self.robot.set_dofs_kp(kp)
        self.robot.set_dofs_kv(kv)

    def randomize_armature(self):
        # joint's rotor inertia
        armature_min, armature_max = 0.01, 0.15
        armature = torch.rand(self.robot.n_dofs, device=self.device) * (armature_max - armature_min) + armature_min
        self.robot.set_dofs_armature(armature)

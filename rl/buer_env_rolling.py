import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat

# 继承原始的BuerEnv并添加滚动相关功能
from .buer_env import BuerEnv, gs_rand_float

class BuerEnvRolling(BuerEnv):
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, device="cuda"):
        super().__init__(num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer, device)
        
        # 添加滚动相关的历史记录
        self.prev_base_pos = torch.zeros_like(self.base_pos)
        self.rolling_phase = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # 将配置中的 action_scale 转为与 DOF 对应的 tensor
        raw_scale = self.env_cfg.get("action_scale", 1.0)
        if not torch.is_tensor(raw_scale):
            # 确保 dtype 与 default_dof_pos 一致，并在正确的 device 上
            raw_scale = torch.tensor(raw_scale, dtype=self.default_dof_pos.dtype, device=self.device)
        self.action_scale = raw_scale
        
    def step(self, actions):
        # 保存之前的位置用于计算滚动
        self.prev_base_pos[:] = self.base_pos[:]
        
        # 截断原始动作
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        # 是否模拟时延
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        # 对各关节按元素缩放
        scaled_actions = exec_actions * self.action_scale
        # 计算目标关节位置
        target_dof_pos = scaled_actions + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
        self.scene.step()

        # 更新状态
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat))
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)

        # 计算滚动相关的度量
        self._compute_rolling_metrics()

        envs_idx = ((self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0).nonzero(as_tuple=False).flatten())
        self._resample_commands(envs_idx)

        # 重置逻辑
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        base_up_vector = gs.utils.geom.transform_by_quat(
            torch.tensor([0.0, 0.0, 1.0], device=self.device).expand_as(self.base_pos),
            self.base_quat
        )
        self.reset_buf |= (base_up_vector[:, 2].abs() > self.env_cfg["termination_z_threshold"])
        self.reset_buf |= (self.base_lin_vel[:, 0].abs() < 0.01) & (self.episode_length_buf > 200)

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0
        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # 计算奖励
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # 构建观测
        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"],
                self.projected_gravity,
                self.commands * self.commands_scale,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],
                self.dof_vel * self.obs_scales["dof_vel"],
                self.actions,
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    def _compute_rolling_metrics(self):
        euler_angles = quat_to_xyz(self.base_quat)
        self.rolling_phase = euler_angles[:, 2]

    def _reward_rolling_velocity(self):
        rolling_angular_vel = self.base_ang_vel[:, 2].abs()
        # forward_vel = self.base_lin_vel[:, 0]
        world_fwd_vel = ((self.base_pos - self.prev_base_pos) / self.dt)[:, 0].clamp(min=0)
        # return rolling_angular_vel * forward_vel.clamp(min=0)
        return rolling_angular_vel * world_fwd_vel

    def _reward_forward_velocity(self):
        world_fwd_vel = ((self.base_pos - self.prev_base_pos) / self.dt)[:, 0]
        # return self.base_lin_vel[:, 0].clamp(min=0)
        return world_fwd_vel.clamp(min=0)

    def _reward_lateral_velocity(self):
        return torch.square(self.base_lin_vel[:, 1])

    def _reward_vertical_velocity(self):
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_action_smoothness(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_joint_acceleration(self):
        return torch.sum(torch.square(self.last_dof_vel - self.dof_vel), dim=1)

    def _reward_energy(self):
        return torch.sum(torch.abs(self.actions * self.dof_vel), dim=1)

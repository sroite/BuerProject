# rl/buer_env_rolling.py

import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat

# 我们直接继承您已经验证过的 BuerEnv 类，这样可以复用所有基础功能
from .buer_env import BuerEnv 

class BuerEnvRolling(BuerEnv):
    def __init__(self, *args, **kwargs):
        # 首先，调用父类的初始化方法，完成所有基础设置
        super().__init__(*args, **kwargs)
        print("--- Rolling Environment Initialized ---")
        
        # 重新初始化奖励函数字典，以包含滚动专用的奖励
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

    # 重写 step 函数，以使用新的滚动专用逻辑
    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
        self.scene.step()

        # 更新状态缓冲区
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)
        
        envs_idx = ((self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0).nonzero(as_tuple=False).flatten())
        self._resample_commands(envs_idx)

        self.reset_buf = self.episode_length_buf > self.max_episode_length

        # --- 使用滚动专用的终止逻辑 ---
        base_up_vector = gs.utils.geom.transform_by_quat(
            torch.tensor([0.0, 0.0, 1.0], device=self.device).expand_as(self.base_pos),
            self.base_quat
        )
        self.reset_buf |= (base_up_vector[:, 2].abs() > self.env_cfg["termination_z_threshold"])

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0
        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())
        
        # --- 计算奖励 ---
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        self.obs_buf = torch.cat([
            self.base_ang_vel * self.obs_scales["ang_vel"],
            self.projected_gravity,
            self.commands * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],
            self.dof_vel * self.obs_scales["dof_vel"],
            self.actions,
        ], dim=-1)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras
    
    def _reward_roll_forward(self):
        # 奖励 Z轴角速度 和 X轴线速度 的乘积
        return self.base_ang_vel[:, 2] * self.base_lin_vel[:, 0]

    def _reward_joint_velocities(self):
        # 奖励所有关节的速度，鼓励腿部持续运动
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_lin_vel_yz(self):
        # 惩罚 Y轴和Z轴的线速度
        return torch.sum(torch.square(self.base_lin_vel[:, 1:]), dim=1)

    def _reward_ang_vel_xz(self):
        # 惩罚 X轴和Z轴的角速度
        return torch.square(self.base_ang_vel[:, 0]) + torch.square(self.base_ang_vel[:, 2])

    def _reward_torques(self):
        # 惩罚关节力矩
        try:
            return torch.sum(torch.square(self.robot.get_dofs_torque(self.motor_dofs)), dim=1)
        except AttributeError:
            return torch.sum(torch.square(self.robot.get_dofs_force(self.motor_dofs)), dim=1)
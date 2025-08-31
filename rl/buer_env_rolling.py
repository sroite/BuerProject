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
        self.last_base_ang_vel = torch.zeros_like(self.base_ang_vel)
        # 新增：记录最远前进距离 (x轴负方向)
        self.max_forward_progress = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        # 将配置中的 action_scale 转为与 DOF 对应的 tensor
        raw_scale = self.env_cfg.get("action_scale", 1.0)
        if not torch.is_tensor(raw_scale):
            # 确保 dtype 与 default_dof_pos 一致，并在正确的 device 上
            raw_scale = torch.tensor(raw_scale, dtype=self.default_dof_pos.dtype, device=self.device)
        self.action_scale = raw_scale
        
        # 定义奖励函数的归一化参数（阈值）
        self.reward_thresholds = {
            "lateral_velocity": 0.1,      # 可接受的横向速度阈值
            "vertical_velocity": 0.1,      # 可接受的垂直速度阈值
            "action_smoothness": 0.1,     # 动作变化阈值
            "joint_acceleration": 1.0,    # 关节加速度阈值
            "angular_acceleration": 0.5,  # 角加速度阈值
            "reversal_threshold": 0.1,    # 后退距离阈值
        }
        
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
        self.last_base_ang_vel[:] = self.base_ang_vel[:]
        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    def _compute_rolling_metrics(self):
        """计算滚动相关的度量"""
        euler_angles = quat_to_xyz(self.base_quat)
        self.rolling_phase = euler_angles[:, 2]

    # ==================== 主要任务奖励 ====================
    
    def _reward_forward_velocity_tracking(self):
        """
        奖励跟踪命令的前进速度
        使用高斯型奖励，鼓励接近命令速度
        返回值范围: [0, 1]
        """
        target_vel = self.commands[:, 0]
        actual_vel = self.base_lin_vel[:, 0]  # 实际速度
        
        # 计算速度误差
        vel_error = torch.abs(target_vel - actual_vel)
        
        # 高斯型奖励，使用tracking_sigma作为标准差
        sigma = self.reward_cfg.get("tracking_sigma", 0.25)
        reward = torch.exp(-vel_error**2 / sigma)
        return reward
    
    def _reward_rolling_velocity(self):
        """
        奖励正向滚动角速度与前进速度的耦合
        鼓励通过持续正向滚动来前进
        返回值范围: [0, 1]
        """
        rolling_angular_vel = self.base_ang_vel[:, 2] # 获取滚动角速度（绕z轴），保留方向信息
        desired_angular_vel = self.base_lin_vel[:, 0] / self.base_pos[:, 2] # 根据机器人高度更新期望滚动角速度
        ang_vel_error = desired_angular_vel + rolling_angular_vel # 移动方向为正时对应角速度为负

        # 计算奖励
        reward = torch.exp(-torch.square(ang_vel_error) / self.reward_cfg.get("rolling_sigma", 0.5))
        return reward
    
    def _reward_reverse_rolling(self):
        """
        惩罚反向滚动
        z轴滚动速度和x轴线速度符号相反
        返回值范围: {-1, 0}
        """
        linear_vel_x = self.base_lin_vel[:, 0]
        rolling_angular_vel = self.base_ang_vel[:, 2]
        
        # 直接使用布尔逻辑
        forward_and_negative_roll = (linear_vel_x > 0) & (rolling_angular_vel < 0)
        backward_and_positive_roll = (linear_vel_x < 0) & (rolling_angular_vel > 0)
        correct_rolling = forward_and_negative_roll | backward_and_positive_roll
        
        reward = torch.where(correct_rolling, 0.0, -1.0)
        return reward

    def _reward_consistent_rolling_direction(self):
        """
        奖励保持一致的滚动方向，惩罚频繁改变滚动方向
        返回值范围: [-1, 1]
        """
        # 计算角速度的变化（特别是方向变化）
        angular_vel_change = self.base_ang_vel[:, 2] - self.last_base_ang_vel[:, 2]

        # 检测方向改变：如果当前和上一时刻的角速度符号不同
        direction_changed = (self.base_ang_vel[:, 2] * self.last_base_ang_vel[:, 2]) < 0

        # 对方向改变给予强惩罚
        direction_penalty = torch.where(
            direction_changed,
            -1.0 * torch.ones_like(angular_vel_change),  # 方向改变时给予固定惩罚
            torch.zeros_like(angular_vel_change)
        )
        
        # 对平滑的速度变化给予小奖励（鼓励稳定滚动）
        smoothness_reward = torch.exp(-torch.abs(angular_vel_change) / 0.5)
        
        return 0.3 * smoothness_reward + 0.7 * direction_penalty
    
    # ==================== 运动质量奖励 ====================
    
    def _reward_lateral_velocity(self):
        """
        惩罚横向速度
        返回值范围: [-1, 0]
        """
        threshold = self.reward_thresholds["lateral_velocity"]
        lateral_vel = torch.abs(self.base_lin_vel[:, 2])
        # 使用平滑的惩罚函数
        penalty = -torch.tanh(lateral_vel / threshold)
        return penalty

    def _reward_vertical_velocity(self):
        """
        惩罚垂直速度
        返回值范围: [-1, 0]
        """
        threshold = self.reward_thresholds["vertical_velocity"]
        vertical_vel = torch.abs(self.base_lin_vel[:, 1])
        # 使用平滑的惩罚函数
        penalty = -torch.tanh(vertical_vel / threshold)
        return penalty
    
    def _reward_body_stability(self):
        """
        奖励身体稳定性（减少roll和pitch的变化）
        返回值范围: [0, 1]
        """
        euler_angles = quat_to_xyz(self.base_quat)
        # 惩罚roll（滚动）和pitch（俯仰）
        roll_penalty = torch.exp(-5.0 * euler_angles[:, 0] ** 2)
        pitch_penalty = torch.exp(-5.0 * euler_angles[:, 1] ** 2)
        return pitch_penalty * roll_penalty

    # ==================== 平滑性奖励 ====================
    
    def _reward_action_smoothness(self):
        """
        奖励动作的平滑性
        返回值范围: [0, 1]
        """
        threshold = self.reward_thresholds["action_smoothness"]
        action_diff = torch.sum(torch.abs(self.last_actions - self.actions), dim=1)
        # 指数衰减奖励
        reward = torch.exp(-action_diff / threshold)
        return reward

    def _reward_joint_acceleration(self):
        """
        惩罚关节加速度
        返回值范围: [-1, 0]
        """
        threshold = self.reward_thresholds["joint_acceleration"]
        joint_acc = torch.sum(torch.abs(self.last_dof_vel - self.dof_vel), dim=1)
        # 归一化惩罚
        penalty = -torch.tanh(joint_acc / threshold)
        return penalty
    
    def _reward_angular_acceleration(self):
        """
        惩罚角加速度（提高滚动平滑性）
        返回值范围: [-1, 0]
        """
        threshold = self.reward_thresholds["angular_acceleration"]
        ang_acc = torch.sum(torch.abs(self.last_base_ang_vel - self.base_ang_vel), dim=1)
        # 归一化惩罚
        penalty = -torch.tanh(ang_acc / threshold)
        return penalty

    
    # ==================== 进度奖励 ====================
    
    def _reward_forward_progress(self):
        """
        奖励持续前进，惩罚后退
        返回值范围: [-1, 1]
        """
        # 计算这一步的前进距离
        step_progress = -(self.base_pos[:, 0] - self.prev_base_pos[:, 0])
        
        # 奖励前进，惩罚后退（使用tanh使奖励有界）
        reward = torch.tanh(step_progress * 10)  # 乘以10来调整敏感度
        return reward
    
    def _reward_survival(self):
        """
        生存奖励
        返回值范围: 1 (常数)
        """
        return torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_float)
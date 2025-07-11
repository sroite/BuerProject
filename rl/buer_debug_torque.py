# rl/buer_debug_torque.py

import argparse
import torch
import time
from .buer_env import BuerEnv
from .buer_train import get_buer_cfgs
import genesis as gs

def main():
    gs.init(logging_level="warning")
    print("--- Starting Direct Torque Injection Test ---")

    # 1. 加载配置
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_buer_cfgs()
    env_cfg["substeps"] = 2

    # 2. 创建一个最简单的环境
    env = BuerEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )
    
    # 3. 定义我们要测试的关节和施加的力矩
    # 我们选择第一条腿的第一个关节 (leg1_lap_joint)
    target_joint_name = 'leg1_lap_joint'
    target_joint_idx = env.motor_dofs[0] # 获取它的索引
    torque_to_apply = 50.0  # 一个非常大的力矩

    print(f"\n--- Simulation Ready ---")
    print(f"Applying a constant torque of {torque_to_apply} Nm to joint '{target_joint_name}'.")
    print("OBSERVE: Does the joint move AT ALL? Or is it completely limp?")
    print("Press Ctrl+C to exit.")

    # 4. 创建一个力矩向量，只在目标关节上有值
    torque_vector = torch.zeros((1, env.num_actions), device=env.device)
    torque_vector[0, 0] = torque_to_apply

    # 5. 仿真主循环
    while True:
        # 不再使用PD控制器，而是直接设置关节力矩
        env.robot.set_dofs_torque(torque_vector, dofs_idx_local=env.motor_dofs)
        env.scene.step()
        time.sleep(0.01)

if __name__ == "__main__":
    main()
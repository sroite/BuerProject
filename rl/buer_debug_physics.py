# rl/buer_debug_physics.py

import argparse
import torch
from .buer_env import BuerEnv
from .buer_train import get_buer_cfgs  # 我们直接从训练脚本导入配置，方便同步
import time
import genesis as gs

def main():
    parser = argparse.ArgumentParser(description="Physics Debugging Script for Buer Robot")
    parser.add_argument("--kp", type=float, default=None, help="Override Kp gain for this run.")
    parser.add_argument("--kd", type=float, default=None, help="Override Kd gain for this run.")
    args = parser.parse_args()

    gs.init(logging_level="warning")

    print("--- Starting Physics Debugging Session ---")

    # 1. 加载我们最新的训练配置
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_buer_cfgs()

    # 2. 如果命令行指定了新的PD增益，就覆盖掉配置文件中的值
    if args.kp is not None:
        env_cfg["kp"] = args.kp
        print(f"Overriding Kp with: {args.kp}")
    if args.kd is not None:
        env_cfg["kd"] = args.kd
        print(f"Overriding Kd with: {args.kd}")

    # 为 env_cfg 添加缺失的 substeps 参数
    env_cfg["substeps"] = 2 # 使用和训练时相同的默认值

    # === 修正: 传入完整的配置, 即使我们不使用所有部分 ===
    # 3. 创建一个最简单的环境，强制显示GUI，且只有一个机器人
    env = BuerEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,          # Pass the full obs_cfg
        reward_cfg=reward_cfg,    # Pass the full reward_cfg
        command_cfg=command_cfg,  # Pass the full command_cfg
        show_viewer=True,
    )

    print("\n--- Simulation Ready ---")
    print(f"Testing with Kp={env.env_cfg['kp']}, Kd={env.env_cfg['kd']}")
    print("Robot will now attempt to hold its default standing pose.")
    print("Observe its behavior: Does it collapse? Oscillate? Stand firm?")
    print("Press Ctrl+C to exit.")

    # 4. 仿真主循环
    while True:
        # 我们不使用AI的动作，而是直接命令机器人回到它的默认关节角度
        # 这相当于一个最强的“站立”指令
        actions = torch.zeros((1, env.num_actions), device=env.device)
        env.step(actions)
        
        # 为了避免仿真速度过快，可以加一个小的延时
        time.sleep(0.01)

if __name__ == "__main__":
    main()
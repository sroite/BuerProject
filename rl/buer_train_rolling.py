# buer_train.py

import argparse
import os
import pickle
import shutil

# === 修改: 导入 BuerEnv ===
from .buer_env_rolling import BuerEnvRolling
from rsl_rl.runners import OnPolicyRunner

import genesis as gs


def get_train_cfg(exp_name, max_iterations):
    # 训练参数 (ppo, policy) 可以暂时沿用 go2 的设置
    train_cfg_dict = {
        "algorithm": {
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
        },
        "runner": {
            "algorithm_class_name": "PPO",
            "experiment_name": exp_name,
            "max_iterations": max_iterations,
            "run_name": "",
            "save_interval": 100,
            "num_steps_per_env": 24,
            # --- 修正: 补全缺失的关键参数 ---
            "policy_class_name": "ActorCritic",
            "runner_class_name": "OnPolicyRunner",
            "checkpoint": -1,
            "load_run": -1,
            "log_interval": 1,
            "resume": False,
        },
        "seed": 1,
    }
    return train_cfg_dict


# rl/buer_train_rolling.py


# === 全新的、为“滚动”量身定制的配置函数 ===
def get_buer_cfgs_rolling():
    # 1. 物理环境配置 (大部分保持不变)
    env_cfg = {
        "num_actions": 15,
        "default_joint_angles": {
            "leg1_lap_joint": 0.0,
            "leg1_calf_joint": 0.0,
            "leg1_foot_joint": 0.0,
            "leg2_lap_joint": 0.0,
            "leg2_calf_joint": 0.0,
            "leg2_foot_joint": 0.0,
            "leg3_lap_joint": 0.0,
            "leg3_calf_joint": 0.0,
            "leg3_foot_joint": 0.0,
            "leg4_lap_joint": 0.0,
            "leg4_calf_joint": 0.0,
            "leg4_foot_joint": 0.0,
            "leg5_lap_joint": 0.0,
            "leg5_calf_joint": 0.0,
            "leg5_foot_joint": 0.0,
        },
        "dof_names": [
            "leg1_lap_joint",
            "leg1_calf_joint",
            "leg1_foot_joint",
            "leg2_lap_joint",
            "leg2_calf_joint",
            "leg2_foot_joint",
            "leg3_lap_joint",
            "leg3_calf_joint",
            "leg3_foot_joint",
            "leg4_lap_joint",
            "leg4_calf_joint",
            "leg4_foot_joint",
            "leg5_lap_joint",
            "leg5_calf_joint",
            "leg5_foot_joint",
        ],
        "kp": 200.0,
        "kd": 50.0,
        "base_init_pos": [0.0, 0.0, 2.5],
        "base_init_quat": [0.707, 0.707, 0.0, 0.0],  # 保持初始站立姿态
        "episode_length_s": 20.0,
        "resampling_time_s": 10.0,  # 给AI更长的时间执行稳定指令
        "action_scale": 0.35,  # 鼓励更大的动作范围以实现滚动
        "simulate_action_latency": True,
        "clip_actions": 100.0,
        "termination_z_threshold": 0.5,  # 如果基座Z轴与世界垂直方向的点积小于0.5 (倾斜>60度)，则终止
    }

    # 2. 观测空间配置 (保持不变)
    obs_cfg = {
        "num_obs": 54,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }

    # 3. 滚动的核心：全新的奖励函数
    reward_cfg = {
        "reward_scales": {
            # === 核心奖励 ===
            # 只有当机器人一边前进，一边绕着身体侧向轴(Z轴)旋转时，才能获得高分
            "roll_forward": 8.0,
            # === 辅助奖励 ===
            # 鼓励腿部持续、大幅度地运动
            "joint_velocities": 0.005,
            # === 惩罚项/约束项 ===
            # 惩罚不希望的移动：侧向(Y)和上下(Z)的窜动
            "lin_vel_yz": -2.0,
            # 惩罚不希望的旋转：身体向前翻滚(X)或原地打转(Y)
            "ang_vel_xz": -0.5,
            # 鼓励节能
            "torques": -0.00002,
        },
    }

    # 4. 指令配置 (简化目标)
    command_cfg = {
        "num_commands": 3,
        # 在这个实验中，我们只给它一个恒定的、向前的速度指令，让它专心学习滚动
        "lin_vel_x_range": [-1.0, 1.0],
        "lin_vel_y_range": [-0.0, 0.0],
        "ang_vel_range": [-0.0, 0.0],
    }
    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="buer_rolling")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--num_envs", type=int, default=40000)
    parser.add_argument("--param_name", type=str, default="test")
    parser.add_argument("--substeps", type=int, default=2)
    parser.add_argument("--max_iterations", type=int, default=1000)
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Enable GUI for visualization during training",
    )
    args = parser.parse_args()

    gs.init(logging_level="warning")

    if args.gui:
        print(
            "GUI mode is enabled. Reducing the number of environments for visualization."
        )
        args.num_envs = 5
        args.max_iterations = 100

    log_dir = f"{args.log_dir}/{args.exp_name}/{args.param_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_buer_cfgs_rolling()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    env_cfg["substeps"] = args.substeps

    # 为了避免删除已有训练成果，可以注释掉这两行
    # if os.path.exists(log_dir):
    #     shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    env = BuerEnvRolling(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=args.gui,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    runner.learn(
        num_learning_iterations=args.max_iterations, init_at_random_ep_len=True
    )


if __name__ == "__main__":
    main()

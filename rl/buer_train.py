# buer_train.py

import argparse
import os
import pickle
import shutil

# === 修改: 导入 BuerEnv ===
from .buer_env import BuerEnv
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

def get_train_cfg(exp_name, max_iterations):
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

def get_buer_cfgs():
    env_cfg = {
        "num_actions": 15,
        "default_joint_angles": {
            'leg1_lap_joint': 0.0, 'leg1_calf_joint': 0.0, 'leg1_foot_joint': 0.0,
            'leg2_lap_joint': 0.0, 'leg2_calf_joint': 0.0, 'leg2_foot_joint': 0.0,
            'leg3_lap_joint': 0.0, 'leg3_calf_joint': 0.3, 'leg3_foot_joint': 0.3,
            'leg4_lap_joint': 0.0, 'leg4_calf_joint': -0.3, 'leg4_foot_joint': -0.3,
            'leg5_lap_joint': 0.0, 'leg5_calf_joint': 0.0, 'leg5_foot_joint': 0.0,
        },
        "dof_names": [
            'leg1_lap_joint', 'leg1_calf_joint', 'leg1_foot_joint',
            'leg2_lap_joint', 'leg2_calf_joint', 'leg2_foot_joint',
            'leg3_lap_joint', 'leg3_calf_joint', 'leg3_foot_joint',
            'leg4_lap_joint', 'leg4_calf_joint', 'leg4_foot_joint',
            'leg5_lap_joint', 'leg5_calf_joint', 'leg5_foot_joint',
        ],
        "kp": 300.0, 
        "kd": 80.0, 
        "termination_z_threshold": 0.5, 
        "base_init_pos": [0.0, 0.0, 1.5],
        "base_init_quat": [0.707, 0.707, 0.0, 0.0], # (w,x,y,z)
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "num_obs": 54, # 3(ang_vel)+3(gravity)+3(cmd)+15(dof_pos)+15(dof_vel)+15(action)
        "obs_scales": {"lin_vel": 2.0, "ang_vel": 0.25, "dof_pos": 1.0, "dof_vel": 0.05},
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 1.0,
        "reward_scales": {
            "tracking_lin_vel_x": 3.0,
            "tracking_lin_vel_y": 2.0,
            "tracking_ang_vel": 1.0,

            "orientation": 0.5,
            "lin_vel_z": -0.1,
            "base_height": -0.1,
            
            "action_rate": 1.0,
            "similar_to_default": -0.05,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.0, 0.0],
        "lin_vel_y_range": [-2.0, 2.0],
        "ang_vel_range": [-1.0, 1.0],
    }
    return env_cfg, obs_cfg, reward_cfg, command_cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="buer_walking")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--num_envs", type=int, default=40000)
    parser.add_argument("--param_name", type=str, default="test")
    parser.add_argument("--substeps", type=int, default=2)
    parser.add_argument("--max_iterations", type=int, default=100)
    parser.add_argument("--gui", action="store_true", help="Enable GUI for visualization during training")
    args = parser.parse_args()

    gs.init(logging_level="warning")

    if args.gui:
        print("GUI mode is enabled. Reducing the number of environments for visualization.")
        args.num_envs = 5
        args.max_iterations = 100 

    log_dir = f"{args.log_dir}/{args.exp_name}/{args.param_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_buer_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    env_cfg["substeps"] = args.substeps

    os.makedirs(log_dir, exist_ok=True)

    env = BuerEnv(
        num_envs=args.num_envs, 
        env_cfg=env_cfg, 
        obs_cfg=obs_cfg, 
        reward_cfg=reward_cfg, 
        command_cfg=command_cfg,
        show_viewer=args.gui
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    
    pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], open(f"{log_dir}/cfgs.pkl", "wb"))

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)

if __name__ == "__main__":
    main()
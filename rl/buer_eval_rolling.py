# buer_eval.py

import argparse
import os
import pickle
import torch

# === 修改: 导入 BuerEnv ===
from .buer_env_rolling import BuerEnvRolling
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="buer_rolling")
    parser.add_argument("-l", "--log_dir", type=str, default="logs")
    parser.add_argument("-p", "--param_name", type=str, default="test")
    parser.add_argument("--ckpt", type=int, default=100) # 要加载的模型序号
    args = parser.parse_args()

    gs.init()

    log_dir = f"{args.log_dir}/{args.exp_name}/{args.param_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))
    # 在评估时不计算奖励
    reward_cfg["reward_scales"] = {}

    # === 修改: 实例化 BuerEnv ===
    env = BuerEnvRolling(
        num_envs=1, # 评估时只使用一个环境
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True, # 显示模拟器窗口
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device="cuda:0")

    obs, _ = env.reset()
    with torch.no_grad():
        while True:
            actions = policy(obs)
            obs, _, rews, dones, infos = env.step(actions)

if __name__ == "__main__":
    main()
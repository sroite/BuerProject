#!/usr/bin/env python3

import argparse
import os
import sys
import pickle
import json

import torch
import genesis as gs

# add Genesis locomotion example to path
sys.path.append(os.environ["HOME"] + "/genesis_ws/Genesis/examples/locomotion")
sys.path.append(os.environ["HOME"] + "/ros/agent_system_ws/src/buer")
from rl.buer_env import BuerEnv
from rsl_rl.runners import OnPolicyRunner

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="buer_walking")
    parser.add_argument("-l", "--log_dir", type=str, required=True, help='Path to logs/xxx directory (contains cfgs.pkl)')
    parser.add_argument("--ckpt", type=int, default=100)
    parser.add_argument("-d", "--device", type=str, required=True, help='Calculation decice (cpu or cuda)')
    args = parser.parse_args()

    log_dir = args.log_dir
    cfgs_path = os.path.join(args.log_dir, "cfgs.pkl")
    if args.device == "cpu":
        tensor_device = gs.cpu
    elif args.device == "gpu":
        tensor_device = gs.gpu
    else:
        print(f"Not supported device: {args.device}")
        return

    print(f"Loading: {log_dir}")
    print(f"Loading: {cfgs_path}")
    print(f"Device:  {tensor_device}")

    gs.init(backend=tensor_device)

    with open(cfgs_path, "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    reward_cfg["reward_scales"] = {}

    env = BuerEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
        device=args.device,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=args.device)
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)

    obs, _ = env.reset()

    model = runner.alg.actor_critic.actor
    model.eval()

    # generate and set a dummy input
    example_obs = torch.randn(1, obs.shape[1], device=args.device)
    traced_model = torch.jit.trace(model, example_obs)

    policy_path = os.path.join(log_dir, f"policy_traced.pt")
    traced_model.save(policy_path)

if __name__ == "__main__":
    main()

"""
# evaluation
python examples/locomotion/export_pretrained_network.py -e go2-walking --ckpt 100
"""

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import cartpoleIsaacLab.tasks  # noqa: F401


def main():
    """
    使用零动作策略运行 Isaac Lab 环境的主函数
    
    该函数创建并运行一个强化学习环境，在每个时间步执行零动作（所有动作值为 0），
    用于测试环境的基本功能或作为基线对比。
    
    函数执行以下步骤：
    1. 解析环境配置（任务类型、设备、环境数量、是否使用 Fabric）
    2. 创建 Gym 环境实例
    3. 打印观察空间和动作空间信息
    4. 重置环境
    5. 在仿真运行时循环执行：计算零动作、应用动作、获取观测和奖励
    6. 关闭仿真器
    
    Args:
        无直接参数，使用全局变量 args_cli 获取命令行参数：
            - task (str): 要运行的任务名称
            - device (str): 计算设备（如 "cuda:0" 或 "cpu"）
            - num_envs (int): 并行环境的数量
            - disable_fabric (bool): 是否禁用 Fabric 优化
    
    Returns:
        None
    
    Raises:
        无显式抛出异常，依赖 Isaac Lab 框架处理环境创建和仿真过程中的错误
    """
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)   # 返回ManagerBasedRLEnv实例 
    
    # # 等效于以下代码：
    # env_cfg = CartpoleEnvCfg()    #  这个被隐藏在工厂里进行实例化了  
    # # 在这个项目里叫cartpoleisaaclab_env_cfg：CartpoleisaaclabEnvCfg，可以到cartpoleisaaclab文件夹下__init__.py中查看注册的环境
    # env_cfg.scene.num_envs = args_cli.num_envs
    # env_cfg.sim.device = args_cli.device
    # # setup RL environment
    # env = ManagerBasedRLEnv(cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # compute zero actions
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            # apply actions
            observations, rewards, terminated, truncated, info = env.step(actions)
            print(f"[INFO]: Observations: {observations}")
            print(f"[INFO]: Rewards: {rewards}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

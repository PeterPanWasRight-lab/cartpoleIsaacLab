# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for a simple Cartpole robot."""


import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.actuators import IdealPDActuatorCfg       # PD控制，可以尝试一下
from isaaclab.assets import ArticulationCfg  
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

"""
IdealPDActuatorCfg 显式PD：
1. 显式计算力：在时间步 t开始时，用上一时刻的状态 (θ_t-1, ω_t-1)显式算出控制力 τ。
2. 施加力积分：将 τ作为已知输入，交给动力学方程积分，得到 t时刻的新状态。
ImplicitActuatorCfg 隐式PD：（数值分析里，对线性系统无条件稳定的积分方法，适合刚体系统）
1. 联立求解：将PD控制律（定义了力与状态的关系）直接与系统的动力学方程耦合，形成一个需要联合求解的代数方程组。
2. 隐式积分：求解器直接解出满足这个耦合关系的、当前时刻​ t的状态。控制力是作为解的一部分被“隐式”地确定。
"""

##
# Configuration
##

CARTPOLE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Classic/Cartpole/cartpole.usd",   # 远程USD文件路径，ISAACLAB_NUCLEUS_DIR是Isaac Lab中一个预定义的环境变量，指向包含各种资产（如机器人模型、场景元素等）的Nucleus服务器目录。这个路径指定了要加载的Cartpole机器人的USD文件位置。
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,                   # 刚体属性配置，启用刚体模拟，如果禁用则物体将不会参与物理模拟，通常用于装饰性元素或不需要物理交互的对象。
            max_linear_velocity=1000.0,                # 这些限制应用于倒立摆的每个刚体部件（小车、摆杆），以确保它们在模拟中不会达到非物理的速度，这有助于保持模拟的稳定性和真实性。
            max_angular_velocity=1000.0,               # rad/s  刚体部件的角速度限制
            max_depenetration_velocity=100.0,          # 碰撞点	m/s	解决物体穿透时的最大分离速度
            enable_gyroscopic_forces=True,             # 是否启用陀螺效应（旋转体的角动量守恒）
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,             # 是否启用自碰撞检测
            solver_position_iteration_count=4,         # 位置求解器的迭代次数，一个约束的解决可能破坏其他约束, 解决物体穿透和位置约束问题,值越高，位置精度越高，但计算开销越大
            solver_velocity_iteration_count=0,         # 速度求解器的迭代次数, 解决摩擦力和速度相关约束,0表示禁用速度迭代，仅使用位置迭代
            sleep_threshold=0.005,                     # 睡眠阈值 当物体动能 < 0.005J 时，停止物理模拟，减少静止物体的计算开销
            stabilization_threshold=0.001,             # 稳定性阈值 当物体穿透深度 < 1mm 时，使用特殊算法稳定物体，防止穿透和震荡
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 2.0), joint_pos={"slider_to_cart": 0.0, "cart_to_pole": 0.0}  # pos表示机器人基座(base)在世界坐标系中的初始位置
    ),
    #  IdealPDActuatorCfg/ImplicitActuatorCfg :\tau_{j, computed} = k_p * (q_{des} - q) + k_d * (\dot{q}_{des} - \dot{q}) + \tau_{ff}
    actuators={
        "cart_actuator": ImplicitActuatorCfg(      # 配置一个关节控制器  采用了PD控制，具体可参考Isaac Lab文档中的ImplicitActuatorCfg类。这个控制器将作用于名为 "slider_to_cart" 的关节，这个关节连接了小车和地面，允许小车沿水平轴移动。
            joint_names_expr=["slider_to_cart"],
            effort_limit_sim=400.0,                #  effort_limit_sim 表示关节的力矩限制，这个限制用于限制小车在运动过程中所允许的力矩。
            stiffness=0.0,                         #  stiffness 表示关节的刚度，这个刚度用于控制小车在运动过程中小车和地面之间的刚度。
            damping=10.0,
        ),
        "pole_actuator": ImplicitActuatorCfg(
            joint_names_expr=["cart_to_pole"], effort_limit_sim=400.0, stiffness=0.0, damping=0.0
        ),
    },
)
"""Configuration for a simple Cartpole robot."""
# 还有其他类型的Actuator 比如DCMotorActuator VariableGearRatioActuator 
# 注意用字符串标的名字要和USD中的一致，比如这里的 "slider_to_cart" 和 "cart_to_pole" 就是USD文件中定义的关节名字。
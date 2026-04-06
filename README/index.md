# 📚 CartpoleIsaacLab 学习文档中心

> 本目录是针对 IsaacLab + skrl 训练 CartPole 智能体的系统性学习文档，从架构到代码逐层深入，图文并茂。

---

## 📂 文档目录

| 序号 | 文件 | 内容简介 |
|------|------|----------|
| 1 | [01_IsaacLab架构总览.md](./01_IsaacLab架构总览.md) | IsaacLab 整体设计哲学、工厂模式、gym注册流程 |
| 2 | [02_skrl程序架构.md](./02_skrl程序架构.md) | skrl 框架核心组件关系、Runner/Agent/Trainer 设计 |
| 3 | [03_Train数据流全景.md](./03_Train数据流全景.md) | 从 train.py 到 PPO 更新的完整数据流动路径 |
| 4 | [04_PPO算法详解.md](./04_PPO算法详解.md) | PPO 核心循环、GAE计算、策略更新详细图解 |
| 5 | [05_环境配置系统.md](./05_环境配置系统.md) | MDP 配置类体系、Reward/Obs/Action 管理器设计 |
| 6 | [06_代码速查手册.md](./06_代码速查手册.md) | 关键函数签名、常用 API、配置字段速查 |

---

## 🗺️ 快速导航：你想了解什么？

### 我想了解"整体是怎么运转的"
→ 先看 **[01_IsaacLab架构总览](./01_IsaacLab架构总览.md)**，再看 **[02_skrl程序架构](./02_skrl程序架构.md)**

### 我想搞清楚"训练时数据怎么流动的"
→ 直接看 **[03_Train数据流全景](./03_Train数据流全景.md)**（本文最推荐，图最多）

### 我想深入理解 PPO 算法代码
→ 看 **[04_PPO算法详解](./04_PPO算法详解.md)**，结合 `重要的文件/ppo.py` 对照阅读

### 我想修改环境（奖励/观测/终止条件）
→ 看 **[05_环境配置系统](./05_环境配置系统.md)**，直接定位到 `cartpoleisaaclab_env_cfg.py`

### 我忘记某个函数/配置怎么写了
→ 看 **[06_代码速查手册](./06_代码速查手册.md)**

---

## 🏗️ 项目文件结构速览

```
cartpoleIsaacLab/
├── scripts/
│   └── skrl/
│       ├── train.py          ← 训练入口（从这里开始读！）
│       └── play.py           ← 推理/可视化入口
├── source/
│   └── cartpoleIsaacLab/
│       └── cartpoleIsaacLab/
│           └── tasks/
│               └── manager_based/
│                   └── cartpoleisaaclab/
│                       ├── cartpoleisaaclab_env_cfg.py  ← 环境配置核心
│                       ├── agents/
│                       │   └── skrl_ppo_cfg.yaml         ← PPO超参数
│                       └── mdp/
│                           └── rewards.py                ← 自定义奖励函数
└── 重要的文件/
    ├── ppo.py       ← skrl PPO 算法实现（核心！）
    ├── runner.py    ← skrl Runner（组装所有组件）
    ├── base.py      ← Trainer 基类（训练主循环）
    ├── gaussian.py  ← 高斯策略模型 Mixin
    ├── shared.py    ← 共享网络模型生成器
    └── common.py    ← 网络容器生成工具
```

---

## ⚡ 极速上手：运行训练

```bash
# 训练
python scripts/skrl/train.py --task Template-Cartpoleisaaclab-v0 --num_envs 64

# 推理
python scripts/skrl/play.py --task Template-Cartpoleisaaclab-v0 --num_envs 4
```

---

*文档生成时间：2026-04-06*

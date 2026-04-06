# 03 · Train 数据流全景

> **目标**：以数据为主线，追踪从 `train.py` 启动到 PPO 网络更新的每一步数据变换。这是最重要的一篇，建议反复阅读。

---

## 1. 全景数据流（鸟瞰图）

```
┌─────────────────────────────────────────────────────────────────────┐
│ train.py                                                            │
│                                                                     │
│  gym.make() → SkrlVecEnvWrapper → Runner(env, cfg) → runner.run()  │
└──────────────────────────────┬──────────────────────────────────────┘
                                │
                    ┌───────────▼──────────────┐
                    │  SequentialTrainer.train()│
                    │  single_agent_train()     │
                    └───────────┬──────────────┘
                                │
        ┌───────────────────────▼───────────────────────┐
        │                主训练循环（4800 timesteps）     │
        │                                               │
        │  ┌─────────────────────────────────────────┐  │
        │  │ 每个 timestep：                          │  │
        │  │                                         │  │
        │  │  states[4096,4]                         │  │
        │  │      ↓ agent.act()                      │  │
        │  │  actions[4096,1], log_prob[4096,1]      │  │
        │  │      ↓ env.step()                       │  │
        │  │  next_states[4096,4]                    │  │
        │  │  rewards[4096,1]                        │  │
        │  │  terminated[4096,1], truncated[4096,1]  │  │
        │  │      ↓ agent.record_transition()        │  │
        │  │  → Memory 写入一行                      │  │
        │  │      ↓ agent.post_interaction()         │  │
        │  │  → 每 32 步触发 PPO _update()           │  │
        │  └─────────────────────────────────────────┘  │
        └───────────────────────────────────────────────┘
```

---

## 2. 详细数据流：每个 timestep 内部

### 阶段一：推理阶段（act）

```
states: Tensor[4096, 4]
  ├── cart_position    (x坐标，范围约-3到3)
  ├── cart_velocity    (速度)
  ├── pole_angle       (杆子角度，0=竖直)
  └── pole_velocity    (角速度)

         ↓  RunningStandardScaler 归一化
         
preprocessed_states: Tensor[4096, 4]
  (减均值除标准差，使输入分布稳定)

         ↓  policy.act({"states": preprocessed_states}, role="policy")

GaussianMixin.act() 内部：
  1. compute(inputs, role) 
       → Linear(4→32) → ELU → Linear(32→32) → ELU → Linear(32→1)
       → mean_actions: Tensor[4096, 1]    ← 动作均值
       → log_std: nn.Parameter[1]        ← 可训练标准差参数（初始为0）
  
  2. Normal(mean_actions, exp(log_std)) 构建高斯分布
  
  3. actions = distribution.rsample()    ← 重参数化采样
       → actions: Tensor[4096, 1]        ← 实际动作（带随机性）
  
  4. log_prob = distribution.log_prob(actions)  ← 对数概率
       → log_prob: Tensor[4096, 1]

         ↓
    
agent._current_log_prob = log_prob      ← 暂存，record_transition时用
return actions, log_prob, {"mean_actions": mean_actions}
```

### 阶段二：环境步进（env.step）

```
actions: Tensor[4096, 1]
  (每个环境一个力矩值，单位由 scale=100 放大)

         ↓  SkrlVecEnvWrapper.step()
         ↓  ManagerBasedRLEnv.step()
         
IsaacLab 内部执行：
  1. ActionManager.process(actions)
       → JointEffortActionCfg 把动作缩放后施加到 slider_to_cart 关节
       → 实际力 = actions × 100 牛顿
  
  2. PhysX 物理引擎步进（2次，因为 decimation=2）
  
  3. ObservationManager.compute()
       → joint_pos_rel: cart_pos, pole_angle
       → joint_vel_rel: cart_vel, pole_vel
       → concatenate → obs[4096, 4]
  
  4. RewardManager.compute()
       → alive:      tensor([1.0, 1.0, ..., 1.0]) × weight=1.0
       → terminating: tensor([0, 0, -1, 0, ...]) × weight=-2.0 (终止的env)
       → pole_pos:   (pole_angle)² × weight=-1.0
       → cart_vel:   |cart_velocity| × weight=-0.01
       → pole_vel:   |pole_velocity| × weight=-0.005
       → 求和 → rewards[4096, 1]
  
  5. TerminationManager.compute()
       → time_out: 超过300步的环境
       → cart_out_of_bounds: |cart_pos| > 3
       → terminated[4096, 1], truncated[4096, 1]

         ↓

next_states: Tensor[4096, 4]
rewards: Tensor[4096, 1]
terminated: Tensor[4096, 1]   ← 真正失败（越界）
truncated: Tensor[4096, 1]    ← 超时（不算失败）
```

### 阶段三：记录转换（record_transition）

```
record_transition 调用时传入:
  states, actions, rewards, next_states, terminated, truncated

内部执行：
  1. rewards_shaper（本项目 scale=0.1）
       → rewards = rewards × 0.1     ← 缩放奖励防止数值过大
  
  2. value.act(preprocessed_states)
       → DeterministicMixin.act()
       → compute() → Linear网络 → V(s): Tensor[4096, 1]
       → value_preprocessor 逆归一化得到实际值
  
  3. time_limit_bootstrap（本项目关闭）
       → 如果开启：rewards += γ × V(s) × truncated
       → （给超时episode补偿，让agent不害怕超时）
  
  4. memory.add_samples(
       states=states,
       actions=actions, 
       rewards=rewards,        ← 已×0.1
       next_states=next_states,
       terminated=terminated,
       truncated=truncated,
       log_prob=self._current_log_prob,   ← act时记录的
       values=values,          ← 刚计算的V(s)
     )

Memory 第 t 行（第 t 个 timestep）写入完成：
┌────────────┬──────────┬──────────┬──────────────┬──────────┐
│ states     │ actions  │ rewards  │ log_prob     │ values   │
│ [4096, 4]  │ [4096,1] │ [4096,1] │ [4096, 1]   │ [4096,1] │
└────────────┴──────────┴──────────┴──────────────┴──────────┘
```

---

## 3. PPO 更新触发条件

```python
# ppo.py post_interaction()
self._rollout += 1
if not self._rollout % self._rollouts and timestep >= self._learning_starts:
    # 每隔 rollouts=32 步，且 timestep >= learning_starts=0
    self.set_mode("train")
    self._update(timestep, timesteps)
    self.set_mode("eval")
```

**意味着：** 每 32 步（32 × 4096 = 131,072 个样本）触发一次完整的 PPO 更新。

---

## 4. PPO _update() 内部数据流

```
第一步：计算 last_values（用于GAE的边界值）

  last_states = self._current_next_states    ← 最后一步的 next_states
  last_values = value.act(last_states)       ← V(s_last)
  

第二步：从 Memory 读取完整轨迹

  rewards:    [32, 4096, 1]     ← 32步 × 4096环境
  terminated: [32, 4096, 1]
  truncated:  [32, 4096, 1]
  values:     [32, 4096, 1]     ← 已存的 V(s_t)

第三步：计算 GAE（广义优势估计）

  对每个时间步 t（从后往前）:
  
  δ_t = r_t + γ × V(s_{t+1}) - V(s_t)        ← TD误差
  A_t = δ_t + γλ × (1-done_t) × A_{t+1}      ← 递归优势
  
  参数: γ=0.99（折扣因子），λ=0.95（GAE系数）
  
  returns = advantages + values               ← R_t = A_t + V(s_t)
  advantages 归一化: (A - mean(A)) / std(A)  ← 稳定训练

第四步：写回 Memory
  
  memory.set_tensor("values", value_preprocessor(values))  ← 归一化
  memory.set_tensor("returns", value_preprocessor(returns))
  memory.set_tensor("advantages", advantages)

第五步：mini-batch 梯度下降（8 epochs × 8 mini-batches）

  for epoch in range(8):
    batches = memory.sample_all(mini_batches=8)
    
    for batch in batches:
      # batch 包含: states[16384,4], actions[16384,1], 
      #             log_prob[16384,1], advantages[16384,1]...
      
      # 计算新的 log_prob（用更新后的网络重新评估旧动作）
      _, new_log_prob, _ = policy.act({
          "states": batch_states,
          "taken_actions": batch_actions    ← 不采样，评估已有动作
      })
      
      # 概率比（PPO核心）
      ratio = exp(new_log_prob - old_log_prob)
      
      # Clipped Surrogate Objective
      surrogate = advantages × ratio
      surrogate_clipped = advantages × clip(ratio, 0.8, 1.2)
      policy_loss = -min(surrogate, surrogate_clipped).mean()
      
      # Value Loss (MSE)
      predicted_V = value.act(batch_states)
      value_loss = 2.0 × MSE(returns, predicted_V)
      
      # 梯度更新
      loss = policy_loss + value_loss
      optimizer.zero_grad()
      loss.backward()
      clip_grad_norm_(params, max_norm=1.0)   ← 梯度裁剪
      optimizer.step()
```

---

## 5. 数据 Shape 变换一览表

| 位置 | 变量 | Shape | 说明 |
|------|------|-------|------|
| 环境输出 | observations | [4096, 4] | 每个并行环境的观测 |
| 策略输入 | preprocessed_states | [4096, 4] | 归一化后 |
| 策略输出 | mean_actions | [4096, 1] | 动作均值 |
| 策略输出 | actions | [4096, 1] | 采样的实际动作 |
| 策略输出 | log_prob | [4096, 1] | 对数概率 |
| 价值输出 | values | [4096, 1] | V(s) 估计 |
| 环境输出 | rewards | [4096, 1] | 当步奖励 |
| Memory 一行 | - | 上述各项 | 一个 timestep 的数据 |
| Memory 完整 | states | [32, 4096, 4] | 32步轨迹 |
| GAE 计算 | advantages | [32, 4096, 1] | 优势函数 |
| GAE 计算 | returns | [32, 4096, 1] | 回报 |
| mini-batch | sampled_states | [16384, 4] | 32×4096/8 |
| mini-batch 后 | 梯度更新 | - | 更新共享网络参数 |

---

## 6. 完整时间线图

```
timestep:  0  1  2  ... 31 | 32 33 ... 63 | 64 ...
           ←── rollout 1 ──→ ←── rollout 2 ──→
           
每步:       act → step → record_transition
                                    ↑
                              写入 Memory 第 t 行
                              
timestep 31:                 post_interaction()
                               → _rollout=32，触发更新！
                               → 计算 GAE
                               → 8 epochs × 8 batches 梯度下降
                               → Memory 清空，开始下一轮收集
                               
timestep 32:                 新一轮 rollout 开始，使用更新后的网络
```

---

## 7. 关键代码对照

下面是 `base.py` 中训练循环和各组件的对应关系：

```python
# base.py - single_agent_train()

states, infos = self.env.reset()           # 初始化状态

for timestep in range(timesteps):
    
    # ① agent.act() → ppo.py line 265
    actions = self.agents.act(states, timestep, timesteps)[0]
    
    # ② env.step() → IsaacLab ManagerBasedRLEnv
    next_states, rewards, terminated, truncated, infos = self.env.step(actions)
    
    # ③ record_transition() → ppo.py line 290
    #    内部: value.act() + memory.add_samples()
    self.agents.record_transition(
        states, actions, rewards, next_states,
        terminated, truncated, infos, timestep, timesteps
    )
    
    # ④ post_interaction() → ppo.py line 376
    #    内部: 每32步调用 _update()
    self.agents.post_interaction(timestep, timesteps)
    
    # ⑤ 向量化环境直接用 next_states
    states = next_states
```

---

*← 上一篇：[02_skrl程序架构](./02_skrl程序架构.md)　　→ 下一篇：[04_PPO算法详解](./04_PPO算法详解.md)*

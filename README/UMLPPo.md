@startuml
skinparam classAttributeIconSize 0
skinparam monochrome true

abstract class Memory {

+ memory_size: int
+ num_envs: int
+ filled: bool
+ env_index: int
+ memory_index: int
+ tensors: dict
+ tensors_view: dict
+ create_tensor(name, size, dtype)
+ add_samples(tensors)
+ {abstract} sample()
+ sample_by_index(names, indexes)
+ sample_all(names, mini_batches)
  }

class RandomMemory {

+ replacement: bool
+ sample()
  }

Memory <|-- RandomMemory

class PPO {

+ policy: Model
+ value: Model
+ optimizer: Optimizer
+ memory: Memory
+ act()
+ record_transition()
+ update()
  }

PPO o--> Memory : 存储交互数据

note right of Memory
  `<b>`核心数据结构 (张量形状):`</b>`
  (memory_size, num_envs, data_size)

* `<b>`memory_index`</b>`: 环形缓冲区的时间步索引
* `<b>`env_index`</b>`: 并行环境的索引

  `<b>`PPO 初始化的具体张量 (在内存中创建):`</b>`

- observations (float32)
- states (float32)
- actions (float32)
- rewards (float32)
- terminated (bool)
- log_prob (float32)
- values (float32)
- returns (float32)
- advantages (float32)
  end note
  @enduml

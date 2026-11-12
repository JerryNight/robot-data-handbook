# ROADMAP

目标：从零到能看懂 VLA，并能为"机器人大脑的上游数据平台"做出技术判断。

## 阶段划分

### 阶段 0 · 建立心智模型
先有全景图，再钻细节。跳过这步会陷入术语泥潭。

- [x] `00-mental-model/what-is-robot-data.md` —— 机器人数据的四种形态与本质特征
- [x] `00-mental-model/the-stack.md` —— 机器人系统五层栈：硬件 → 驱动 → 中间件 → 算法 → 学习
- [x] `00-mental-model/analogies.md` —— 用软件工程概念映射机器人概念的对照表

### 阶段 1 · 机器人最小集
只为读懂数据服务，不做完整机器人学。

- [x] `01-robotics-primer/dof-and-joints.md` —— 自由度、关节空间 vs 笛卡尔空间、关节类型
- [x] `01-robotics-primer/coordinate-frames.md` —— 坐标系、TF2、URDF（数据对齐的地基）
- [x] `01-robotics-primer/kinematics.md` —— 正运动学、逆运动学、雅可比、奇异性
- [x] `01-robotics-primer/control-101.md` —— 位置/速度/力矩控制、PID、阻抗控制

### 阶段 2 · 传感器数据
每种传感器：输出什么张量、多高频率、噪声特性、数据量级。

- [x] `02-sensor-data/camera-rgbd.md` —— 针孔与坐标系、内参与畸变、三种深度原理、视差精度、三种假点、外参、卷帘快门、存储
- [x] `02-sensor-data/imu-and-proprioception.md` —— IMU 漂移、重力泄漏、ZUPT、编码器 vs 机械回差
- [x] `02-sensor-data/force-torque-tactile.md` —— 六维力、重力补偿、零点漂移、触觉碎片化
- [x] `02-sensor-data/lidar-pointcloud.md` —— 各向异性网格、空洞率、存储与量化
- [x] `02-sensor-data/time-sync.md` —— 对齐误差的上限、插值方向、时间戳来源、晶振漂移
- [x] `02-sensor-data/action-and-state.md` —— ★ 动作空间与状态向量（最容易被忽视）

### 阶段 3 · 数据流水线（核心章节）
- [x] `03-data-pipeline/recording.md` —— rosbag2 的容器结构，metadata 对账，两种静默缺陷
- [x] `03-data-pipeline/clock-and-trigger.md` —— 漂移/抖动/PTP 天花板/触发链预算
- [x] `03-data-pipeline/calibration.md` —— 焦距-距离退化、残差指纹、借重力检测漂移
- [x] `03-data-pipeline/formats.md` —— I/O 放大 100 倍、编码取舍、LeRobotDataset 结构
- [x] `03-data-pipeline/storage-and-versioning.md` —— 去重实测、血缘哈希、manifest 的边界
- [x] `03-data-pipeline/annotation.md` —— kappa 悖论、阈值敏感性、人工精度上限

### 阶段 4 · 公开数据集
- [x] `04-datasets/open-x-embodiment.md` —— 22 本体 / 7 维动作的代价：冗余自由度、量化、量纲
- [x] `04-datasets/droid.md` —— 564 场景的采集成本：0.35% 人力效率、覆盖饱和曲线
- [x] `04-datasets/self-collected.md` —— 七项自检：动作-状态一致性、时间戳、去重、饱和、收尾、标注、元数据

### 阶段 5 · 从数据学策略
- [x] `05-learning-from-data/imitation-learning.md` —— BC / DAgger / ACT / Diffusion Policy
- [x] `05-learning-from-data/action-representation.md` —— ★ 离散 token vs 连续回归 vs flow matching
- [x] `05-learning-from-data/evaluation.md` —— 为什么机器人评测这么难

### 阶段 6 · VLA
- [x] `06-vla/what-is-vla.md` —— 和"视觉+语言模型"的本质区别
- [x] `06-vla/architecture.md` —— 视觉编码器 / 语言主干 / 动作头的三种接法
- [x] `06-vla/models.md` —— RT-2 / OpenVLA / π0 / GR00T 的取舍

### 阶段 7 · 开放问题（北极星）
- [x] `07-open-problems/data-platform.md` —— ★ 全书收口：两段损耗的总账、跨本体代价、自动判定的正确指标

## 当前进度

| 阶段 | 状态 |
|---|---|
| 0 心智模型 | ✅ 3/3 |
| 1 机器人最小集 | ✅ 4/4（每篇配可运行实验） |
| 2 传感器数据 | ✅ 6/6（每篇配可运行实验） |
| 3 数据流水线 | ✅ 6/6（每篇配可运行实验） |
| 4 公开数据集 | ✅ 3/3（每篇配可运行实验） |
| 5 从数据学策略 | ✅ 3/3（每篇配可运行实验） |
| 6 VLA | ✅ 3/3（每篇配可运行实验） |
| 7 开放问题 | ✅ 1/1（配可运行实验） |

共 **29 篇**，**全部完成**。可运行实验 **26 个**（`examples/`），全部零依赖、CI 每次运行。

### 接下来可以做什么

这本书已经写完，但它是一份**活文档**——这个领域半年就换代。三个方向：

1. **补新模型**：`06-vla/models.md` 的核对清单还在，新模型按那六条过一遍就能定位。
2. **补实验**：每个 `examples/` 脚本顶上都有「这个实验在验证什么」的注释，换个假设重跑就能得到新结论。
3. **补数据侧**：真正缺的是「同一份数据在不同流水线里损耗多少」的对照实验——需要真实数据，这个仓库里没有。

## 写作原则

- 每篇必须能回答：**做数据平台的人读完要记住什么**
- 事实性数字标注来源与时间（这个领域半年就变）
- 宁可写短、写准，不写长而空
- 遇到不确定的地方，明确写"不确定"而不是含糊带过

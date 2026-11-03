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

- [ ] `02-sensor-data/camera-rgbd.md` —— RGB / 深度 / 内外参、曝光、卷帘快门
- [ ] `02-sensor-data/imu-and-proprioception.md` —— IMU、关节编码器、本体感受
- [ ] `02-sensor-data/force-torque-tactile.md` —— 力/力矩传感器、触觉阵列
- [ ] `02-sensor-data/lidar-pointcloud.md` —— 点云格式与稀疏性
- [ ] `02-sensor-data/action-and-state.md` —— ★ 动作空间与状态向量（最容易被忽视）

### 阶段 3 · 数据流水线（核心章节）
- [ ] `03-data-pipeline/recording.md` —— rosbag2 / MCAP 的容器结构与回放语义
- [ ] `03-data-pipeline/time-and-sync.md` —— 时钟域、时间戳来源、多传感器对齐
- [ ] `03-data-pipeline/calibration.md` —— 内参/外参/手眼标定，标定误差如何污染数据
- [ ] `03-data-pipeline/formats.md` —— HDF5 / LeRobotDataset / RLDS / Parquet / 视频编码
- [ ] `03-data-pipeline/storage-and-versioning.md` —— 大数据量存储、切片、版本与血缘
- [ ] `03-data-pipeline/annotation.md` —— 语言指令、成功/失败、阶段切分标注

### 阶段 4 · 公开数据集
- [ ] `04-datasets/open-x-embodiment.md` —— 多本体聚合的范式与代价
- [ ] `04-datasets/droid.md` —— 大规模真实场景采集的组织方式
- [ ] `04-datasets/self-collected.md` —— 自采数据的典型坑（含清单）

### 阶段 5 · 从数据学策略
- [ ] `05-learning-from-data/imitation-learning.md` —— BC / DAgger / ACT / Diffusion Policy
- [ ] `05-learning-from-data/action-representation.md` —— ★ 离散 token vs 连续回归 vs flow matching
- [ ] `05-learning-from-data/evaluation.md` —— 为什么机器人评测这么难

### 阶段 6 · VLA
- [ ] `06-vla/what-is-vla.md` —— 和"视觉+语言模型"的本质区别
- [ ] `06-vla/architecture.md` —— 视觉编码器 / 语言主干 / 动作头的三种接法
- [ ] `06-vla/models.md` —— RT-2 / OpenVLA / π0 / GR00T 的取舍

### 阶段 7 · 开放问题（北极星）
- [ ] `07-open-problems/data-platform.md` —— 采集成本、跨本体复用、自动标注、评测基准

## 当前进度

| 阶段 | 状态 |
|---|---|
| 0 心智模型 | ✅ 3/3 |
| 1 机器人最小集 | ✅ 4/4（每篇配可运行实验） |
| 2 传感器数据 | ⬜ 0/5 |
| 3 数据流水线 | ⬜ 0/6 |
| 4 公开数据集 | ⬜ 0/3 |
| 5 从数据学策略 | ⬜ 0/3 |
| 6 VLA | ⬜ 0/3 |
| 7 开放问题 | ⬜ 0/1 |

共 **28 篇**，已完成 **7 篇**，可运行实验 **4 个**（`examples/`）。

下一步：阶段 2 传感器数据，从 `camera-rgbd.md` 和 `action-and-state.md` 开始。

## 写作原则

- 每篇必须能回答：**做数据平台的人读完要记住什么**
- 事实性数字标注来源与时间（这个领域半年就变）
- 宁可写短、写准，不写长而空
- 遇到不确定的地方，明确写"不确定"而不是含糊带过

# 可运行示例

这里的每个脚本都能**直接跑，零依赖**——只用 Python 标准库，不需要 `pip install`。

```bash
python examples/01_robotics_primer/01_joint_space_vs_cartesian.py
```

## 为什么零依赖

两个考虑：

1. **降低门槛**。读者 clone 下来就能跑，不用配环境、不用纠结 numpy 版本。
2. **保证代码不会烂**。没有依赖就没有版本冲突，而且 CI 能在两个 Python 版本上验证。

代价是没有 matplotlib，图表用 ASCII 字符画。对理解原理够用，而且逼着你看数值而不是看漂亮的曲线。

## 目录

### 阶段 1 · 机器人最小集

| 脚本 | 对应文章 | 演示什么 |
|---|---|---|
| `01_robotics_primer/01_joint_space_vs_cartesian.py` | [自由度与关节空间](../01-robotics-primer/dof-and-joints.md) | FK/IK、多解、奇异性、工作空间 |
| `01_robotics_primer/02_frames_and_transforms.py` | [坐标系与 TF](../01-robotics-primer/coordinate-frames.md) | 齐次变换、TF 树、外参误差、旋转插值的坑 |
| `01_robotics_primer/03_ik_and_singularity.py` | [运动学](../01-robotics-primer/kinematics.md) | 数值 IK、冗余、奇异、关节限位 |
| `01_robotics_primer/04_pid_and_impedance.py` | [控制入门](../01-robotics-primer/control-101.md) | PID、静摩擦、阻抗控制、数据特征对照 |

### 阶段 2 · 传感器数据

| 脚本 | 对应文章 | 演示什么 |
|---|---|---|
| `02_sensor_data/01_action_space.py` | [动作与状态](../02-sensor-data/action-and-state.md) | 三种动作表示、归一化三个坑、开环误差累积、四种控制层级、chunk vs 延迟、逆解两支 |
| `02_sensor_data/02_camera_model.py` | [RGB 与深度](../02-sensor-data/camera-rgbd.md) | 针孔模型、内参缩放、畸变、视差精度、深度量化、外参误差、卷帘快门、存储体积 |
| `02_sensor_data/03_imu_drift.py` | [IMU 与本体感知](../02-sensor-data/imu-and-proprioception.md) | 零偏双积分、重力泄漏、ZUPT、编码器 vs 回差 |
| `02_sensor_data/04_time_sync.py` | [时间同步](../02-sensor-data/time-sync.md) | 最近邻两个方向、插值、时间戳来源、晶振漂移 |
| `02_sensor_data/05_force_torque.py` | [力/力矩与触觉](../02-sensor-data/force-torque-tactile.md) | 重力补偿、三种错误读法、零点漂移、触觉带宽 |
| `02_sensor_data/06_pointcloud.py` | [点云](../02-sensor-data/lidar-pointcloud.md) | 各向异性网格、空洞率、存储格式、量化压缩 |

### 阶段 3 · 数据流水线

| 脚本 | 对应文章 | 演示什么 |
|---|---|---|
| `03_data_pipeline/01_rosbag2_layout.py` | [录制格式](../03-data-pipeline/recording.md) | 生成真实结构的 bag、metadata 对账、丢帧与时钟跳变检测 |
| `03_data_pipeline/02_clock_and_trigger.py` | [时钟与触发链](../03-data-pipeline/clock-and-trigger.md) | ppm 漂移、软件/硬件打戳、PTP 不对称、触发链预算 |
| `03_data_pipeline/03_calibration.py` | [标定](../03-data-pipeline/calibration.md) | 焦距-距离退化、残差指纹、多视角重影、借重力查漂移 |
| `03_data_pipeline/04_formats.py` | [格式](../03-data-pipeline/formats.md) | 五种布局的 I/O 放大、编码取舍、LeRobotDataset、转换成本 |
| `03_data_pipeline/05_storage_versioning.py` | [存储与版本](../03-data-pipeline/storage-and-versioning.md) | 按块去重、血缘哈希、子集引用、manifest 校验 |
| `03_data_pipeline/06_annotation.py` | [标注](../03-data-pipeline/annotation.md) | kappa 悖论、阈值敏感性、人工切分精度、语言熵 |

### 阶段 4 · 公开数据集

| 脚本 | 对应文章 | 演示什么 |
|---|---|---|
| `04_datasets/01_schema_alignment.py` | [多本体聚合](../04-datasets/open-x-embodiment.md) | 四种 schema 的字段矩阵、零空间占比、量化放大、频率与量纲、装不下的本体 |
| `04_datasets/02_collection_cost.py` | [DROID](../04-datasets/droid.md) | 公开数字对账、人力效率 0.35%、场景覆盖饱和曲线、每场景预算取舍 |
| `04_datasets/03_dataset_linter.py` | [自采数据](../04-datasets/self-collected.md) | 七项检查，在埋了缺陷的合成数据上验证召回率 |

### 阶段 5 · 从数据学策略

| 脚本 | 对应文章 | 演示什么 |
|---|---|---|
| `05_learning_from_data/01_bc_and_dagger.py` | [模仿学习](../05-learning-from-data/imitation-learning.md) | 起点偏移成功率曲线、覆盖缺口指标、DAgger 迭代、双峰动作、chunk 与反应延迟 |
| `05_learning_from_data/02_action_representation.py` | [动作表示](../05-learning-from-data/action-representation.md) | 量化误差 Δ/√12、全局 vs 逐维归一化、token 吞吐预算、量化与安全余量 |
| `05_learning_from_data/03_evaluation_power.py` | [评测](../05-learning-from-data/evaluation.md) | Wilson 区间、两比例功效分析、测试分布对结论的影响、评测集泄漏 |

### 阶段 6 · VLA

| 脚本 | 对应文章 | 演示什么 |
|---|---|---|
| `06_vla/01_what_language_carries.py` | [VLA 到底是什么](../06-vla/what-is-vla.md) | 指令指代力（候选数与剩余比特）、可行接近方向的几何收缩、文字接口的带宽上限 |
| `06_vla/02_wiring_cost.py` | [架构](../06-vla/architecture.md) | 带宽下限 vs 实测延迟、30 Hz 预算反推、按方差 vs 按任务选特征的 R²、prefill 算力墙 / decode 带宽墙 |
| `06_vla/03_data_ledger.py` | [四个模型的账本](../06-vla/models.md) | 图像/动作字节比、帧率与帧间冗余、采集 vs 训练的算力成本 |

### 阶段 7 · 开放问题

| 脚本 | 对应文章 | 演示什么 |
|---|---|---|
| `07_open_problems/01_platform_ledger.py` | [数据平台](../07-open-problems/data-platform.md) | 工位时间账与 Amdahl（布景摊销 1.41×）、采集到进梯度的漏斗、三连杆臂的自运动流形与力矩歧义、合并语料的分辨率损失、kappa 与阈值分歧率、标注吞吐 |

## 环境要求

- Python 3.9+
- 无第三方依赖

Windows 下如果中文输出乱码，加环境变量：

```bash
PYTHONUTF8=1 python examples/01_robotics_primer/01_joint_space_vs_cartesian.py
```

（脚本内部已经尝试 `sys.stdout.reconfigure(encoding="utf-8")`，大多数情况不需要手动设置。）

## 改造这些脚本

每个脚本都是**自包含**的：所有函数都在一个文件里，没有 import 其他模块，方便你直接改。

建议的玩法：

- 改参数，看结论是否还成立（比如把连杆长度比例改极端、把 IMU 零偏调大一个数量级）
- 把输出表格和文章里的数字对比，确认自己跑出的结果一致
- 把某个函数抽出来，用到自己的项目里（都是 MIT 许可）

## 一键跑全部

如果改动了多个脚本，跑这个确认都没坏：

```bash
python scripts/run_all_examples.py          # 静默
python scripts/run_all_examples.py -v       # 显示每个脚本的完整输出
```

它会依次执行所有示例，任何一个报错就退出并打印错误。

## CI

仓库挂了 GitHub Actions（`.github/workflows/ci.yml`），每次 push 和 PR 会跑两件事：

| Job | 内容 |
|---|---|
| `examples` | 在 Python 3.9 和 3.12 上跑全部示例 |
| `links` | 检查所有 Markdown 里的相对链接是否有效 |

第一个 job 同时也是一道**依赖闸门**：因为 CI 里不执行 `pip install`，一旦有人给示例引入了第三方库，构建会立刻红。

本地想复现这两个检查：

```bash
python scripts/run_all_examples.py
python scripts/check_links.py
```

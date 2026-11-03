# GLOSSARY · 中英术语对照

机器人领域最大的入门门槛不是数学，是**黑话**。同一个东西在不同资料里有三四种叫法，论文里的英文和国内的译法又对不上。

这份表按主题分组，每条给一句话解释——不是字典式定义，是"它到底在说什么"。

> 最后更新：2026-11。这个领域术语变动快，发现过时或译错的请提 issue。

## 1. 机器人与运动

| English | 中文 | 一句话 |
|---|---|---|
| DoF (Degree of Freedom) | 自由度 | 机器人能独立运动的维度数；7-DoF 机械臂 = 7 个关节各转一个角度 |
| Joint Space | 关节空间 | 用每个关节的角度/位移来描述机器人姿态的坐标系 |
| Cartesian Space | 笛卡尔空间 | 用末端执行器在三维空间的位置 + 姿态来描述状态；与控制方式、任务描述强相关 |
| End Effector (EE) | 末端执行器 | 机械臂最末端那个干活的东西；夹爪、吸盘、灵巧手都算 |
| Forward Kinematics (FK) | 正运动学 | 已知关节角 → 算出末端在哪；有解析解，便宜 |
| Inverse Kinematics (IK) | 逆运动学 | 已知末端要去哪 → 反推关节角；可能无解、多解，是机器人数据里误差的一大来源 |
| Jacobian | 雅可比矩阵 | 关节速度与末端速度的线性映射；奇异位形下不可逆 |
| Singularity | 奇异位形 | 雅可比降秩的姿态，此时某些方向动不了，IK 会炸 |
| Workspace | 工作空间 | 末端能到达的所有位置的集合 |
| Self-Motion Manifold | 自运动流形 | 冗余自由度下、末端位姿不变而关节可自由变的一维（或更高维）解集；07 章实测第一个关节可铺开 136° |
| Elbow-up / Elbow-down | 肘上 / 肘下 | 同一个末端位置的两个 IK 分支；关节向量相差最多 193°，末端坐标完全看不出区别 |
| URDF | 统一机器人描述格式 | 机器人硬件结构的 XML 描述：连杆、关节、惯量、碰撞体。**是机器人界的 Schema** |
| MJCF | MuJoCo 模型格式 | MuJoCo 仿真器用的机器人描述格式，比 URDF 表达力更强 |
| DH Parameters | DH 参数 | 用四个参数描述相邻连杆关系的老派约定，读论文时会遇到 |
| Manipulator | 机械臂/操作器 | 用来抓取、移动物体的机械结构 |
| End-to-End | 端到端 | 从传感器输入直接到动作输出，中间不拆分子模块 |
| Embodiment | 本体 / 具身形态 | 机器人的物理身体配置（几臂、几个轮子、什么相机装哪）；**数据能不能复用常取决于这个** |

## 2. 控制

| English | 中文 | 一句话 |
|---|---|---|
| PID Control | PID 控制 | 用误差的比例/积分/微分项算控制量，最基础也最常用的反馈控制器 |
| Impedance Control | 阻抗控制 | 不直接控位置，而是控制"受到外力时的顺从程度"；接触类任务必需 |
| Admittance Control | 导纳控制 | 阻抗控制的对偶形式：测力 → 算出该走多少 |
| Torque Control | 力矩控制 | 直接给每个关节下发力矩指令，最底层、最灵活、最难 |
| Trajectory | 轨迹 | 随时间变化的位置序列；数据里常以轨迹为单位切分 |
| Waypoint | 路点 | 轨迹上的关键点，不是连续轨迹 |
| MPC (Model Predictive Control) | 模型预测控制 | 每步基于模型预测未来一段并优化，滚动执行 |
| Whole-Body Control (WBC) | 全身控制 | 人形/四足这类多肢体系统的联合控制 |
| Teleoperation | 遥操作 | 人带着机器人动，是当前**真机数据采集的主要方式** |
| Gripper | 夹爪 | 最常用的二指/多指末端，动作通常是一个开合标量 |

## 3. 传感器与硬件

| English | 中文 | 一句话 |
|---|---|---|
| Proprioception | 本体感受 | 机器人对自身状态的感知：关节角、速度、力矩 |
| Encoder | 编码器 | 装在关节上测角度的传感器 |
| IMU (Inertial Measurement Unit) | 惯性测量单元 | 测加速度 + 角速度，频率高（100–1000 Hz）、漂移大 |
| RGB-D Camera | 彩色深度相机 | 同时输出彩色图和逐像素深度 |
| Intrinsics | 内参 | 相机自身的焦距、主点、畸变系数 |
| Extrinsics | 外参 | 相机相对机器人本体的位姿；**外参错了，整段数据都废** |
| Hand-Eye Calibration | 手眼标定 | 标定相机与末端执行器之间的变换 |
| Force/Torque Sensor (F/T) | 力/力矩传感器 | 装在腕部测六维力，频率可达 1 kHz |
| Tactile Sensor | 触觉传感器 | 贴在指尖测接触分布，目前没有统一格式 |
| LiDAR | 激光雷达 | 扫描测距产出点云，主要用在导航 |
| Point Cloud | 点云 | 一堆 (x, y, z) 点，可能带强度/颜色 |
| Rolling Shutter | 卷帘快门 | 逐行曝光，运动快时图像会扭曲；廉价相机的通病 |
| Calibration | 标定 | 确定传感器参数的过程，是数据质量的第一道关口 |

## 4. 中间件与系统

| English | 中文 | 一句话 |
|---|---|---|
| ROS / ROS 2 | 机器人操作系统 | 事实标准的中间件；ROS 2 底层换成 DDS，支持实时与多机 |
| DDS | 数据分发服务 | ROS 2 底层的发布订阅协议，有 QoS 概念 |
| Topic | 话题 | 发布订阅的通道，例如 `/camera/image_raw` |
| Node | 节点 | 一个跑着的进程单元 |
| TF / TF2 | 坐标变换树 | 维护整机各坐标系之间的时空关系，带时间戳的历史变换 |
| Launch File | 启动文件 | 一次拉起多个节点的配置 |
| rosbag / rosbag2 | 数据包 | ROS 的录制格式，把 topic 按时间录下来可回放 |
| MCAP | MCAP 容器 | 现代化机器人日志容器格式，rosbag2 已支持，比老 bag 更好索引 |
| QoS | 服务质量 | 决定可靠性/实时性取舍的策略，配错了会丢数据 |
| Sim2Real | 仿真到真实 | 仿真训练的模型迁移到真机，核心难点是分布差异 |
| Domain Randomization | 域随机化 | 训练时随机化仿真参数，逼模型学会鲁棒 |
| Digital Twin | 数字孪生 | 真实机器人在仿真里的高保真副本 |

## 5. 数据

| English | 中文 | 一句话 |
|---|---|---|
| Demonstration | 示范 / 演示数据 | 人操作（遥操作或拖动）产生的一段完整任务轨迹，模仿学习的原料 |
| Episode | 回合 | 一次从开始到结束的完整尝试，是数据集的基本单位 |
| Trajectory | 轨迹 | 一个 episode 内的状态-动作序列 |
| Rollout | 推演/采样 | 让策略在环境里跑一遍产生的数据 |
| HDF5 | HDF5 文件格式 | 层级式科学数据容器，ACT/ALOHA 系数据集常用 |
| RLDS | RLDS 数据集格式 | Google 的 RL Datasets 格式（TFDS 之上），Open X-Embodiment 用 |
| LeRobotDataset | LeRobot 数据集格式 | HuggingFace 定义，Parquet + 视频，正在成为社区事实标准 |
| Parquet | 列式存储格式 | 分析型数据首选，机器人状态/动作序列也大量用它 |
| Zarr | 分块数组格式 | 适合超大张量，云端切片友好 |
| Dataset Versioning | 数据集版本化 | 数据也会变，需要像代码一样的版本与血缘 |
| Data Lineage | 数据血缘 | 一条样本从哪来、经过哪些处理 |
| Annotation | 标注 | 给数据打标签：语言指令、成功失败、阶段切分 |
| Action Space | 动作空间 | 动作向量的定义（维度、范围、含义）。**不统一，跨本体数据就没法合并** |
| Observation / `obs` | 观测 | 策略在某一时刻看到的全部输入（图像 + 本体状态） |

## 6. 学习方法

| English | 中文 | 一句话 |
|---|---|---|
| Behavior Cloning (BC) | 行为克隆 | 最直接的模仿学习：监督学习，输入观测输出动作 |
| DAgger | 数据集聚合 | 让策略自己跑、人纠正错误，解决 BC 的分布漂移 |
| Imitation Learning (IL) | 模仿学习 | 从示范数据学策略的总称 |
| Reinforcement Learning (RL) | 强化学习 | 靠奖励信号试错学习，真机成本高，多在仿真里做 |
| Diffusion Policy | 扩散策略 | 用扩散模型生成动作序列，能表达多模态动作分布 |
| ACT (Action Chunking Transformer) | 动作分块 Transformer | ALOHA 用的方法，一次输出一段动作，性价比极高 |
| Policy | 策略 | 从观测到动作的映射函数，就是"大脑" |
| Generalist Policy | 通用策略 | 一个模型跨任务、跨本体干活 |
| Foundation Model | 基础模型 | 大规模预训练的通用模型，机器人领域对标 LLM |
| Covariate Shift | 分布偏移 / 协变量偏移 | 策略跑起来后访问的状态，和训练数据的状态分布不一致；BC 的核心失败模式 |
| Compounding Error | 误差累积 | 每一步的小误差改变下一步的输入，误差随步数放大；长程任务的主要死因 |
| Multimodal Action | 多模态动作 | 同一状态下有多种正确动作；MSE 回归必然给出它们的均值（通常不可行） |
| Action Chunking | 动作分块 | 一次预测 k 步动作并整段执行，降低决策频率、提高连贯性，代价是反应延迟 |
| Temporal Ensembling | 时序集成 | 对重叠 chunk 的预测做加权平均，缓解 chunk 带来的反应延迟（ACT 的做法） |
| State Coverage | 状态覆盖 | 训练数据实际覆盖到的状态集合；比 episode 条数更能预测部署表现 |
| Quantization Bin | 量化分箱 | 把连续动作切成 N 档；步长 Δ 决定精度上限（RMSE = Δ/√12）和安全余量（≥ Δ/2） |

## 7. VLA 与基础模型

| English | 中文 | 一句话 |
|---|---|---|
| VLA (Vision-Language-Action) | 视觉-语言-动作模型 | 输入图像 + 自然语言指令，直接输出机器人动作的模型 |
| VLM (Vision-Language Model) | 视觉语言模型 | VLA 的"前半身"，能看图说话但不能动 |
| Action Head | 动作头 | 接在 VLM 后面把隐状态转成动作的模块，VLA 的关键设计点 |
| Discrete Action Token | 离散动作 token | 把动作切成 bin 当词表预测，复用 LLM 的交叉熵，简单但精度受限 |
| Flow Matching | 流匹配 | π0 用的动作生成方式，连续、平滑，比扩散采样快 |
| RT-1 / RT-2 | （Google 系列） | 最早的 VLA 代表工作；RT-2 首次把网络知识迁到机器人动作 |
| OpenVLA | 开源 VLA | 7B 参数开源 VLA，社区微调基线 |
| π0 / π0.5 | （Physical Intelligence） | 主打 flow matching 动作头的通用策略 |
| GR00T | （NVIDIA） | 面向人形的通用机器人基础模型 |
| Open X-Embodiment | 开放跨本体数据集 | 聚合 20+ 种机器人、百万级 episode 的联合数据集 |
| DROID | 大规模真实操作数据集 | 多机构分布式采集，主打场景多样性 |
| Cross-Embodiment | 跨本体 | 一个模型吃多种机器人的数据，泛化的核心命题 |
| Action Expert | 动作专家 | π0 里那 300M 参数的小网络，专门负责去噪出动作，与 VLM 共享注意力 |
| Dual-System | 双系统 | GR00T N1 的做法：慢的视觉语言系统（10 Hz）+ 快的动作系统（120 Hz） |
| FAST | 频域动作 tokenizer | DCT + BPE 压缩动作序列，约 10 倍压缩，让自回归 VLA 能处理高频动作 |
| Referential Ambiguity | 指代歧义 | 「拿红色的」在 20 个物体里还剩 5.41 个候选，缺的 2.44 bit 只能从图像补 |
| Approach Direction | 接近方向 | 夹爪/小臂接近目标的方位；拥挤桌面上可行方向可能只剩三成 |

## 7.5 推理成本（做部署/选型时会用到）

| English | 中文 | 一句话 |
|---|---|---|
| Prefill | 预填充 | 把全部输入 token（含图像）跑一遍；算力受限，图像 token 全在这里 |
| Decode | 解码 | 逐 token 生成；带宽受限，每个 token 都要把权重读一遍 |
| Prefix Caching | 前缀缓存 | 缓存不变的图像/指令 KV，避免每步重算 prefill |
| Roofline / Bandwidth Bound | 带宽墙 | batch=1 推理时算力利用率常低于 1%，延迟由显存流量决定 |
| Effective Bandwidth | 有效带宽 | 实测能拿到的带宽，通常是峰值的 30%~90%，取决于 kernel 大小 |
| Parallel Decoding | 并行解码 | 一次前向出整块动作，把 N 次权重读取压成 1 次 |
| Real-time Margin | 实时余量 | 一次决策买到的时长 ÷ 决策耗时；小于 1 就是物理上跑不到这个频率 |

## 8. 评测

| English | 中文 | 一句话 |
|---|---|---|
| Success Rate | 成功率 | 最常用指标，但采样次数少时方差极大 |
| Generalization | 泛化 | 换物体/换场景/换光照还能不能干成 |
| LIBERO / SimplerEnv / RoboArena | 评测基准 | 常见仿真或混合评测环境，真机评测仍无统一标准 |
| Held-out Task | 留出任务 | 训练时没见过的任务，用来测真泛化 |
| Confidence Interval (Wilson) | 置信区间 | 成功率的误差棒；20 次试验的 75% 对应区间宽度约 36 个百分点 |
| Statistical Power | 统计功效 | 检出真实差异的概率；区分 70% vs 80% 每组需约 290 次试验 |
| Data Leakage (Scene-level) | 评测集泄漏 | 评测场景与训练场景重叠，导致成功率虚高；20% 重叠约虚高 7 pt |
| Cohen's Kappa | 一致性系数 | 扣掉「瞎猜也能对」之后的标注一致性；成功率 92% 时全判成功准确率 92% 但 kappa = 0 |
| Inter-rater Disagreement | 标注分歧率 | 两个人独立标注同一个连续量的预期分歧比例，只由 `\|误差−阈值\|/σ` 决定 |

## 9. 数据平台（做上游平台时的口径）

| English | 中文 | 一句话 |
|---|---|---|
| Station Hour | 工位小时 | 采集系统的计费单位：人 + 机器人 + 场地占用的时间；**不是**数据时长 |
| Effective Data Ratio | 有效数据占比 | 遥操作时长 / 工位时间；本章模型里是 24.4% |
| Retention Funnel | 留存漏斗 | 采到 → 成功 → 质检 → 去重 → 多样性；乘积约 41% |
| Amdahl's Law | 阿姆达尔定律 | 只优化一部分，整体加速受它在总量里的占比限制；遥操作占 33%，减半只换来 19% |
| Scene Amortization | 布景摊销 | 一次摆好场景连续录 N 条，把布景成本摊薄；1.41× 且不需要新硬件 |
| Normalization Scope | 归一化作用域 | 逐维 / 逐本体 / 全局；选全局会让小工作空间本体只剩 8.5/256 个档位 |
| Continuous Success Metric | 连续成败量 | 存末端误差、接触力等连续值而不是 bool；bool 无法溯源、无法定位系统性偏差 |
| Label Provenance | 标签溯源 | 标签 + 判据 + 置信度 + 谁标的 + 什么版本；没有它，标签出错时无法回溯 |

# 02 · 传感器数据

每种传感器：输出什么张量、多高频率、什么噪声特性、数据量级多大。

这一章是数据平台的**输入层**。搞清楚每种模态的数据形态，才知道存储、传输、对齐该怎么设计。

**每篇都配了一个可运行的实验脚本**，零依赖，跑一遍比读三遍有用。

## 本章文章

| 文章 | 配套实验 | 要回答的问题 |
|---|---|---|
| [camera-rgbd.md](camera-rgbd.md) | [02_camera_model.py](../examples/02_sensor_data/02_camera_model.py) | 一张图里必须带哪些参数？缺了会怎样？ |
| [imu-and-proprioception.md](imu-and-proprioception.md) | [03_imu_drift.py](../examples/02_sensor_data/03_imu_drift.py) | IMU 为什么不能用？编码器精度到底卡在哪？ |
| [force-torque-tactile.md](force-torque-tactile.md) | [05_force_torque.py](../examples/02_sensor_data/05_force_torque.py) | 六维力的读数里混着什么？触觉为什么无法聚合？ |
| [lidar-pointcloud.md](lidar-pointcloud.md) | [06_pointcloud.py](../examples/02_sensor_data/06_pointcloud.py) | 点云为什么不能当成图？稀疏到什么程度？ |
| [time-sync.md](time-sync.md) | [04_time_sync.py](../examples/02_sensor_data/04_time_sync.py) | 不同频率的传感器怎么拼成"同一时刻"？ |
| [action-and-state.md](action-and-state.md) | [01_action_space.py](../examples/02_sensor_data/01_action_space.py) | ★ 动作空间怎么定义？为什么这是跨本体的卡点？ |

## 数据量级参考（典型 30 秒 episode）

| 模态 | 频率 | 原始大小量级 |
|---|---|---|
| 3 路 RGB (640×480) | 30 Hz | 压缩后 5–20 MB |
| 本体状态 (14 维) | 100 Hz | < 1 MB |
| 力矩 (6 维) | 1 kHz | 数 MB |
| 深度 (640×480) | 30 Hz | 压缩后 30–80 MB |
| 16 线点云 | 10 Hz | 压缩后 5–15 MB |

结论：**图像/深度占 95% 以上的体积**。这是为什么主流格式都选择把视频单独编码，而不是塞进列式存储。

详细的推导在 [camera-rgbd.md §7](camera-rgbd.md) —— 原始 RGB 和 H.264 差 147 倍。

## 这一章的四条主线

**1. 每个传感器的"原始值"和"物理量"之间隔着一段换算。**
相机隔着内参，IMU 隔着零偏和姿态，力传感器隔着重力补偿，编码器隔着减速比和零位。
把换算参数丢掉，数据就退化成数字。

**2. 所有传感器的误差都会在某个维度上放大。**
相机的外参误差随距离线性放大，IMU 的零偏误差随时间的平方放大，力传感器的重力分量随姿态变化。
**哪一个维度在放大，决定了这个模态适合什么任务、不适合什么任务。**

**3. 最隐蔽的错误是"数值合理但物理上错"。**
力传感器把重力算成接触力、深度图把量化台阶算成噪声、IMU 把姿态误差吸收成零偏 —— 这些都不会报错，不会让 loss 变差，只会让模型在真机上失效。

**4. 时间对齐是所有模态共同的、且只能在采集端解决的问题。**
模型侧完全无能为力。见 [time-sync.md](time-sync.md)。

## 读完该记住的四件事

1. 一份图像数据的可用性，取决于它带了几个参数（内参/外参/单位/时间戳来源），不取决于分辨率。
2. IMU 能自洽但不能自证 —— 没有绝对参考，姿态永远不可恢复。
3. 触觉的瓶颈是语义不统一，不是带宽。
4. `action` 的语义包含"控制层级"，比维度对齐更根本。

详见 [ROADMAP](../ROADMAP.md)。

# 01 · 机器人最小集

只为读懂数据服务，不做完整机器人学。

这一章的目的不是让你会造机器人，而是让你在看到一个数据字段 `qpos[7]`、一个 `base_link → wrist_3_link` 的变换时，知道它在物理上指什么。

**每篇都配了一个可运行的实验脚本**，零依赖，跑一遍比读三遍有用。

## 本章文章

| 文章 | 配套实验 | 要回答的问题 |
|---|---|---|
| [dof-and-joints.md](dof-and-joints.md) | [01_joint_space_vs_cartesian.py](../examples/01_robotics_primer/01_joint_space_vs_cartesian.py) | 自由度是什么？关节空间和笛卡尔空间差在哪？为什么这个区别影响动作定义 |
| [coordinate-frames.md](coordinate-frames.md) | [02_frames_and_transforms.py](../examples/01_robotics_primer/02_frames_and_transforms.py) | 坐标系为什么需要一棵树？TF2 怎么维护时间维度？外参误差怎么毁数据 |
| [kinematics.md](kinematics.md) | [03_ik_and_singularity.py](../examples/01_robotics_primer/03_ik_and_singularity.py) | 雅可比矩阵到底在算什么？逆运动学为什么没有唯一解？冗余怎么用？奇异和关节限位怎么毁数据 |
| [control-101.md](control-101.md) | [04_pid_and_impedance.py](../examples/01_robotics_primer/04_pid_and_impedance.py) | 位置/速度/力矩控制有什么本质区别？你的数据是在哪一层记录的 |

## 为什么这章不能跳

跳过坐标系，后面所有多传感器数据都对不齐。这是新手最常见的坑，而且症状是"数据看起来很怪"而不是报错。

跳过控制模式，你会把位置控制和力矩控制的 action 混在一起训练，loss 正常但效果莫名其妙地差。

## 一条验证链

这四篇构成一条完整的验证链，可以串起来理解：

```
关节角 --FK--> 末端位置 --TF--> 世界坐标 --控制--> 下一帧关节角
  ↑                                              |
  └──────────── 闭环反馈 ←────────────────────────┘
```

- `dof-and-joints` 讲第一步（FK 和它的多解性质）
- `coordinate-frames` 讲第二步（坐标系变换）
- `kinematics` 讲第一步的逆问题（IK，以及为什么它会失败）
- `control-101` 讲最后一步（怎么从当前位置走到目标）

## 读完该记住的四件事

1. `qpos` 是关节角，不是坐标，不要拿它做笛卡尔式的插值和距离计算。
2. 变换必须带时间戳，插值旋转必须用 slerp，不能用矩阵线性插值。
3. 冗余导致"同一末端轨迹对应无数组关节轨迹"，这是动作表示两难的根源。
4. `action` 的语义包含"控制层级"，比维度对齐更根本。

详见 [ROADMAP](../ROADMAP.md)。

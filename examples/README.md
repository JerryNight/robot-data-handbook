# 可运行示例

这里的每个脚本都能**直接跑，零依赖**——只用 Python 标准库，不需要 `pip install`。

```bash
python examples/01_robotics_primer/01_joint_space_vs_cartesian.py
```

## 为什么零依赖

两个考虑：

1. **降低门槛**。读者 clone 下来就能跑，不用配环境、不用纠结 numpy 版本。
2. **保证代码不会烂**。没有依赖就没有版本冲突，三年后还能跑。

代价是没有 matplotlib，图表用 ASCII 字符画。对理解原理够用，而且逼着你看数值而不是看漂亮的曲线。

## 目录

| 脚本 | 对应文章 | 演示什么 |
|---|---|---|
| `01_robotics_primer/01_joint_space_vs_cartesian.py` | [自由度与关节空间](../01-robotics-primer/dof-and-joints.md) | FK/IK、多解、奇异性、工作空间 |
| `01_robotics_primer/02_frames_and_transforms.py` | [坐标系与 TF](../01-robotics-primer/coordinate-frames.md) | 齐次变换、TF 树、外参误差、旋转插值的坑 |
| `01_robotics_primer/03_ik_and_singularity.py` | [运动学](../01-robotics-primer/kinematics.md) | 数值 IK、冗余、奇异、关节限位 |
| `01_robotics_primer/04_pid_and_impedance.py` | [控制入门](../01-robotics-primer/control-101.md) | PID、静摩擦、阻抗控制、数据特征对照 |

## 环境要求

- Python 3.8+
- 无第三方依赖

Windows 下如果中文输出乱码，加环境变量：

```bash
PYTHONUTF8=1 python examples/01_robotics_primer/01_joint_space_vs_cartesian.py
```

（脚本内部已经尝试 `sys.stdout.reconfigure(encoding="utf-8")`，大多数情况不需要手动设置。）

## 改造这些脚本

每个脚本都是**自包含**的：所有函数都在一个文件里，没有 import 其他模块，方便你直接改。

建议的玩法：

- 改参数，看结论是否还成立（比如把连杆长度比例改极端）
- 把输出表格和文章里的数字对比，确认自己跑出的结果一致
- 把某个函数抽出来，用到自己的项目里（都是 MIT 许可）

## 一键跑全部

如果改动了多个脚本，跑这个确认都没坏：

```bash
python scripts/run_all_examples.py
```

它会依次执行所有示例，任何一个报错就退出并打印错误。

## 关于 CI

仓库暂时没挂 GitHub Actions —— 添加 workflow 文件需要 GitHub token 带 `workflow` 权限，当前授权的 scope 不含它。等文章稳定后再补。

在那之前，`run_all_examples.py` 就是手动的检查闸门。

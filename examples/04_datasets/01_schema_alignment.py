#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验 1：多本体对齐的代价

公开数据集看起来是「一堆 episode」，实际是一堆互不兼容的约定。
本实验量化三件事：

  [1] 四种 schema 的字段交集有多小
  [2] 动作统一到 7 维末端空间，丢掉了什么，丢多少
  [3] 控制频率不同，同样的物理速度对应完全不同的动作数值
  [4] 7 维空间装不下的本体

全部可精确计算，不依赖任何外部数据。

运行：python examples/04_datasets/01_schema_alignment.py
依赖：无（只用标准库）
"""

import math


# ---------------------------------------------------------------- 输出对齐

def width(s):
    """终端里一个汉字占两列。不做这个换算，中文表格一定错位。"""
    return sum(2 if ord(c) > 0x2E80 else 1 for c in s)


def pad(s, n):
    return s + " " * max(0, n - width(s))


# ---------------------------------------------------------------- [1] schema

# 三态：✓ 有字段 / △ 只在文档或元数据里，不是机器可读字段 / · 没有
FIELD_ROWS = [
    # (key, 显示名, 缺了会怎样)
    ("image_exterior", "外视相机", "基本的视觉输入"),
    ("image_wrist", "腕部相机", "抓取精度，遮挡时唯一可用视角"),
    ("state_joint", "关节位置", "本体感受，动作对齐的参照"),
    ("state_ee", "末端位姿", "跨本体可比的状态表示"),
    ("action", "动作", "监督信号"),
    ("instruction", "语言指令", "VLA 的语言条件"),
    ("is_first_last", "首末帧标记", "切 episode，算不清就拼错了"),
    ("reward_terminal", "reward/terminal", "区分成败，做筛选"),
    ("intrinsics", "相机内参", "多视角几何、点云重建"),
    ("extrinsics", "相机外参", "跨本体视觉对齐的前提"),
    ("embodiment_id", "本体标识", "按本体分组归一化"),
    ("control_hz", "控制频率", "解释动作的物理量纲"),
    ("success_flag", "成败标注", "过滤失败样本、算成功率"),
]

SCHEMAS = {
    "OXE / RLDS": {
        "image_exterior": "✓", "image_wrist": "·", "state_joint": "✓",
        "state_ee": "·", "action": "✓", "instruction": "✓",
        "is_first_last": "✓", "reward_terminal": "✓", "intrinsics": "·",
        "extrinsics": "·", "embodiment_id": "△", "control_hz": "△",
        "success_flag": "·",
    },
    "DROID": {
        "image_exterior": "✓", "image_wrist": "✓", "state_joint": "✓",
        "state_ee": "✓", "action": "✓", "instruction": "✓",
        "is_first_last": "✓", "reward_terminal": "✓", "intrinsics": "✓",
        "extrinsics": "✓", "embodiment_id": "△", "control_hz": "△",
        "success_flag": "△",
    },
    "LeRobot v2": {
        "image_exterior": "✓", "image_wrist": "✓", "state_joint": "✓",
        "state_ee": "·", "action": "✓", "instruction": "✓",
        "is_first_last": "✓", "reward_terminal": "·", "intrinsics": "·",
        "extrinsics": "·", "embodiment_id": "△", "control_hz": "✓",
        "success_flag": "·",
    },
    "自采（典型）": {
        "image_exterior": "✓", "image_wrist": "·", "state_joint": "✓",
        "state_ee": "△", "action": "✓", "instruction": "△",
        "is_first_last": "·", "reward_terminal": "·", "intrinsics": "·",
        "extrinsics": "·", "embodiment_id": "·", "control_hz": "△",
        "success_flag": "·",
    },
}


def part1_schema_matrix():
    print("\n[1] 四种 schema 的字段覆盖：交集比想象的小")
    print("-" * 78)

    names = list(SCHEMAS)
    head = pad("字段", 16) + "".join(pad(n, 14) for n in names) + "缺了会怎样"
    print("  " + head)
    print("  " + "-" * 76)

    counts = {n: {"✓": 0, "△": 0, "·": 0} for n in names}
    all_have, any_have = [], []
    for key, label, why in FIELD_ROWS:
        row = pad(label, 16)
        marks = []
        for n in names:
            m = SCHEMAS[n][key]
            counts[n][m] += 1
            marks.append(m)
            row += pad(m, 14)
        print("  " + row + why)
        if all(m == "✓" for m in marks):
            all_have.append(label)
        if any(m == "✓" for m in marks):
            any_have.append(label)

    print()
    print("  图例：✓ 有字段  △ 只在文档/元数据里  · 没有")
    print()
    print("  四个数据集都有字段的：%d 个 / %d 个 —— %s"
          % (len(all_have), len(FIELD_ROWS), "、".join(all_have)))
    print("  至少一个有的：%d 个" % len(any_have))
    print()
    print("  关键的四个字段，没有一个被所有数据集做成机器可读字段：")
    for label in ("相机外参", "本体标识", "控制频率", "成败标注"):
        key = [k for k, l, _ in FIELD_ROWS if l == label][0]
        holders = [n for n in names if SCHEMAS[n][key] == "✓"]
        soft = [n for n in names if SCHEMAS[n][key] == "△"]
        if not holders:
            note = "没有一个数据集有字段，%s 只写在文档里" % ("、".join(soft) if soft else "无")
        else:
            note = "字段：" + "、".join(holders)
            if soft:
                note += "；仅文档：" + "、".join(soft)
        print("    %s：%s" % (pad(label, 10), note))
    print()
    print("  这不是巧合。这几个字段恰好是**决定数据能否被复用**的那些：")
    print("    外参 → 视觉跨本体对齐；本体标识 → 分组归一化；")
    print("    控制频率 → 解释动作量纲；成败 → 决定能不能做筛选。")
    print()
    print("  而 OXE 的 60 个数据集，是被**转换**到统一 schema 的。")
    print("  转换只能保留下游 schema 里有的字段 —— 原数据集里的标定、")
    print("  采集元数据，在转换那一步就永久消失了。这是聚合的隐性成本。")


# ---------------------------------------------------------------- [2] 动作投影

def jacobian_2r(q1, q2, l1=0.4, l2=0.4):
    s1, c1 = math.sin(q1), math.cos(q1)
    s12, c12 = math.sin(q1 + q2), math.cos(q1 + q2)
    return [[-l1 * s1 - l2 * s12, -l2 * s12],
            [l1 * c1 + l2 * c12, l2 * c12]]


def sing_vals2(J):
    """2x2 矩阵的奇异值，闭式解。"""
    a, b = J[0]
    c, d = J[1]
    s = a * a + b * b + c * c + d * d
    det = a * d - b * c
    disc = max(0.0, s * s - 4.0 * det * det)
    r = math.sqrt(disc)
    hi = math.sqrt(max(0.0, (s + r) / 2.0))
    lo = math.sqrt(max(0.0, (s - r) / 2.0))
    return hi, lo


def jacobian_3r(q, l=(0.3, 0.3, 0.3)):
    """3R 平面臂：2 维任务空间 + 3 个关节 => 有 1 维零空间。"""
    q1, q2, q3 = q
    l1, l2, l3 = l
    s1, c1 = math.sin(q1), math.cos(q1)
    s12, c12 = math.sin(q1 + q2), math.cos(q1 + q2)
    s123, c123 = math.sin(q1 + q2 + q3), math.cos(q1 + q2 + q3)
    return [
        [-l1 * s1 - l2 * s12 - l3 * s123, -l2 * s12 - l3 * s123, -l3 * s123],
        [l1 * c1 + l2 * c12 + l3 * c123, l2 * c12 + l3 * c123, l3 * c123],
    ]


def null_vector(J):
    """2x3 矩阵的零空间基向量 = 两行的叉积。"""
    r0, r1 = J
    n = [r0[1] * r1[2] - r0[2] * r1[1],
         r0[2] * r1[0] - r0[0] * r1[2],
         r0[0] * r1[1] - r0[1] * r1[0]]
    norm = math.sqrt(sum(v * v for v in n))
    if norm < 1e-12:
        return None
    return [v / norm for v in n]


def part2_projection_loss():
    print("\n[2] 统一到 7 维末端空间：丢掉的是「冗余自由度」")
    print("-" * 78)
    print("  OXE 的做法是把所有本体的动作映射到同一个 7 维向量：")
    print("    [x, y, z, roll, pitch, yaw, gripper]")
    print("  表达的是**末端相对夹爪坐标系的位姿变化**，不是关节角。")
    print()
    print("  对 6 自由度臂，这是无损的（6 维任务空间对 6 个关节）。")
    print("  对 7 自由度臂，有 1 维冗余 —— 那一维在映射里消失了。")
    print("  对平面 3R 臂（2 维任务 + 3 关节），同样丢 1 维。")
    print()

    print("  量化：给定一个关节运动 q̇，它有多少比例落在零空间里？")
    print("  （零空间分量 = 关节在动，末端完全不动 —— 投影后彻底不可见）")
    print()
    print("  %s%s%s%s" % (pad("位姿 (q1,q2,q3)", 22), pad("零空间占比", 14),
                        pad("末端速度", 12), "说明"))
    print("  " + "-" * 66)

    qdot = [1.0, 1.0, 1.0]          # 三个关节同向运动
    norm_q = math.sqrt(sum(v * v for v in qdot))
    for q in ((0.0, math.radians(60), math.radians(-30)),
              (math.radians(30), math.radians(90), math.radians(0)),
              (math.radians(-20), math.radians(30), math.radians(60)),
              (math.radians(10), math.radians(20), math.radians(10))):
        J = jacobian_3r(q)
        n = null_vector(J)
        if n is None:
            print("  %s%s" % (pad("退化位形", 22), "零空间维数 > 1"))
            continue
        dot = sum(a * b for a, b in zip(qdot, n))
        frac = abs(dot) / norm_q
        # 末端速度 = J q̇
        v = [sum(J[i][k] * qdot[k] for k in range(3)) for i in range(2)]
        vmag = math.sqrt(v[0] ** 2 + v[1] ** 2)
        print("  %s%s%s%s" % (
            pad("(%.0f°, %.0f°, %.0f°)" % tuple(math.degrees(x) for x in q), 22),
            pad("%.0f%%" % (frac * 100), 14),
            pad("%.3f m/s" % vmag, 12),
            "投影后这部分动作消失"))

    print()
    print("  这个比例不是小数目：上面四个位形是 11%–34%，而且是**随位形变化的**。")
    print("  换句话说，「同一个末端轨迹」对应的关节运动并不唯一 ——")
    print("  投影只保留下其中一个解，其余全被丢掉。")
    print()
    print("  后果很具体：肘部姿态、避障姿态、腕部绕臂轴的转角，")
    print("  这些「末端看不到但影响成败」的自由度，统一动作空间表达不了。")
    print("  模型只能学到一个**平均的冗余解**，换个场景就可能撞桌子。")

    print("\n  第二个损失：量化。末端空间的动作要离散化才能输出，")
    print("  离散误差回到关节空间会被雅可比放大。放大倍数 = 1/σ_min：")
    print()
    print("  %s%s%s%s" % (pad("q2 (deg)", 12), pad("σ_min", 12),
                        pad("放大 1/σ_min", 14), "1mm 量化 → 关节角误差"))
    print("  " + "-" * 64)

    for deg in (90, 60, 30, 10, 5, 2):
        J = jacobian_2r(0.0, math.radians(deg))
        _hi, lo = sing_vals2(J)
        if lo < 1e-9:
            continue
        err_rad = 0.001 / lo
        print("  %s%s%s%s" % (pad("%d" % deg, 12), pad("%.4f" % lo, 12),
                              pad("%.1f×" % (1.0 / lo), 14),
                              "%.2f°" % math.degrees(err_rad)))

    print()
    print("  同一个 1mm 的动作量化精度：")
    print("    q2=90° → 0.23°；q2=2° → 9.18°，放大 40 倍。")
    print("  也就是说「7 维动作保留 3 位小数」这个决定，")
    print("  在伸直位姿下等价于「关节角只能精确到 0.16 弧度」。")
    print("  这不是理论问题：真机上表现为接近奇异时抖动、抓不准。")


# ---------------------------------------------------------------- [3] 频率

def part3_frequency_mismatch():
    print("\n[3] 控制频率不同：同一个物理速度，动作数值差 10 倍")
    print("-" * 78)
    print("  设机器人末端以 10 cm/s 匀速移动，动作是「每步的位移增量」。")
    print()
    print("  %s%s%s%s" % (pad("数据集频率", 14), pad("每步时长", 12),
                        pad("动作数值", 14), "同一段 1 秒运动切成几步"))
    print("  " + "-" * 64)

    for hz in (3, 5, 10, 15, 30):
        step_mm = 100.0 / hz
        print("  %s%s%s%s" % (pad("%d Hz" % hz, 14), pad("%.0f ms" % (1000 / hz), 12),
                              pad("%.1f mm" % step_mm, 14), "%d 步" % hz))

    print()
    print("  3 Hz 和 30 Hz 之间，动作数值差 10 倍 —— 物理上是同一个运动。")
    print("  如果把这些数据集混在一起、用一套归一化统计量，")
    print("  模型看到的是「数值差 10 倍的同一个任务」。")
    print()
    print("  观测里没有任何东西告诉它当前样本来自几 Hz 的数据集：")
    print("  图像里看不出帧率，状态量纲又都一样。这是一个**不可辨识**的歧义。")

    print("\n  能不能靠重采样补救？不能。上采样不产生信息：")
    print("  线性插值的最大误差 ≈ a·T²/8（a = 加速度，T = 原始采样周期）")
    print()
    print("  %s%s%s%s" % (pad("原始频率", 12), pad("T", 10),
                        pad("插值误差 (a=2m/s²)", 20), "对比该频率的动作步长"))
    print("  " + "-" * 66)
    for hz in (3, 5, 10, 15, 30):
        T = 1.0 / hz
        err_mm = 2.0 * T * T / 8.0 * 1000
        step_mm = 100.0 / hz
        print("  %s%s%s%s" % (pad("%d Hz" % hz, 12), pad("%.0f ms" % (T * 1000), 10),
                              pad("%.1f mm" % err_mm, 20),
                              "%.1f mm（误差 %.0f%%）" % (step_mm, err_mm / step_mm * 100)))

    print()
    print("  3 Hz 的数据插值到 10 Hz，插值误差 27.8mm，")
    print("  而它本身每一步的动作只有 33.3mm —— 误差接近一个动作步长。")
    print("  快动作段（加速度大）会更糟。**这条数据通道已经被采样率限死了**，")
    print("  任何插值都只是在编造中间态。")
    print()
    print("  另一个方向的错觉：降采样总该安全吧？")
    print("  接触冲击的持续时间约 10–20ms。30 Hz 采样周期 33ms ——")
    print("  连 30Hz 都采不到一次完整冲击，10 Hz 更是直接跳过整个事件。")
    print("  所以力/触觉通道的采样率不能跟着图像走（详见 02 章）。")


# ---------------------------------------------------------------- [4] 装不下

def part4_unfittable():
    print("\n[4] 7 维空间装不下的本体")
    print("-" * 78)
    print("  统一的代价不只是丢信息，还有「根本塞不进去」：")
    print()
    rows = [
        ("单臂 6 DoF + 夹爪", 7, "7 维刚好装下，无损"),
        ("单臂 7 DoF + 夹爪", 8, "冗余自由度被丢弃"),
        ("双臂 6+6 + 2 夹爪", 14, "只能压成单臂，另一半信息消失"),
        ("移动底盘 + 7 DoF 臂", 10, "底盘运动无处安放"),
        ("四足 12 关节 + 7 DoF 臂", 20, "腿的接触与步态信息没有维度"),
    ]
    print("  %s%s%s%s" % (pad("本体", 24), pad("原生动作维度", 14),
                          pad("装入 7 维", 12), "后果"))
    print("  " + "-" * 76)
    for name, dim, note in rows:
        if dim <= 7:
            status = "装得下"
        else:
            status = "丢 %d 维" % (dim - 7)
        print("  %s%s%s%s" % (pad(name, 24), pad("%d" % dim, 14),
                              pad(status, 12), note))

    print()
    print("  所以「OXE 有 22 个本体、1M+ 轨迹」，但 RT-X 的实验只用")
    print("  **9 个单臂本体**训练 —— 这是必然的取舍，不是选择。")
    print("  双臂、四足的数据在数据集里，但进不了共享动作空间。")
    print()
    print("  这解释了为什么后来的 VLA 工作（π0、GR00T 等）都在重新设计")
    print("  动作表示：要么按本体分组输出头，要么用「动作 token」把")
    print("  维度问题推到序列长度上。**没有人真正解决了统一，只是换了地方付账。**")


# ---------------------------------------------------------------- 结论

def summary():
    print("\n" + "=" * 78)
    print("结论（数据平台视角）")
    print("=" * 78)
    print("  1. 聚合的隐性成本发生在**转换那一步**：原始 schema 里有的字段，")
    print("     下游 schema 没有就永久丢了。转换前先想清楚要留什么。")
    print("  2. 统一动作空间 = 统一到某个本体的视角。冗余自由度、")
    print("     未使用的维度、装不下的本体，代价都在「表达不了」上。")
    print("  3. 动作数值的物理量纲由控制频率决定。频率不进 schema，")
    print("     动作就是一堆无量纲的数 —— 跨数据集混训必然错。")
    print("  4. 所以「本体标识」和「控制频率」这两个看似无聊的字段，")
    print("     实际是跨本体训练能不能成立的前提。自己的数据集必须存。")


def main():
    print("=" * 78)
    print("实验 1：多本体对齐的代价")
    print("=" * 78)
    part1_schema_matrix()
    part2_projection_loss()
    part3_frequency_mismatch()
    part4_unfittable()
    summary()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验 1：多本体对齐的代价
========================

公开数据集看起来是「一堆 episode」，实际是一堆互不兼容的约定。把 60 个数据集
转成同一种格式之后，还剩下一件事没解决：**动作是什么**。

这个脚本量化六件事，每一件都是「看起来统一了、实际丢了什么」：

  [1] 四种 schema 的字段交集有多小（3 / 13）
  [2] 统一到 7 维末端空间，丢掉的是零空间 —— 11%–34% 的关节运动
  [3] 同一条末端直线，两条合法的关节解差多少（肘角 171°）
  [4] 末端 1 mm 的量化误差，回到关节空间被放大 160 倍
  [5] 控制频率不同，同一个物理速度对应差 10 倍的动作数值
  [6] 七个量纲塞进一个向量，一维吃掉 99.4% 的损失
  [7] 7 维装不下的本体：22 个里只有 9 个能进

全部可精确计算，不依赖任何外部数据。

运行：python examples/04_datasets/01_schema_alignment.py
依赖：无（只用标准库）
"""

import math
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------- 输出对齐

def width(s):
    """终端里一个汉字占两列。不做这个换算，中文表格一定错位。"""
    return sum(2 if ord(c) > 0x2E80 else 1 for c in s)


def pad(s, n):
    return s + " " * max(0, n - width(s))


# ---------------------------------------------------------------- [1] schema

# 三态：✓ 有字段 / △ 只在文档或元数据里，不是机器可读字段 / · 没有
FIELD_ROWS = [
    # (key, 显示名, 缺了会怎样, 是否决定可复用性)
    ("image_exterior", "外视相机", "基本的视觉输入", False),
    ("image_wrist", "腕部相机", "抓取精度，遮挡时唯一可用视角", False),
    ("state_joint", "关节位置", "本体感受，动作对齐的参照", False),
    ("state_ee", "末端位姿", "跨本体可比的状态表示", False),
    ("action", "动作", "监督信号", False),
    ("instruction", "语言指令", "VLA 的语言条件", False),
    ("is_first_last", "首末帧标记", "切 episode，算不清就拼错了", False),
    ("reward_terminal", "reward/terminal", "区分成败，做筛选", False),
    ("intrinsics", "相机内参", "多视角几何、点云重建", False),
    ("extrinsics", "相机外参", "跨本体视觉对齐的前提", True),
    ("embodiment_id", "本体标识", "按本体分组归一化", True),
    ("control_hz", "控制频率", "解释动作的物理量纲", True),
    ("success_flag", "成败标注", "过滤失败样本、算成功率", True),
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
    print("-" * 74)

    names = list(SCHEMAS)
    head = pad("字段", 16) + "".join(pad(n, 13) for n in names) + "缺了会怎样"
    print("  " + head)
    print("  " + "-" * 72)

    all_have, any_have, critical = [], [], []
    for key, label, why, is_critical in FIELD_ROWS:
        row = pad(label, 16)
        marks = []
        for n in names:
            m = SCHEMAS[n][key]
            marks.append(m)
            row += pad(m, 13)
        print("  " + row + why)
        if all(m == "✓" for m in marks):
            all_have.append(label)
        if any(m == "✓" for m in marks):
            any_have.append(label)
        if is_critical:
            holders = [n for n, m in zip(names, marks) if m == "✓"]
            soft = [n for n, m in zip(names, marks) if m == "△"]
            critical.append({"field": label, "hard": holders, "soft": soft,
                             "n_hard": len(holders)})

    print()
    print("  图例：✓ 有字段  △ 只在文档/元数据里  · 没有")
    print()
    print("  四个数据集都有字段的：%d 个 / %d 个 —— %s"
          % (len(all_have), len(FIELD_ROWS), "、".join(all_have)))
    print("  至少一个有的：%d 个" % len(any_have))
    print()
    print("  四个「决定能不能复用」的字段，没有一个被全部做成机器可读字段：")
    for c in critical:
        if not c["hard"]:
            note = "0 个数据集有字段，%s 只写在文档里" % "、".join(c["soft"])
        else:
            note = "%d 个有字段（%s）" % (c["n_hard"], "、".join(c["hard"]))
            if c["soft"]:
                note += "，%s 只在文档里" % "、".join(c["soft"])
        print("    %s%s" % (pad(c["field"], 10), note))
    print()
    print("  而 OXE 的 60 个数据集是被**转换**到统一 schema 的。")
    print("  转换只能保留下游 schema 里有的字段 —— 原数据集里的标定、")
    print("  采集元数据，在转换那一步就永久消失了。")
    print()
    print("  结论：字段交集 3 / %d。「转换」不是搬运，是按目标 schema 剪枝。"
          % len(FIELD_ROWS))

    return {"n_fields": len(FIELD_ROWS), "all_have": all_have,
            "any_have": any_have, "critical": critical,
            "n_common": len(all_have)}


# ---------------------------------------------------------------- 几何工具

LINK = (0.3, 0.3, 0.3)


def fk3(q, l=LINK):
    """平面 3R 臂的正解：返回末端 (x, y)。"""
    q1, q2, q3 = q
    x = l[0] * math.cos(q1) + l[1] * math.cos(q1 + q2) + l[2] * math.cos(q1 + q2 + q3)
    y = l[0] * math.sin(q1) + l[1] * math.sin(q1 + q2) + l[2] * math.sin(q1 + q2 + q3)
    return x, y


def jac3(q, l=LINK):
    """2x3 雅可比：2 维任务空间 + 3 个关节 => 1 维零空间。"""
    q1, q2, q3 = q
    s1, c1 = math.sin(q1), math.cos(q1)
    s12, c12 = math.sin(q1 + q2), math.cos(q1 + q2)
    s123, c123 = math.sin(q1 + q2 + q3), math.cos(q1 + q2 + q3)
    return [[-l[0] * s1 - l[1] * s12 - l[2] * s123,
             -l[1] * s12 - l[2] * s123,
             -l[2] * s123],
            [l[0] * c1 + l[1] * c12 + l[2] * c123,
             l[1] * c12 + l[2] * c123,
             l[2] * c123]]


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


def pinv2x3(J):
    """J⁺ = Jᵀ (J Jᵀ)⁻¹。"""
    a, b, c = J[0]
    d, e, f = J[1]
    a11 = a * a + b * b + c * c
    a12 = a * d + b * e + c * f
    a22 = d * d + e * e + f * f
    det = a11 * a22 - a12 * a12
    if abs(det) < 1e-14:
        return None
    i11, i12, i22 = a22 / det, -a12 / det, a11 / det
    return [[a * i11 + d * i12, a * i12 + d * i22],
            [b * i11 + e * i12, b * i12 + e * i22],
            [c * i11 + f * i12, c * i12 + f * i22]]


def ik3(tx, ty, q1, branch=+1, l=LINK):
    """给定 q1 求 q2, q3。branch=+1 / −1 是两个肘部朝向。"""
    vx, vy = tx - l[0] * math.cos(q1), ty - l[0] * math.sin(q1)
    r = math.hypot(vx, vy)
    if r < 1e-9:
        return None
    C = (r * r + l[1] ** 2 - l[2] ** 2) / (2 * l[1])
    if abs(C) > r:
        return None
    base = math.atan2(vy, vx)
    d = math.acos(max(-1.0, min(1.0, C / r)))
    q12 = base + branch * d
    wx = l[0] * math.cos(q1) + l[1] * math.cos(q12)
    wy = l[0] * math.sin(q1) + l[1] * math.sin(q12)
    return (q1, q12 - q1, math.atan2(ty - wy, tx - wx) - q12)


def trace_path(q0, path):
    """沿末端直线积分 q̇ = J⁺ ẋ（最小关节运动解）。"""
    q = list(q0)
    out = [tuple(q)]
    for i in range(len(path) - 1):
        cur, nxt = path[i], path[i + 1]
        P = pinv2x3(jac3(q))
        dx = (nxt[0] - cur[0], nxt[1] - cur[1])
        q = [q[j] + P[j][0] * dx[0] + P[j][1] * dx[1] for j in range(3)]
        out.append(tuple(q))
    return out


def straight(p0, p1, n=120):
    return [(p0[0] + (p1[0] - p0[0]) * i / n,
             p0[1] + (p1[1] - p0[1]) * i / n) for i in range(n + 1)]


# ---------------------------------------------------------------- [2] 零空间

def part2_nullspace(qdots=None):
    print("\n[2] 统一到 7 维末端空间：丢掉的是「零空间」")
    print("-" * 74)
    print("  OXE 把动作统一成 7 维向量 [x, y, z, roll, pitch, yaw, gripper]，")
    print("  也就是末端位姿的变化，不是关节角。")
    print()
    print("  6 自由度臂 → 6 维位姿：方阵，雅可比可逆，局部无损。")
    print("  7 自由度臂 → 6 维位姿：多出 1 维。那一维在映射里消失了。")
    print()
    print("  用平面 3R 臂（2 维任务 + 3 关节，同样多 1 维）量化：")
    print("  给定关节运动 q̇，有多少比例落在零空间（末端完全不动）？")
    print()

    if qdots is None:
        qdots = (1.0, 1.0, 1.0)
    norm_q = math.sqrt(sum(v * v for v in qdots))

    rows = []
    print("  %s%s%s" % (pad("位形 (q1,q2,q3)", 22), pad("零空间占比", 14), "末端速度"))
    print("  " + "-" * 54)
    for deg in ((0, 60, -30), (30, 90, 0), (-20, 30, 60), (10, 20, 10)):
        q = tuple(math.radians(d) for d in deg)
        J = jac3(q)
        n = null_vector(J)
        if n is None:
            print("  %s退化位形（零空间维数 > 1）" % pad("(%d°, %d°, %d°)" % deg, 22))
            continue
        frac = abs(sum(a * b for a, b in zip(qdots, n))) / norm_q
        v = [sum(J[i][k] * qdots[k] for k in range(3)) for i in range(2)]
        vmag = math.hypot(v[0], v[1])
        rows.append({"q_deg": deg, "null_frac": frac, "v": vmag})
        print("  %s%s%s" % (pad("(%d°, %d°, %d°)" % deg, 22),
                            pad("%.0f%%" % (frac * 100), 14),
                            "%.3f m/s" % vmag))

    lo = min(r["null_frac"] for r in rows)
    hi = max(r["null_frac"] for r in rows)
    print()
    print("  %.0f%% – %.0f%% 的关节运动在末端空间里完全不可见，而且这个比例"
          % (lo * 100, hi * 100))
    print("  **随位形变化**，不是一个常数。肘部姿态、避障姿态、腕部绕臂轴的")
    print("  转角 —— 这些「末端看不出但决定成败」的自由度，7 维动作表达不了。")
    print()
    print("  这还不是最糟的：同一个末端轨迹对应一族关节轨迹，")
    print("  而数据集里只录了其中一条 —— 下一节算这个差别有多大。")

    return {"rows": rows, "lo": lo, "hi": hi}


# ---------------------------------------------------------------- [3] 两条解

def part3_two_solutions(start=(0.35, 0.25), end=(0.12, 0.02), q1_deg=10.0):
    print("\n[3] 同一条末端直线，两条合法的关节解差多少")
    print("-" * 74)
    print("  起点 (%.2f, %.2f) m，终点 (%.2f, %.2f) m，走一条直线，120 步。"
          % (start[0], start[1], end[0], end[1]))
    print("  起点处两个肘部朝向各给一个解，然后各自沿直线积分。")
    print("  两条路径的末端跟踪误差都在 3 mm 以内（一阶积分的残差），")
    print("  可以认为走的是**同一条末端直线**。")
    print()

    path = straight(start, end)
    sols = {}
    for branch, name in ((+1, "肘下"), (-1, "肘上")):
        q0 = ik3(start[0], start[1], math.radians(q1_deg), branch)
        if q0 is None:
            continue
        trace = trace_path(q0, path)
        err = max(math.hypot(fk3(q)[0] - p[0], fk3(q)[1] - p[1])
                  for q, p in zip(trace, path))
        sols[name] = {"q0": q0, "trace": trace, "track_mm": err * 1000.0}

    names = list(sols)
    print("  %s%s%s%s" % (pad("解", 10), pad("起点关节角", 30),
                          pad("末端跟踪误差", 16), "说明"))
    print("  " + "-" * 70)
    for name in names:
        s = sols[name]
        ang = ", ".join("%.0f°" % math.degrees(v) for v in s["q0"])
        print("  %s%s%s%s" % (pad(name, 10), pad("(%s)" % ang, 30),
                              pad("%.1f mm" % s["track_mm"], 16),
                              "同一个末端轨迹的合法解"))

    a, b = sols[names[0]]["trace"], sols[names[1]]["trace"]
    def wrap(d):
        """把角度差折回 (−180°, 180°]：346° 其实等于 −14°。"""
        return (d + 180.0) % 360.0 - 180.0

    diffs = []
    for j in range(3):
        d = max(abs(wrap(math.degrees(x[j] - y[j]))) for x, y in zip(a, b))
        diffs.append(d)
    print()
    print("  两条路径的最大关节差：q1 %.0f°　q2 %.0f°　q3 %.0f°"
          % (diffs[0], diffs[1], diffs[2]))
    print("  末端轨迹一模一样，肘关节差 %.0f°，腕关节差 %.0f°。" % (diffs[1], diffs[2]))
    print()
    print("  7 维动作里，这个差别**一个数都没有**。所以：")
    print("    · 训练时模型学不到「录数据的人选了哪种解」；")
    print("    · 评测时必须自己再解一次 IK，解出来的很可能不是录的那条；")
    print("    · 于是「同一个动作」在训练和部署时是两个不同的物理运动。")
    print()
    print("  这就是「统一动作空间」最容易被忽略的代价：它不是压缩，")
    print("  是**把多值映射压成单值**，而压掉的正是决定成败的那一维。")

    return {"names": names, "diffs": diffs, "sols": {
        k: {"track_mm": v["track_mm"], "q0_deg": tuple(math.degrees(x) for x in v["q0"]),
            "trace_deg": [tuple(math.degrees(x) for x in p) for p in v["trace"]]}
        for k, v in sols.items()}}


# ---------------------------------------------------------------- [4] 量化

def part4_quantization(step_mm=1.0):
    print("\n[4] 末端 %.0f mm 的量化误差，回到关节空间被放大多少" % step_mm)
    print("-" * 74)
    print("  7 维动作要被离散化才能输出（连续回归也有精度上限）。")
    print("  末端空间的误差回到关节空间，放大倍数 = 1 / σ_min：")
    print()

    rows = []
    print("  %s%s%s%s" % (pad("q2 (deg)", 11), pad("σ_min", 11),
                          pad("放大 1/σ_min", 15), "末端 %.0fmm → 关节角误差" % step_mm))
    print("  " + "-" * 62)
    for deg in (90, 60, 30, 10, 5, 2):
        J = jacobian_2r(0.0, math.radians(deg))
        _hi, lo = sing_vals2(J)
        if lo < 1e-9:
            continue
        err_rad = step_mm / 1000.0 / lo
        rows.append({"deg": deg, "sigma_min": lo, "gain": 1.0 / lo,
                     "err_deg": math.degrees(err_rad)})
        print("  %s%s%s%s" % (pad("%d" % deg, 11), pad("%.4f" % lo, 11),
                              pad("%.1f×" % (1.0 / lo), 15),
                              "%.2f°" % math.degrees(err_rad)))

    first, last = rows[0], rows[-1]
    print()
    print("  同一个 %.0f mm 的精度：q2=90° 时是 %.2f°；q2=2° 时是 %.2f°，"
          % (step_mm, first["err_deg"], last["err_deg"]))
    print("  放大 %.0f 倍。" % (last["gain"] / first["gain"]))
    print()
    print("  也就是说「保留 3 位小数」这个决定，在伸直位形下等价于")
    print("  「关节角只能精确到 %.2f 弧度」。真机上表现为接近奇异时抖动、"
          % math.radians(last["err_deg"]))
    print("  抓不准 —— 而这在数据里完全看不出来。")
    print()
    print("  所以精度上限由**最坏位形**决定，不由平均位形决定。")

    return {"rows": rows, "ratio": last["err_deg"] / first["err_deg"]}


# ---------------------------------------------------------------- [5] 频率

def part5_frequency(speed_cm_s=10.0, accel=2.0, freqs=(3, 5, 10, 15, 30)):
    print("\n[5] 控制频率不同：同一个物理速度，动作数值差 %.0f 倍"
          % (max(freqs) / min(freqs)))
    print("-" * 74)
    print("  末端以 %.0f cm/s 匀速移动，动作是「每步的位移增量」。"
          % speed_cm_s)
    print()

    base = []
    print("  %s%s%s%s" % (pad("数据集频率", 13), pad("每步时长", 11),
                          pad("动作数值", 13), "同一段 1 秒运动切成几步"))
    print("  " + "-" * 62)
    for hz in freqs:
        step_mm = speed_cm_s * 10.0 / hz
        base.append({"hz": hz, "step_mm": step_mm})
        print("  %s%s%s%s" % (pad("%d Hz" % hz, 13), pad("%.0f ms" % (1000.0 / hz), 11),
                              pad("%.1f mm" % step_mm, 13), "%d 步" % hz))

    ratio = base[0]["step_mm"] / base[-1]["step_mm"]
    print()
    print("  %d Hz 和 %d Hz 之间动作数值差 %.0f 倍 —— 物理上是同一个运动。"
          % (base[0]["hz"], base[-1]["hz"], ratio))
    print("  观测里没有任何东西告诉模型「这条样本来自几 Hz 的数据集」：")
    print("  图像里看不出帧率，状态量纲又都一样。这是**不可辨识**的歧义。")
    print()
    print("  能不能靠重采样补救？上采样不产生信息。")
    print("  线性插值的最大误差 ≈ a·T²/8（a = %.0f m/s²）：" % accel)
    print()

    interp = []
    print("  %s%s%s%s" % (pad("原始频率", 11), pad("T", 9),
                          pad("插值误差", 12), "占该频率动作步长"))
    print("  " + "-" * 54)
    for b in base:
        T = 1.0 / b["hz"]
        err_mm = accel * T * T / 8.0 * 1000.0
        interp.append({"hz": b["hz"], "err_mm": err_mm,
                       "frac": err_mm / b["step_mm"]})
        print("  %s%s%s%s" % (pad("%d Hz" % b["hz"], 11), pad("%.0f ms" % (T * 1000), 9),
                              pad("%.1f mm" % err_mm, 12),
                              "%.0f%%" % (err_mm / b["step_mm"] * 100)))

    print()
    print("  %d Hz 的数据插值到 %d Hz：误差 %.1f mm，而它本身每一步只有 %.1f mm"
          % (base[0]["hz"], 10, interp[0]["err_mm"], base[0]["step_mm"]))
    print("  —— 误差接近一整个动作步长。**这条通道已经被采样率限死了**，")
    print("  任何插值都只是在编造中间态。")

    return {"base": base, "interp": interp, "ratio": ratio}


# ---------------------------------------------------------------- [6] 量纲

def part6_loss_share(sig_xyz=0.01, sig_rpy=0.02, sig_grip=0.5):
    print("\n[6] 七个量纲塞进一个向量：一维吃掉 %.1f%% 的损失"
          % (sig_grip ** 2 / (3 * sig_xyz ** 2 + 3 * sig_rpy ** 2 + sig_grip ** 2) * 100))
    print("-" * 74)
    print("  7 维动作里装的是三个不同单位的量：米、弧度、开关量。")
    print("  假设各维的动作幅度（标准差）是：")
    print("    xyz %.1f cm　rpy %.2f rad　gripper %.2f（0/1 开关）"
          % (sig_xyz * 100, sig_rpy, sig_grip))
    print()
    print("  一个问题：如果用一个标量 MSE 训练，损失被谁主导？")
    print("  注意全局归一化乘的是**同一个常数**，它不改变各维的占比。")
    print()

    dims = [("x", sig_xyz), ("y", sig_xyz), ("z", sig_xyz),
            ("roll", sig_rpy), ("pitch", sig_rpy), ("yaw", sig_rpy),
            ("gripper", sig_grip)]
    total = sum(s ** 2 for _n, s in dims)
    rows = []
    print("  %s%s%s" % (pad("维度", 12), pad("标准差", 13), "占损失的份额"))
    print("  " + "-" * 44)
    for n, s in dims:
        share = s * s / total
        rows.append({"dim": n, "sigma": s, "share": share})
        print("  %s%s%s" % (pad(n, 12), pad("%.3f" % s, 13), "%.2f%%" % (share * 100)))

    grip = rows[-1]["share"]
    small = sum(r["share"] for r in rows[:6])
    print()
    print("  gripper 一维占 **%.2f%%** 的损失，xyz 三维合计 %.2f%%。"
          % (grip * 100, sum(r["share"] for r in rows[:3]) * 100))
    print()
    print("  这意味着：**一个完全放弃 xyz、只学夹爪的模型，损失只比最优解高 %.2f%%。**"
          % (small * 100))
    print("  也就是说「位置差了 1 cm」这件事，在一个标量损失里几乎看不见 ——")
    print("  训练照样收敛，曲线照样好看，误差全在没人看的维度里。")
    print()
    print("  按维归一化之后，7 维各占 1/7 = %.1f%%。这才是「每一维都算数」。" % (100 / 7))
    print("  所以 schema 的这件事不是预处理细节：**它决定模型能不能看到某一维**。")

    return {"rows": rows, "grip_share": grip, "small_share": small,
            "sigmas": (sig_xyz, sig_rpy, sig_grip)}


# ---------------------------------------------------------------- [7] 装不下

def part7_unfittable(native_dims=(7, 8, 14, 10, 20), pool_dim=7):
    print("\n[7] 7 维空间装不下的本体")
    print("-" * 74)
    print("  统一的代价不只是丢信息，还有「根本塞不进去」：")
    print()

    bodies = [
        ("单臂 6 DoF + 夹爪", 7, "7 维刚好装下，无损"),
        ("单臂 7 DoF + 夹爪", 8, "冗余自由度被丢弃"),
        ("移动底盘 + 7 DoF 臂", 10, "底盘运动无处安放"),
        ("双臂 6+6 + 2 夹爪", 14, "只能压成单臂，另一半信息消失"),
        ("四足 12 关节 + 7 DoF 臂", 20, "腿的接触与步态信息没有维度"),
    ]
    rows = []
    print("  %s%s%s%s" % (pad("本体", 24), pad("原生维度", 12),
                          pad("装入 %d 维" % pool_dim, 13), "后果"))
    print("  " + "-" * 72)
    for name, dim, note in bodies:
        status = "装得下" if dim <= pool_dim else "丢 %d 维" % (dim - pool_dim)
        rows.append({"name": name, "dim": dim, "lost": max(0, dim - pool_dim)})
        print("  %s%s%s%s" % (pad(name, 24), pad("%d" % dim, 12),
                              pad(status, 13), note))

    fit = sum(1 for r in rows if r["lost"] == 0)
    print()
    print("  这解释了 OXE 的「22 个本体、1M+ 轨迹，但 RT-X 只用 9 个单臂本体训练」。")
    print("  剩下的数据在库里，但进不了共享动作空间 —— 22 个里 %.0f%% 能进。"
          % (9 / 22 * 100))
    print()
    print("  后来的 VLA 都在重新设计动作表示：按本体分组输出头，或者把动作")
    print("  离散成 token 把维度问题推到序列长度上。**没有人真正解决了统一，")
    print("  只是换了个地方付账。**")

    return {"rows": rows, "fit": fit, "usable": 9 / 22.0}


# ---------------------------------------------------------------- 结论

def part8_conclusion(p1, p2, p3, p4, p5, p6, p7):
    print("\n" + "=" * 74)
    print("结论（数据平台视角）")
    print("=" * 74)
    print("  1. 字段交集只有 %d / %d。转换只能保留目标 schema 里有的字段，"
          % (p1["n_common"], p1["n_fields"]))
    print("     所以「聚合」是剪枝，不是搬运 —— 转之前先想清楚要留什么。")
    print("  2. 统一动作空间是一次投影，投影有零空间：%.0f%%–%.0f%% 的关节运动"
          % (p2["lo"] * 100, p2["hi"] * 100))
    print("     在末端空间里不存在。同一个末端轨迹，肘角可以差 %.0f°。" % p3["diffs"][1])
    print("  3. 精度上限由最坏位形决定：%.0f mm 末端量化在伸直位形放大 %.0f 倍"
          % (1.0, p4["rows"][-1]["gain"]))
    print("     （%.2f°），是最佳位形的 %.0f 倍。" % (p4["rows"][-1]["err_deg"],
                                              p4["ratio"]))
    print("  4. 动作的量纲由控制频率决定：%d Hz 与 %d Hz 的动作数值差 %.0f 倍，"
          % (p5["base"][0]["hz"], p5["base"][-1]["hz"], p5["ratio"]))
    print("     插值补不回来。频率必须进 schema。")
    print("  5. 七个量纲进一个向量，gripper 一维吃掉 %.1f%% 的标量损失。"
          % (p6["grip_share"] * 100))
    print("     按维归一化不是可选优化，是让每一维都可见的前提。")
    print("  6. %d 维装不下所有本体：22 个里 %.0f%% 能进共享动作空间。"
          % (7, p7["usable"] * 100))
    print("     要多样性就得放弃共享 head —— 这是取舍，不是 bug。")
    print("=" * 74)


def main():
    print("=" * 74)
    print("实验 1：多本体对齐的代价")
    print("=" * 74)
    p1 = part1_schema_matrix()
    p2 = part2_nullspace()
    p3 = part3_two_solutions()
    p4 = part4_quantization()
    p5 = part5_frequency()
    p6 = part6_loss_share()
    p7 = part7_unfittable()
    part8_conclusion(p1, p2, p3, p4, p5, p6, p7)


if __name__ == "__main__":
    main()

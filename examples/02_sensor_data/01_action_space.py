#!/usr/bin/env python3
"""
实验 1：动作向量的三种表示，以及各自的坑
==========================================

同一个"抓杯子放到架子上"的任务，动作可以写成三种东西：

  A. 绝对关节角   a_t = q_t              （机器人此刻该处的姿态）
  B. 关节增量     a_t = q_{t+1} - q_t    （比上一刻多转多少）
  C. 末端位移     a_t = p_{t+1} - p_t    （手往哪走多少）

三者理论上可以互相转换，实际用起来差别巨大。这个脚本量化三件事：

  1. 一个 7 维动作向量里，各维的量级能差多少倍
  2. 归一化时如果不分维度，小量纲的维度还剩多少分辨率
  3. 增量表示在开环执行时为什么会累积误差

运行：
    python examples/02_sensor_data/01_action_space.py

只依赖 Python 标准库。
"""

import math
import random
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

L1 = 1.0   # 大臂 m
L2 = 0.7   # 小臂 m

# 典型 7-DoF 协作臂 + 平行夹爪：一段"抓杯子放到架子上"任务里各维的总运动幅度。
# 前三个关节负责大范围摆位，腕部三关节只做微调，夹爪只有几厘米行程。
DIMS = [
    ("j1 肩部旋转", 1.20, "rad"),
    ("j2 肩部俯仰", 0.85, "rad"),
    ("j3 肘部", 0.60, "rad"),
    ("j4 前臂旋转", 0.30, "rad"),
    ("j5 腕部俯仰", 0.15, "rad"),
    ("j6 腕部旋转", 0.08, "rad"),
    ("grip 夹爪", 0.04, "m"),
]


# ---------------------------------------------------------------- 运动学

def fk(t1, t2):
    return (L1 * math.cos(t1) + L2 * math.cos(t1 + t2),
            L1 * math.sin(t1) + L2 * math.sin(t1 + t2))


def ik(x, y):
    """2 连杆逆解，取肘下那组（theta2 <= 0）。"""
    r2 = x * x + y * y
    c2 = (r2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if c2 < -1.0 or c2 > 1.0:
        return None
    t2 = -math.acos(max(-1.0, min(1.0, c2)))
    t1 = math.atan2(y, x) - math.atan2(L2 * math.sin(t2), L1 + L2 * math.cos(t2))
    return t1, t2


def jacobian(t1, t2):
    """2x2 雅可比：关节角速度 -> 末端线速度。"""
    return [
        [-L1 * math.sin(t1) - L2 * math.sin(t1 + t2), -L2 * math.sin(t1 + t2)],
        [L1 * math.cos(t1) + L2 * math.cos(t1 + t2), L2 * math.cos(t1 + t2)],
    ]


def jac_norm(t1, t2):
    """雅可比的 2-范数（用 Frobenius 近似上界），衡量"关节误差放大到末端多少倍"。"""
    j = jacobian(t1, t2)
    return math.sqrt(sum(v * v for row in j for v in row))


# ---------------------------------------------------------------- 工具

def series_stats(vals):
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    return min(vals), max(vals), mean, math.sqrt(var)


def gen_demo_trajectory(steps=300):
    """
    按 DIMS 里的幅度生成一段合成轨迹，每维用不同周期和相位的正弦。
    不是为了模拟真实轨迹形状，只是让每维都有真实的运动量级。
    """
    out = []
    for i, (_, amp, _) in enumerate(DIMS):
        k = 1 + i * 0.35          # 不同关节错开周期
        ph = i * 0.8
        if i == len(DIMS) - 1:    # 夹爪：0 -> amp -> 0，模拟"张开-夹住-松开"
            s = [amp * (0.5 - 0.5 * math.cos(2 * math.pi * t / steps)) for t in range(steps)]
        else:
            s = [amp * math.sin(2 * math.pi * k * t / steps + ph) for t in range(steps)]
        out.append(s)
    return out


# ---------------------------------------------------------------- 三个部分

def part1_scale_gap():
    print("\n[1] 一个 7 维动作向量里，各维量级差多少")
    print("-" * 74)
    traj = gen_demo_trajectory()
    print(f"  {'维度':<14}{'单位':<6}{'最小值':>9}{'最大值':>9}{'跨度':>9}{'标准差':>9}")
    spans = []
    for i, (name, _, unit) in enumerate(DIMS):
        lo, hi, mean, sd = series_stats(traj[i])
        spans.append(hi - lo)
        print(f"  {name:<14}{unit:<6}{lo:>9.4f}{hi:>9.4f}{hi - lo:>9.4f}{sd:>9.4f}")

    ratio = spans[0] / spans[-1]
    print(f"\n  最大跨度 {spans[0]:.3f} rad  最小跨度 {spans[-1]:.3f} m")
    print(f"  两者相差 {ratio:.0f} 倍 —— 而且单位都不一样（弧度 vs 米）。")
    print("  这不是设计缺陷，是物理现实：肩关节要转一整圈，夹爪只开 4 厘米。")

    # 增量表示下的统计差异
    print("\n  换成「相邻步增量」后，各维的量级又变了一层：")
    print(f"  {'维度':<14}{'增量标准差':>12}{'占原跨度比例':>16}")
    for i, (name, _, _) in enumerate(DIMS):
        d = [traj[i][t + 1] - traj[i][t] for t in range(len(traj[i]) - 1)]
        _, _, _, sd = series_stats(d)
        print(f"  {name:<14}{sd:>12.6f}{sd / spans[i] * 100:>15.2f}%")


def part2_normalization():
    print("\n[2] 归一化：不分维度会损失什么")
    print("-" * 74)
    spans = [2 * amp for _, amp, _ in DIMS]

    # 方案 A：统一按"都是弧度"处理，用 [-1.2, 1.2] 这个统一范围
    uniform = 1.2
    lo_u, hi_u = -uniform, uniform
    step_a = (hi_u - lo_u) / 255.0        # 8-bit 量化台阶

    print(f"  方案 A：所有维度共用一个范围 [{lo_u}, {hi_u}]，8-bit 量化，台阶 {step_a:.5f}")
    print(f"  {'维度':<14}{'真实跨度':>10}{'可用级数':>10}{'等效位宽':>10}{'实际台阶':>14}")
    for i, (name, _, unit) in enumerate(DIMS):
        if i == len(DIMS) - 1:      # 夹爪只在 [0, amp]
            lo_i, hi_i = 0.0, DIMS[i][1]
        else:
            lo_i, hi_i = -DIMS[i][1], DIMS[i][1]
        span = hi_i - lo_i
        levels = span / step_a
        bits = math.log2(levels) if levels > 1 else 0.0
        print(f"  {name:<14}{span:>10.4f}{levels:>10.1f}{bits:>9.1f}bit"
              f"{step_a:>11.5f} {unit}")

    print("\n  夹爪只剩 4 级：相当于开关，抓 3 厘米的杯子和抓 5 厘米的杯子输出一样。")
    print("  这就是 RT-2 一类模型要给每一维单独 min-max + 256 bins 的原因。")

    # 方案 B：每维独立 min-max —— 但也有坑
    print("\n  方案 B：每维独立 min-max（训练集统计量）")
    print("  正常情况每维都能用满 255 级，但依赖两个假设：")
    print("    (a) 该维在训练集里确实动过 —— 否则 std=0，归一化除零")
    print("    (b) 部署时的值不超出训练集范围 —— 否则被裁剪")

    # (a) 零方差维度
    still = [0.08] * 100
    lo, hi, mean, sd = series_stats(still)
    print(f"\n  (a) 某任务里腕部旋转全程没动：span={hi - lo:.6f}  std={sd:.6f}")
    print("      归一化公式 (x - min) / (max - min) 分母为 0 -> ZeroDivisionError，")
    print("      或者加了 eps 后变成恒等于 0 —— 这一维的信息彻底消失。")

    # (b) 超范围裁剪
    train_lo, train_hi = 0.010, 0.035      # 训练集里夹爪只见过这个范围
    deploy = 0.040                          # 部署时要抓更大的杯子
    clipped = max(train_lo, min(train_hi, deploy))
    print(f"\n  (b) 训练集夹爪范围 [{train_lo}, {train_hi}] m，部署时指令 {deploy} m")
    print(f"      裁剪后 {clipped} m，损失 {(deploy - clipped) / deploy * 100:.0f}% 行程")
    print(f"      表现：模型以为张到最大了，实际只张开 "
          f"{(clipped - train_lo) / (0.04 - train_lo) * 100:.0f}%，抓不住。")


def part3_drift(n_trials=4000, seed=20261103):
    print("\n[3] 绝对 vs 增量：开环执行时误差怎么长")
    print("-" * 74)
    print("  场景：策略输出带噪声，机器人开环执行 n 步不看反馈（action chunk 的常态）。")
    print("  每一步的关节噪声 σ = 0.01 rad。")
    print("  绝对表示：a_t = q_t + ε  ->  误差不累积")
    print("  增量表示：a_t = Δq_t + ε ->  误差按随机游走累积，σ ∝ √n")
    print()

    # 参考轨迹：末端从 (1.2, 0.2) 直线走到 (0.6, 1.0)，100 步
    steps = 100
    x0, y0 = 1.2, 0.2
    x1, y1 = 0.6, 1.0
    ref_q = []
    for t in range(steps):
        s = t / (steps - 1)
        x, y = x0 + (x1 - x0) * s, y0 + (y1 - y0) * s
        ref_q.append(ik(x, y))

    noise = 0.01
    rng = random.Random(seed)

    def run(mode, n):
        """开环执行前 n 步，返回 (关节误差, 末端误差) 的期望。"""
        q_err_sum = 0.0
        p_err_sum = 0.0
        for _ in range(n_trials):
            q = list(ref_q[0])
            # 增量模式下先有一次初始定位误差（绝对位置也得知道）
            dq_total = [0.0, 0.0]
            for t in range(n):
                if mode == "absolute":
                    q_try = [ref_q[t][k] + rng.gauss(0, noise) for k in range(2)]
                    dq_total = [q_try[k] - ref_q[t][k] for k in range(2)]
                    q = q_try
                else:
                    dq = [ref_q[t + 1][k] - ref_q[t][k] + rng.gauss(0, noise)
                          for k in range(2)]
                    dq_total = [dq_total[k] + dq[k] - (ref_q[t + 1][k] - ref_q[t][k])
                                for k in range(2)]
                    q = [q[k] + dq[k] for k in range(2)]
            q_err_sum += math.hypot(*dq_total)
            pe = math.hypot(fk(q[0], q[1])[0] - fk(*ref_q[n])[0],
                            fk(q[0], q[1])[1] - fk(*ref_q[n])[1])
            p_err_sum += pe
        return q_err_sum / n_trials, p_err_sum / n_trials

    print(f"  {'执行步数 n':>10}{'绝对·关节':>14}{'增量·关节':>14}"
          f"{'绝对·末端':>12}{'增量·末端':>12}")
    print("  " + "-" * 64)
    rows = {}
    for n in (1, 5, 10, 25, 50):
        qa, pa = run("absolute", n)
        qd, pd = run("delta", n)
        rows[n] = (qa, pa, qd, pd)
        print(f"  {n:>10}{qa:>12.4f}rad{qd:>12.4f}rad"
              f"{pa * 100:>10.2f}cm{pd * 100:>10.2f}cm")

    ratio = rows[50][2] / rows[50][0]
    print(f"\n  n=50 时，增量表示的关节误差是绝对表示的 {ratio:.1f} 倍。")
    print("  原因就是随机游走：每次噪声独立，累积方差 = n·σ²，标准差 ∝ √n。")
    print("  绝对表示每步都被参考位置『拉住』，误差不随步数增长。")
    print("\n  这不是说绝对表示更好 —— 绝对表示要求模型输出精确的全局姿态，")
    print("  而增量表示对本体差异更鲁棒（换个机器人初始位置不同也能用）。")
    print("  真实系统里的选择是：动作头输出增量，但每 k 步用真实本体状态重新锚定。")


def main():
    print("=" * 74)
    print("实验 1：动作向量的三种表示")
    print("=" * 74)
    part1_scale_gap()
    part2_normalization()
    part3_drift()

    print("\n" + "=" * 74)
    print("结论（数据平台视角）：")
    print("  1. action 必须存『每一维的物理含义和单位』，不能只存一个 float 数组")
    print("  2. 归一化统计量是数据集的一部分，要和数据一起版本化")
    print("  3. 数据集要记录 action 是绝对还是增量 —— 同一个数组，语义差了积分")
    print("=" * 74)


if __name__ == "__main__":
    main()

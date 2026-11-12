#!/usr/bin/env python3
"""
实验 1：动作向量的三种表示，以及各自的坑
==========================================

同一个"抓杯子放到架子上"的任务，动作可以写成三种东西：

  A. 绝对关节角   a_t = q_t              （机器人此刻该处的姿态）
  B. 关节增量     a_t = q_{t+1} - q_t    （比上一刻多转多少）
  C. 末端位移     a_t = p_{t+1} - p_t    （手往哪走多少）

三者理论上可以互相转换，实际用起来差别巨大。这个脚本量化六件事：

  1. 一个 7 维动作向量里，各维的量级能差多少倍
  2. 归一化时如果不分维度，小量纲的维度还剩多少分辨率
  3. 增量表示在开环执行时为什么会累积误差
  4. 同一个「移动 1 cm」，四种控制层级下发的数值差三个数量级
  5. chunk 长度为什么由推理延迟和控制频率决定，而不是调参调出来的
  6. 逆解有两支，同一末端位移对应的关节增量完全不同

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
    """2 连杆逆解，取 theta2 <= 0 的那组（此时肘部在肩-腕连线上方）。"""
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


def build_ref(steps=100, x0=1.2, y0=0.2, x1=0.6, y1=1.0):
    """参考轨迹：末端从 (x0, y0) 直线走到 (x1, y1)，返回每步的关节角。"""
    return [ik(x0 + (x1 - x0) * t / (steps - 1),
               y0 + (y1 - y0) * t / (steps - 1)) for t in range(steps)]


def drift_table(n_list=(1, 5, 10, 25, 50), noise=0.01, n_trials=4000, seed=20261103):
    """
    开环执行 n 步后，两种表示各自的关节误差（rad）与末端误差（m）。

    绝对表示：a_t = q_t + ε，每步都被参考位置拉住，误差不随步数增长
    增量表示：a_t = Δq_t + ε，误差按随机游走累积，σ ∝ √n
    """
    ref_q = build_ref(max(100, max(n_list) + 1))
    rng = random.Random(seed)

    def run(mode, n):
        q_err_sum = 0.0
        p_err_sum = 0.0
        for _ in range(n_trials):
            q = list(ref_q[0])
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

    return {n: run("absolute", n) + run("delta", n) for n in n_list}


def part3_drift():
    print("\n[3] 绝对 vs 增量：开环执行时误差怎么长")
    print("-" * 74)
    print("  场景：策略输出带噪声，机器人开环执行 n 步不看反馈（action chunk 的常态）。")
    print("  每一步的关节噪声 σ = 0.01 rad。")
    print("  绝对表示：a_t = q_t + ε  ->  误差不累积")
    print("  增量表示：a_t = Δq_t + ε ->  误差按随机游走累积，σ ∝ √n")
    print()

    print(f"  {'执行步数 n':>10}{'绝对·关节':>14}{'增量·关节':>14}"
          f"{'绝对·末端':>12}{'增量·末端':>12}")
    print("  " + "-" * 64)
    rows = drift_table()
    for n, (qa, pa, qd, pd) in rows.items():
        print(f"  {n:>10}{qa:>12.4f}rad{qd:>12.4f}rad"
              f"{pa * 100:>10.2f}cm{pd * 100:>10.2f}cm")

    ratio = rows[50][2] / rows[50][0]
    print(f"\n  n=50 时，增量表示的关节误差是绝对表示的 {ratio:.1f} 倍。")
    print("  原因就是随机游走：每次噪声独立，累积方差 = n·σ²，标准差 ∝ √n。")
    print("  绝对表示每步都被参考位置『拉住』，误差不随步数增长。")
    print("\n  这不是说绝对表示更好 —— 绝对表示要求模型输出精确的全局姿态，")
    print("  而增量表示对本体差异更鲁棒（换个机器人初始位置不同也能用）。")
    print("  真实系统里的选择是：动作头输出增量，但每 k 步用真实本体状态重新锚定。")


def part4_control_modes():
    print("\n[4] 四种控制层级：同一个“移动 1 cm”，下发的数值差三个数量级")
    print("-" * 74)
    print("  场景：2 连杆臂在 q = (0.60, -0.90) rad，末端沿 x 移动 1 cm。")
    print("  同样一件事，四种控制方式下 action 是完全不同的物理量：")
    print()

    t1, t2 = 0.60, -0.90
    x, y = fk(t1, t2)
    tgt = ik(x + 0.01, y)
    dt = 1.0 / 30.0
    dq1, dq2 = tgt[0] - t1, tgt[1] - t2

    # 重力力矩（点质量模型：大臂 5kg、小臂 3kg、负载 1kg）
    g = 9.81
    m1, m2, mp = 5.0, 3.0, 1.0
    tau1 = (m1 * L1 / 2 + m2 * L1 + mp * L1) * g * math.cos(t1) \
        + (m2 * L2 / 2 + mp * L2) * g * math.cos(t1 + t2)
    tau2 = (m2 * L2 / 2 + mp * L2) * g * math.cos(t1 + t2)

    print(f"  末端从 ({x:.3f}, {y:.3f}) 到 ({x + 0.01:.3f}, {y:.3f})："
          f"关节需要转 Δq = ({dq1:+.4f}, {dq2:+.4f}) rad")
    print()
    rows = [
        ("位置控制", "目标关节角 q*", "rad", f"({tgt[0]:.3f}, {tgt[1]:.3f})", "≈1"),
        ("速度控制", "关节角速度", "rad/s", f"({dq1 / dt:+.3f}, {dq2 / dt:+.3f})", "≈0.3"),
        ("力矩控制", "关节力矩", "N·m", f"({tau1:.1f}, {tau2:.1f})", "≈70"),
        ("阻抗控制", "刚度/阻尼", "N/m, N·s/m", "(500, 20)", "≈500"),
    ]
    for a, b, c, d, e in rows:
        print(f"  {a:<8}{b:<14}{c:<12}{d:<22}量级 {e}")

    print()
    print("  从 0.3 到 500 —— 三个数量级。而且单位完全不同。")
    print("  如果数据集只存了一个 float 数组，这四种情况在数据里长得一模一样。")
    print()
    print("  能不能从数据反推控制模式？看它和 state 的相关性：")
    print()

    # 生成一段 2 连杆轨迹，算各模式下 action 与 q、Δq 的相关系数
    n = 240
    traj = []
    for i in range(n):
        s = i / n
        traj.append((0.6 + 0.4 * math.sin(2 * math.pi * s),
                     -0.9 + 0.3 * math.sin(2 * math.pi * 2 * s)))

    def corr(a, b):
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        va = sum((v - ma) ** 2 for v in a)
        vb = sum((v - mb) ** 2 for v in b)
        if va <= 0 or vb <= 0:
            return 0.0
        cov = sum((a[i] - ma) * (b[i] - mb) for i in range(len(a)))
        return cov / math.sqrt(va * vb)

    q1 = [p[0] for p in traj]
    d1 = [traj[i + 1][0] - traj[i][0] for i in range(n - 1)]
    pos_a = q1[1:]
    vel_a = [d / dt for d in d1]
    trq_a = [(m1 * L1 / 2 + m2 * L1 + mp * L1) * g * math.cos(traj[i][0]) for i in range(n - 1)]
    imp_a = [500.0] * (n - 1)

    print(f"  {'控制层级':<12}{'corr(action, q)':>18}{'corr(action, Δq)':>20}")
    print("  " + "-" * 50)
    for name, arr in (("位置控制", pos_a), ("速度控制", vel_a),
                      ("力矩控制", trq_a), ("阻抗控制", imp_a)):
        print(f"  {name:<10}{corr(arr, q1[1:]):>16.2f}{corr(arr, d1):>18.2f}")

    print()
    print("  位置控制与 q 几乎完全相关（1.00，因为它就是 q 本身）；")
    print("  速度控制与 Δq 完全相关（1.00，它就是 Δq/dt）；")
    print("  力矩控制与 q 也几乎完全相关（-0.99）—— 这段轨迹里 q 在 0.2~1.0 之间，")
    print("  cos(q) 近似线性，所以看上去和位置控制一样，只是符号相反；")
    print("  阻抗控制是常数 —— 方差为 0，相关系数无定义，一点信息都读不出来。")
    print()
    print("  注意三者的 |corr| 都在 1 附近：绝对值分不出谁是谁，只有符号和量级给一点线索，")
    print("  而符号还取决于重力方向、零位定义这些外部约定。")
    print("  所以『从数据反推控制模式』只能猜，不能作为可靠依据。")
    print("  唯一可靠的做法是：采集时就把 control_mode 写进元数据。")


def part5_chunk_vs_latency():
    print("\n[5] chunk 长度不是超参，是推理延迟的函数")
    print("-" * 74)
    print("  控制频率 30 Hz（每 33.3 ms 一个控制周期）。模型推理需要 L 毫秒。")
    print("  如果 chunk=1，每个控制周期都要推理一次 —— 推理跟不上控制，就会丢帧。")
    print()

    ctrl_hz = 30.0
    period = 1000.0 / ctrl_hz
    print(f"  {'推理延迟 L':>12}{'chunk=1 的实际控制频率':>24}{'需要的最小 chunk':>18}")
    print("  " + "-" * 56)
    for lat in (10, 20, 50, 100, 300):
        eff = min(ctrl_hz, 1000.0 / lat)
        need = math.ceil(lat / period)
        print(f"  {lat:>10}ms{eff:>22.1f}Hz{need:>16}")

    print()
    print("  chunk 变长的代价是“反应延迟”：物体在动的时候，这个延迟直接变成跟踪误差。")
    print()
    print(f"  {'chunk 长度':>10}{'重规划周期':>14}{'物体 0.5 m/s 时的滞后':>24}")
    print("  " + "-" * 52)
    for k in (1, 2, 5, 10, 20):
        t = k * period
        print(f"  {k:>10}{t:>12.0f}ms{t / 1000 * 0.5 * 1000:>20.0f}mm")

    print()
    print("  读法：推理要 100 ms 的模型，chunk 至少 3 步才跟得上 30 Hz 控制；")
    print("  而 chunk 一旦超过 6 步，物体以 0.5 m/s 移动时反应滞后就超过 100 mm。")
    print("  两条约束一起看：L <= 200 ms 时可行区间是 [ceil(L/33.3), 6]，非空；")
    print("  L = 300 ms 时最小 chunk 已到 9 步，滞后 150 mm —— 桌面抓取还能用")
    print("  （桌上的物体大多不动，滞后不是问题，所以实践中常见 8~20 步），")
    print("  但传送带分拣这类动态任务就做不了，只能换更快的模型或降低控制频率。")
    print()
    print("  关键结论：chunk 长度由『推理延迟 + 控制频率』决定，不是调参调出来的。")
    print("  所以它必须写进数据 schema —— 数据按 20 步切分，训练时改成 10 步，")
    print("  等于重新切一遍数据，不是改个参数。")


def part6_cross_embodiment():
    print("\n[6] 跨本体：同一末端动作，关节解有无穷多组")
    print("-" * 74)
    print("  同一台 2 连杆臂，末端从 A 点移到 B 点（1 cm）。")
    print("  逆解有两支：theta2 < 0 时肘部在肩-腕连线之上，theta2 > 0 时在之下。")
    print("  两支都能做到，需要的关节增量却完全不同：")
    print()

    t1, t2 = 0.60, -0.90
    x, y = fk(t1, t2)
    x2, y2 = x + 0.01, y

    def ik_up(xx, yy):
        r2 = xx * xx + yy * yy
        c2 = (r2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
        if c2 < -1.0 or c2 > 1.0:
            return None
        tt2 = math.acos(max(-1.0, min(1.0, c2)))
        tt1 = math.atan2(yy, xx) - math.atan2(L2 * math.sin(tt2), L1 + L2 * math.cos(tt2))
        return tt1, tt2

    a_down = ik(x, y)
    a_down2 = ik(x2, y2)
    a_up = ik_up(x, y)
    a_up2 = ik_up(x2, y2)

    print(f"  A 点 ({x:.4f}, {y:.4f}) -> B 点 ({x2:.4f}, {y2:.4f})，末端位移 1.0 cm")
    print()
    print(f"  {'构型':<10}{'起点 q':>20}{'终点 q':>20}{'Δq':>20}")
    print("  " + "-" * 72)
    for name, qa, qb in (("th2<0", a_down, a_down2), ("th2>0", a_up, a_up2)):
        print(f"  {name:<8}({qa[0]:+.3f},{qa[1]:+.3f})   ({qb[0]:+.3f},{qb[1]:+.3f})   "
              f"({qb[0] - qa[0]:+.4f},{qb[1] - qa[1]:+.4f})")

    dd = math.hypot(a_down2[0] - a_down[0], a_down2[1] - a_down[1])
    du = math.hypot(a_up2[0] - a_up[0], a_up2[1] - a_up[1])
    print()
    print(f"  th2<0 的 Δq 模长 {dd:.4f} rad，th2>0 的 {du:.4f} rad —— 差 {abs(du - dd) / dd * 100:.0f}%。")
    print("  而末端做的是同一个 1 cm 的动作。")
    print()
    print("  真实 7-DoF 臂更极端：多出来的自由度构成“零空间”，")
    print("  同一末端位姿对应一个连续的解族 —— 取哪一组取决于初始构型、")
    print("  关节限位、以及上一时刻的姿态（也就是“机器人的习惯”）。")
    print()
    print("  推论：")
    print("  · 关节空间的动作数据永远跨不了本体 —— 维度、零位、方向、习惯全都不同")
    print("  · 只能在末端空间对齐（Open X-Embodiment 的选择），代价是丢掉零空间信息")
    print("  · 或者用一种与本体无关的“相对表达”：末端位移 + 夹爪开度 + 少量上下文")


def main():
    print("=" * 74)
    print("实验 1：动作向量的三种表示")
    print("=" * 74)
    part1_scale_gap()
    part2_normalization()
    part3_drift()
    part4_control_modes()
    part5_chunk_vs_latency()
    part6_cross_embodiment()

    print("\n" + "=" * 74)
    print("结论（数据平台视角）：")
    print("  1. action 必须存『每一维的物理含义和单位』，不能只存一个 float 数组")
    print("  2. 归一化统计量是数据集的一部分，要和数据一起版本化")
    print("  3. 数据集要记录 action 是绝对还是增量 —— 同一个数组，语义差了积分")
    print("  4. 控制层级（位置/速度/力矩/阻抗）也必须在元数据里 —— 数据里看不出来")
    print("  5. chunk 长度由推理延迟和控制频率决定，属于 schema 不属于超参")
    print("=" * 74)


if __name__ == "__main__":
    main()

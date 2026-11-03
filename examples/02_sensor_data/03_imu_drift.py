#!/usr/bin/env python3
"""
实验 3：IMU 为什么不能用，以及编码器精度到底卡在哪
====================================================

IMU 是机器人上最便宜的传感器，也是数据里最难用的一个。这个脚本量化：

  1. 加速度计零偏一次积分变速度误差、二次积分变位置误差
  2. 陀螺仪零偏造成的姿态误差，会通过"重力投影"放大成巨大的加速度误差
  3. 零速检测（ZUPT）为什么能把漂移压下来
  4. 关节编码器的分辨率根本不是瓶颈 —— 机械回差才是

运行：
    python examples/02_sensor_data/03_imu_drift.py

只依赖 Python 标准库。
"""

import math
import random
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

G = 9.81


# ---------------------------------------------------------------- 模拟

def integrate(bias_a=0.0, noise_a=0.0, gyro_err_deg=0.0,
              dt=0.005, t_end=8.0, seed=1, zupt=False, zupt_period=1.0):
    """
    静止状态下的 IMU 积分。真值是"不动"，所以算出来的一切都是误差。

    参数：
        bias_a        加速度计零偏 m/s²（恒定，不随时间变）
        noise_a       白噪声标准差 m/s²
        gyro_err_deg  姿态估计误差（度），通过重力投影变成水平加速度误差
        zupt          是否启用零速更新（同时在线估计零偏）
        zupt_period   多久触发一次静止检测（秒）

    返回 (轨迹 [(t, v_est, p_est)], 零偏估计历史 [(t, b_est)])
    """
    rng = random.Random(seed)
    # 姿态误差让重力在水平轴上的投影不再是 0
    g_leak = G * math.sin(math.radians(gyro_err_deg))

    v = 0.0
    p = 0.0
    b_est = 0.0
    out = []
    hist_b = []
    n = int(t_end / dt)
    next_check = zupt_period

    for i in range(n):
        t = i * dt
        a_meas = bias_a + g_leak + (rng.gauss(0, noise_a) if noise_a > 0 else 0.0)

        # 用当前的零偏估计做补偿后再积分
        a_use = a_meas - b_est
        v += a_use * dt
        p += v * dt

        if zupt and t >= next_check:
            # 已知此刻静止：速度归零，并且"此刻的读数就是零偏"——用它更新估计
            v = 0.0
            b_est = 0.8 * b_est + 0.2 * a_meas
            hist_b.append((t, b_est))
            next_check += zupt_period

        if i % 10 == 0:
            out.append((t, v, p))
    return out, hist_b


# ---------------------------------------------------------------- 各部分

def part1_bias_double_integration():
    print("\n[1] 加速度计零偏：一次积分变速度，二次积分变位置")
    print("-" * 72)
    print("  静止时真实加速度 = 0，但传感器读数 = 零偏 b。")
    print("  于是 v(t) = b·t，p(t) = 0.5·b·t²。误差随时间的**平方**增长。")
    print()
    header = f"  {'零偏 b':>10}" + "".join(f"{t:>10}s" for t in (1, 5, 10, 30))
    print(header)
    print("  " + "-" * 50)
    for b in (0.005, 0.01, 0.05, 0.2):
        row = "".join(f"{0.5 * b * t * t:>10.2f}m" for t in (1, 5, 10, 30))
        print(f"  {b:>8.3f}m/s²{row}")

    print("\n  b = 0.01 m/s² 是消费级 IMU 静止时的典型零偏（好一点的能做到 0.002）。")
    print("  也就是说：**光靠 IMU 积分，10 秒后位置误差就是 0.5 米**。")
    print("  这就是为什么手机导航一进隧道就废 —— 不是算法差，是物理上不可能。")
    print()
    print("  注意 30 秒那一列：0.01 的零偏，30 秒后是 4.5 米。")
    print("  时间翻 3 倍，误差翻 9 倍。所以 IMU 只适合做短时高频的补充，")
    print("  不适合做长时的位置来源。")


def part2_gravity_leak():
    print("\n[2] 陀螺仪零偏的真正杀伤力：重力投影")
    print("-" * 72)
    print("  陀螺零偏看起来温和：0.01°/s 的零偏，60 秒才偏 0.6°。")
    print("  但姿态误差的后果不是『角度错 0.6 度』，而是『重力分错了方向』：")
    print()
    print(f"  {'姿态误差':>10}{'达到它需要':>14}{'重力水平泄漏':>16}{'等效加速度零偏':>18}")
    print("  " + "-" * 60)
    for deg in (0.1, 0.6, 1.0, 2.0, 5.0):
        # 用 0.01°/s 的零偏，到达这个角度需要的时间
        t_reach = deg / 0.01
        leak = G * math.sin(math.radians(deg))
        print(f"  {deg:>9.1f}°{t_reach:>12.0f}s{leak:>14.3f}m/s²{leak / 0.01:>15.0f}x")

    print("\n  1° 的姿态误差，产生 0.171 m/s² 的虚假加速度 ——")
    print("  是 IMU 本身零偏（0.01）的 17 倍。")
    print()
    print("  这就是 IMU 的死循环：")
    print("    陀螺零偏 -> 姿态慢慢偏 -> 重力投影错 -> 加速度计积分出虚假速度")
    print("    -> 位置开始以 t² 飞走 -> 而修正姿态又需要准确的重力方向")
    print()
    print("  出路只有一条：用一个不漂的绝对参考（相机、激光、GPS）来锁住姿态。")
    print("  这就是 VIO / 视觉惯性里程计存在的全部理由。")


def rms_error(t_end, n=200, **kw):
    """多次独立实验的位置误差 RMS（白噪声要靠统计才看得准）。"""
    vals = []
    for s in range(n):
        h, _ = integrate(seed=s, t_end=t_end, **kw)
        vals.append(h[-1][2])
    return math.sqrt(sum(v * v for v in vals) / len(vals))


def part3_noise_and_zupt():
    print("\n[3] 噪声 vs 零偏，以及零速检测能救回多少")
    print("-" * 72)

    print("  场景 A：只有零偏 b=0.01，没有噪声")
    h, _ = integrate(bias_a=0.01, t_end=8.0)
    print(f"    8 秒后位置误差 {h[-1][2]:.3f} m（理论 0.5·0.01·64 = 0.320）")

    print("\n  场景 B：只有白噪声 σ=0.02 m/s²，零偏为 0（200 Hz 采样，200 次实验取 RMS）")
    e8 = rms_error(8.0, noise_a=0.02)
    e30 = rms_error(30.0, noise_a=0.02)
    print(f"    8 秒后位置误差 RMS {e8:.3f} m")
    print(f"    30 秒后位置误差 RMS {e30:.3f} m（时间 ×3.75，误差 ×{e30 / e8:.1f}）")
    print("    白噪声的积分是随机游走：位置误差 ∝ t^1.5。")
    print(f"    理论外推 (30/8)^1.5 = {(30 / 8) ** 1.5:.1f} 倍，实测 {e30 / e8:.1f} 倍，吻合。")
    print("    比零偏的 t² 温和得多，而且多次实验会正负抵消。")
    print("    所以 IMU 选型时噪声密度不是关键指标，零偏稳定性才是。")

    print("\n  场景 C：零偏 + 噪声 + 姿态误差 1°，全上")
    hc, _ = integrate(bias_a=0.01, noise_a=0.02, gyro_err_deg=1.0,
                      t_end=8.0, seed=7)
    print(f"    8 秒后位置误差 {hc[-1][2]:.3f} m  —— 比场景 A 大 {hc[-1][2] / 0.320:.0f} 倍")
    print("    姿态误差是主导项。这也说明为什么标定 IMU 姿态比校准零偏更重要。")

    print("\n  场景 D：同样条件，每秒做一次零速更新（ZUPT）+ 在线估计零偏")
    hz, hb = integrate(bias_a=0.01, noise_a=0.02, gyro_err_deg=1.0,
                       t_end=8.0, seed=7, zupt=True, zupt_period=1.0)
    print(f"    8 秒后位置误差 {hz[-1][2]:.3f} m  —— 比场景 C 小 "
          f"{hc[-1][2] / hz[-1][2]:.0f} 倍")
    print(f"    零偏估计收敛过程：")
    for t, b in hb:
        print(f"      t={t:>4.1f}s   b_est={b:>7.4f} m/s²")

    print(f"\n  注意 b_est 收敛到 {hb[-1][1]:.3f}，而真实零偏只有 0.010。")
    print(f"  它还会继续往 0.181 爬（EMA 系数 0.2，8 次只走了 79%）。")
    print(f"  多出来的约 0.17 正是姿态误差造成的重力泄漏 G·sin(1°) = 0.171。")
    print("  静止时『加速度计零偏』和『姿态误差造成的重力投影』在数学上完全等价，")
    print("  传感器分不出来。所以它把两者一起吸收进了零偏估计 —— 校准看起来成功了，")
    print("  实际上姿态仍然错着，机器人一动起来误差立刻回来。")
    print()
    print("  这是 IMU 最深的那个坑：**它能自洽，但不能自证。**")
    print("  想让姿态真的正确，必须引入一个不依赖积分的绝对参考（相机、激光、磁力计、GPS）。")
    print("  这就是 VIO / 视觉惯性里程计存在的全部理由。")


def part4_encoder_vs_backlash():
    print("\n[4] 关节编码器：分辨率早就过剩了")
    print("-" * 72)
    print("  一个 19-bit 绝对值编码器 + 100:1 谐波减速器：")
    ticks = 2 ** 19
    res_motor = 360.0 / ticks
    res_joint = res_motor / 100
    print(f"    电机端分辨率 = 360° / {ticks} = {res_motor:.6f}° / tick")
    print(f"    关节端分辨率 = {res_motor:.6f}° / 100 = {res_joint:.7f}°")
    print(f"    在 0.5m 臂展处，相当于末端 {0.5 * math.radians(res_joint) * 1e6:.2f} 微米")
    print()
    print(f"  {'部件':>20}{'典型误差(角秒)':>18}{'折合末端(0.5m)':>18}")
    print("  " + "-" * 56)
    rows = [
        ("编码器分辨率", res_joint),
        ("谐波减速器回差", 0.02),
        ("谐波减速器传动误差", 0.05),
        ("连杆加工+装配", 0.10),
        ("整臂重复定位精度", 0.10),
    ]
    for name, deg in rows:
        print(f"  {name:>20}{deg * 3600:>16.2f}"
              f"{0.5 * math.radians(deg) * 1000:>15.3f}mm")

    print(f"\n  编码器比机械回差精确 {0.02 / res_joint:.0f} 倍 —— 三个数量级。")
    print("  也就是说：**你花钱买的高分辨率编码器，精度被减速器回差吃掉了**。")
    print()
    print("  这对数据平台的含义：")
    print("  - 数据集里报『关节精度 0.001°』是没意义的，真实精度由机械决定")
    print("  - 同一个动作正着做和反着做，关节角读数会差一个回差量 ——")
    print("    这在数据里表现为『同样的指令、不同的状态』，是模仿学习里最难处理的噪声之一")
    print("  - 采集时要记录运动方向，事后可以做回差补偿")

    print("\n  还有一个存储选择：存 tick 还是存弧度？")
    print("  - 存 tick：无损，但下游必须知道减速比、零位、方向符号才能用")
    print("  - 存弧度：直接可用，但一旦标定参数变了，历史数据就不可比了")
    print("  主流做法是存弧度 + 把标定文件（减速比/零位/方向/回差）作为数据集元数据一起版本化。")


def main():
    print("=" * 72)
    print("实验 3：IMU 漂移与编码器精度上限")
    print("=" * 72)
    part1_bias_double_integration()
    part2_gravity_leak()
    part3_noise_and_zupt()
    part4_encoder_vs_backlash()

    print("\n" + "=" * 72)
    print("结论（数据平台视角）：")
    print("  1. IMU 的原始数据必须原样保留（含零偏、温度），事后才有补偿的余地")
    print("  2. 姿态估计误差比零偏更致命，所以 IMU 数据要配绝对参考一起存")
    print("  3. 静止段是宝贵的数据 —— 它是唯一能在线估计零偏的窗口，别裁掉")
    print("  4. 关节精度的瓶颈在机械，数据集标注精度时要按真实瓶颈写，别按传感器写")
    print("=" * 72)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
实验 3：IMU 为什么不能用，以及编码器精度到底卡在哪
====================================================

IMU 是机器人上最便宜的传感器，也是数据里最难用的一个。这个脚本量化：

  1. 加速度计零偏一次积分变速度误差、二次积分变位置误差
  2. 陀螺仪零偏造成的姿态误差，会通过"重力投影"放大成巨大的加速度误差
  3. 零速检测（ZUPT）为什么能把漂移压下来 —— 以及它为什么救不了姿态
  4. 关节编码器的分辨率根本不是瓶颈 —— 机械回差才是
  5. 陀螺量程：一次撞击削顶，积分出来的角度直接少一截
  6. 采样率与混叠：100 Hz 采样看到的 60 Hz 振动，真实是 40 Hz
  7. 温度：零偏慢慢爬，比噪声更顽固

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

    print(f"\n  注意 b_est 在 t={hb[-1][0]:.0f}s 时是 {hb[-1][1]:.3f}，最终会爬到 0.181，而真实零偏只有 0.010。")
    print(f"  它还会继续往 0.181 爬（EMA 系数 0.2，7 次只走了 79%）。")
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


def part5_gyro_saturation():
    print("\n[5] 陀螺量程：一次撞击把波形削掉一截，角度就少了一块")
    print("-" * 72)
    print("  场景：机械臂撞到桌子，手腕在 20ms 内被弹开 8°。")
    print("  真实角速度是三角形脉冲，峰值 800°/s；而这只陀螺的量程是 ±300°/s。")
    print()

    dt = 1e-5
    dur = 0.020
    peak = 800.0
    limit = 300.0

    def rate(t):
        if t <= 0 or t >= dur:
            return 0.0
        if t < dur / 2:
            return peak * (t / (dur / 2))
        return peak * (1 - (t - dur / 2) / (dur / 2))

    ang_true = ang_meas = 0.0
    n = int(dur / dt)
    clipped_rows = 0
    for i in range(n):
        t = i * dt
        r = rate(t)
        rc = max(-limit, min(limit, r))
        if r != rc:
            clipped_rows += 1
        ang_true += r * dt
        ang_meas += rc * dt

    print(f"  真实转角      {ang_true:>6.2f}°")
    print(f"  削顶后积分    {ang_meas:>6.2f}°")
    print(f"  丢失          {ang_true - ang_meas:>6.2f}°（{(1 - ang_meas / ang_true) * 100:.0f}%）")
    print(f"  脉冲本身只占 1 秒窗口的 {dur / 1.0 * 100:.0f}%，")
    print(f"  但这 {dur * 1000:.0f}ms 里有 {clipped_rows / n * 100:.0f}% 的采样点顶在量程上。")

    print()
    print("  注意：在整段曲线上，它看起来只是『尖角变钝了』——")
    print("  形状还在，起点终点都对，只有中间那一小段被压平。")
    print("  但积分算的是**面积**，尖顶被压平，面积就少了一大块。")
    print()
    err = ang_true - ang_meas
    leak = G * math.sin(math.radians(err))
    print(f"  后果：姿态永久少转 {err:.2f}°，于是重力水平泄漏 {leak:.3f} m/s²。")
    print(f"  这个假加速度如果进入位置积分：1 秒偏 {0.5 * leak:.2f} 米，10 秒偏 {0.5 * leak * 100:.0f} 米。")
    print("  真机上当然不会真的走那么远（手臂会先撞到别的东西），")
    print("  但这说明 3° 的姿态误差在位置层面已经完全不可用了。")
    print()
    print("  工程做法：")
    print("  - 选量程时按『最坏情况』而不是按『正常运动』：正常运动 50°/s，")
    print("    但撞击瞬间可能是它的 20 倍")
    print("  - 数据里记录量程设置（range），否则事后无法判断有没有削顶")
    print("  - 入库校验：角速度序列里出现连续多个恰好等于 ±量程 的采样点，就是削顶了")


def part6_aliasing():
    print("\n[6] 采样率与混叠：你看到的频率可能不是真的")
    print("-" * 72)
    print("  奈奎斯特：采样率 fs 只能无歧义地表示低于 fs/2 的频率。")
    print("  高于 fs/2 的成分不会消失，它会『折叠』成一个更低的假频率。")
    print()

    fs = 100.0
    print(f"  fs = {fs:.0f} Hz（很多机器人只在控制周期里顺手记一份 IMU）")
    print(f"  可无歧义表示的上限 = fs/2 = {fs / 2:.0f} Hz")
    print()
    print(f"  {'真实频率':<8}{'观测到的频率':<14}说明")
    print("  " + "-" * 52)
    for f in (10, 40, 60, 90):
        alias = abs(f - fs * round(f / fs))
        if f < fs / 2:
            note = "正常，但注意它也可能是折叠来的"
        else:
            note = f"折叠：|{fs:.0f} − {f:.0f}| = {alias:.0f} Hz"
        print(f"  {f:<10.0f}{alias:<16.0f}{note}")

    print()
    print("  看 40 这一行：它本身没问题。问题在于 60 Hz 也会变成 40 Hz ——")
    print("  所以你看到一条 40 Hz 的振动时，无法判断它真实是 40 还是 60。")
    print("  齿轮啮合、电机换向、碰撞冲击，很多都落在 60–90 这个区间，")
    print("  折叠之后正好掉进看起来完全正常的地方。")
    print()
    print("  同一个 60 Hz 振动，换 1 kHz 采样：观测 60 Hz，还原了真相。")
    print("  所以『控制用 100 Hz 就够』和『记录用 100 Hz 就够』是两件事 ——")
    print("  控制关心的是低频跟随，记录要保住高频细节，后者需要高得多的采样率。")


def part7_temperature():
    print("\n[7] 温度：零偏慢慢爬，比噪声更顽固")
    print("-" * 72)
    print("  加速度计零偏的温漂典型值约 0.15 mg/°C = 0.0015 m/s²/°C。")
    print("  开机 25°C，跑 10 分钟后电机把 IMU 烤到 45°C。")
    print()

    tempco = 0.0015
    b0 = 0.010
    print(f"  {'时刻':>10}{'温度':>8}{'零偏':>12}{'相对开机':>10}{'10s 位置误差':>14}")
    print("  " + "-" * 58)
    for minute, temp in ((0, 25), (2, 29), (5, 35), (10, 45)):
        b = b0 + tempco * (temp - 25)
        print(f"  {minute:>8}min{temp:>6}°C{b:>10.4f}m/s²{b / b0:>9.1f}x"
              f"{0.5 * b * 100:>12.1f}m")

    b_hot = b0 + tempco * 20
    print()
    print(f"  十分钟后零偏是开机的 {b_hot / b0:.0f} 倍。它不抖、不跳、不报错，")
    print("  只是稳定地偏大 —— 比噪声难发现得多，因为它在频谱上几乎看不见（频率接近 0）。")
    print()
    print("  这也是为什么『静止段』如此重要：只有在静止段里，你才能把当时的零偏量出来。")
    print("  但注意：静止段量到的是『此刻温度下』的零偏，不是永远有效的常数。")


def main():
    print("=" * 72)
    print("实验 3：IMU 漂移与编码器精度上限")
    print("=" * 72)
    part1_bias_double_integration()
    part2_gravity_leak()
    part3_noise_and_zupt()
    part4_encoder_vs_backlash()
    part5_gyro_saturation()
    part6_aliasing()
    part7_temperature()

    print("\n" + "=" * 72)
    print("结论（数据平台视角）：")
    print("  1. IMU 的原始数据必须原样保留（含零偏、温度、量程），事后才有补偿的余地")
    print("  2. 姿态估计误差比零偏更致命，所以 IMU 数据要配绝对参考一起存")
    print("  3. 静止段是宝贵的数据 —— 它是唯一能在线估计零偏的窗口，别裁掉")
    print("  4. 关节精度的瓶颈在机械，数据集标注精度时要按真实瓶颈写，别按传感器写")
    print("  5. 记录频率要高于『控制的频率』：100 Hz 的控制够用，100 Hz 的记录不够")
    print("=" * 72)


if __name__ == "__main__":
    main()

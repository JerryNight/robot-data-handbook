#!/usr/bin/env python3
"""
实验 4：多传感器时间对齐，误差从哪来
======================================

数据平台最容易被忽略、后果最严重的一件事：把不同频率、不同时钟的传感器
拼成"同一时刻"的一行。

这个脚本量化六个误差源，并给出两种可操作的检查手段：

  1. 采样率差 -> 最近邻对齐的固有误差（半周期）
  2. 插值 vs 最近邻，差多少
  3. 时间误差换算成空间误差（速度 × 时间）
  4. 时间戳取"曝光开始"还是"应用层收到"，差多少
  5. 晶振频率偏差，一小时后累积成多少毫秒
  6. 对齐精度如何限制"能学什么任务"
  7. 四种同步方案的残余误差（差三个数量级）
  8. 埋一个事件 + 互相关，事后测出真实滞后

运行：
    python examples/02_sensor_data/04_time_sync.py

只依赖 Python 标准库。
"""

import bisect
import math
import random
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------- 仿真

def camera_timestamps(t_end, fps=30.0, jitter=0.002, seed=1):
    """
    相机帧时间戳。抖动围绕标称周期波动，不累积
    （真实相机由外部触发或内部定时器驱动，误差不会逐帧累加）。
    """
    rng = random.Random(seed)
    period = 1.0 / fps
    out = []
    k = 0
    while True:
        t = k * period + rng.gauss(0, jitter)
        if t >= t_end:
            return out
        out.append(t)
        k += 1


def joint_timestamps(t_end, rate=100.0):
    """关节数据：由驱动器自己的实时时钟驱动，抖动可忽略。"""
    n = int(t_end * rate)
    return [i / rate for i in range(n)]


def joint_position(t, amp=1.0, period=1.0):
    """关节轨迹：振幅 1 rad、周期 1 s 的正弦。这是"真值"。"""
    return amp * math.sin(2 * math.pi * t / period)


def joint_velocity(t, amp=1.0, period=1.0):
    return amp * 2 * math.pi / period * math.cos(2 * math.pi * t / period)


def nearest_err(src, dst, warmup=1.0, tail=1.0):
    """
    src 里的每个时刻，在 dst 里找最近的点，返回误差（秒）。
    跳过首尾各 warmup/tail 秒，避免边界处"只能往前找"造成的伪大误差。
    """
    out = []
    n = len(dst)
    if n == 0:
        return out
    t_end = dst[-1]
    for t in src:
        if t < warmup or t > t_end - tail:
            continue
        i = bisect.bisect_left(dst, t)
        cands = [dst[j] for j in (i - 1, i) if 0 <= j < n]
        best = min(cands, key=lambda x: abs(x - t))
        out.append(t - best)
    return out


# ---------------------------------------------------------------- 各部分

def part1_nearest_neighbor():
    print("\n[1] 最近邻对齐：误差上限由『被取值的那个信号』决定")
    print("-" * 74)
    t_end = 10.0
    rate = 100.0      # 关节采样率 <- 改这里看方向 A 的上限怎么变
    cam = camera_timestamps(t_end)
    jnt = joint_timestamps(t_end, rate)
    print(f"  相机 {len(cam)} 帧 @ 30Hz（半周期 {1000 / 30 / 2:.2f} ms）")
    print(f"  关节 {len(jnt)} 帧 @ {rate:.0f}Hz（半周期 {1000 / rate / 2:.2f} ms）")
    print()

    e_a = nearest_err(cam, jnt)
    e_b = nearest_err(jnt, cam)

    def show(name, errs):
        mean_abs = sum(abs(x) for x in errs) / len(errs)
        mx = max(abs(x) for x in errs)
        print(f"  {name}")
        print(f"    误差范围 [{min(errs) * 1000:+.2f}, {max(errs) * 1000:+.2f}] ms"
              f"   平均 {mean_abs * 1000:.2f} ms   最大 {mx * 1000:.2f} ms")

    show("方向 A：每帧图像 -> 找最近的关节采样（关节是稀疏的一方）", e_a)
    show("方向 B：每个关节采样 -> 找最近的图像（图像是稀疏的一方）", e_b)

    print("\n  关键在于：误差上限不是看『两个信号里谁更稀疏』，而是看")
    print("  『你去取值的那一方有多稀疏』。")
    print(f"  方向 A 取的是关节值，所以受关节 {rate:.0f}Hz 限制（±{1000 / rate / 2:.1f}ms）；")
    print("  方向 B 取的是图像，所以受相机 30Hz 限制（±16.7ms）。")
    print()
    print("  这就是为什么对齐只有一个正确方向：**把高频信号搬到低频时刻，")
    print("  永远不要反过来**。反过来意味着你在用 30Hz 的信息去猜 100Hz 的时刻。")

    print("\n  相机自身抖动（围绕标称周期波动，不累积）会进一步撑大方向 B：")
    for jit in (0.0, 0.002, 0.005):
        c = camera_timestamps(t_end, jitter=jit)
        eb = nearest_err(jnt, c)
        ea = nearest_err(c, jnt)
        print(f"    抖动 σ={jit * 1000:.0f}ms -> 方向 A 最大 "
              f"{max(abs(x) for x in ea) * 1000:5.2f} ms   "
              f"方向 B 最大 {max(abs(x) for x in eb) * 1000:5.2f} ms")


def part2_interpolation():
    print("\n[2] 插值：把低频信号搬到高频时刻")
    print("-" * 74)
    print("  正确姿势：不是把图像搬到关节时刻，而是把关节角插值到图像时刻。")
    print("  因为关节是平滑的标量信号，可以插值；图像是离散的，插值没有物理意义。")
    print()

    t_end = 4.0
    cam = camera_timestamps(t_end, seed=3)
    jnt = joint_timestamps(t_end)

    print(f"  {'对齐方式':>16}{'平均角度误差':>16}{'最大角度误差':>16}{'末端位置误差*':>18}")
    print("  " + "-" * 66)

    # 最近邻
    e_nn = []
    for t in cam:
        if t < 1.0 or t > jnt[-1] - 1.0:
            continue
        j = max(0, min(len(jnt) - 1, round(t * 100)))
        e_nn.append(joint_position(t) - joint_position(jnt[j]))

    # 线性插值
    e_lin = []
    for t in cam:
        if t < 1.0 or t > jnt[-1] - 1.0:
            continue
        f = t * 100
        i = int(f)
        i = max(0, min(len(jnt) - 2, i))
        frac = f - i
        q = joint_position(jnt[i]) * (1 - frac) + joint_position(jnt[i + 1]) * frac
        e_lin.append(joint_position(t) - q)

    arm = 0.5        # 假设 0.5m 臂展，1 rad 关节误差 -> 0.5m 末端误差
    for name, e in (("最近邻", e_nn), ("线性插值", e_lin)):
        mean_abs = sum(abs(x) for x in e) / len(e)
        mx = max(abs(x) for x in e)
        print(f"  {name:>14}{mean_abs * 1000:>13.2f}mrad{mx * 1000:>13.2f}mrad"
              f"{mean_abs * arm * 1000:>15.2f}mm")

    ratio = (sum(abs(x) for x in e_nn) / len(e_nn)) / (sum(abs(x) for x in e_lin) / len(e_lin))
    print(f"\n  插值比最近邻精确 {ratio:.0f} 倍。")
    print("  注意这里的关节轨迹是 1 Hz 正弦 —— 运动越慢、越平滑，插值越准。")
    print("  如果关节在做一个 10 Hz 的抖动（接触碰撞、步态冲击），插值也会失真。")
    print("  * 末端误差按 0.5m 臂展、小角度近似折算，只看量级。")


def part3_spatial_equivalent():
    print("\n[3] 时间误差换算成空间误差")
    print("-" * 74)
    print("  时间对不齐，等价于空间对不齐。换算系数就是末端速度。")
    print()
    print(f"  {'末端速度':>10}{'1ms':>9}{'5ms':>9}{'8.3ms':>10}{'16.7ms':>10}{'33ms':>9}")
    print("  " + "-" * 60)
    for v in (0.1, 0.5, 1.0, 2.0, 4.0):
        row = "".join(f"{v * dt * 1000:>8.1f}mm" for dt in (0.001, 0.005, 0.0083, 0.0167, 0.033))
        print(f"  {v:>8.1f}m/s{row}")

    print("\n  读法：末端 1 m/s 时，8.3ms 的对齐误差 = 8.3mm。")
    print("  而杯子直径 70mm —— 这已经是杯口半径的四分之一。")
    print()
    print("  更要命的是**方向性**：如果所有观测都系统性滞后 8ms，")
    print("  模型会学到『动作要提前 8ms 开始』。训练 loss 正常，")
    print("  真机上表现为过冲或碰撞。只有换一个采样率不同的平台才会暴露出来。")


def part4_timestamp_source():
    print("\n[4] 时间戳取哪一刻？这是数据平台最常犯的错")
    print("-" * 74)
    print("  一帧图像的时间戳可以取至少四个时刻：")
    print("    t0 曝光开始（快门打开）   <- 物理上最有意义")
    print("    t1 曝光结束（快门关闭）   <- 比 t0 晚一个曝光时间")
    print("    t2 数据传输完成（USB 收到）")
    print("    t3 驱动交给应用层")
    print()

    exposure = 0.010     # 曝光 10ms
    transfer = 0.004     # USB 传输 4ms
    driver = 0.002       # 驱动队列 2ms
    print(f"  典型值：曝光 {exposure * 1000:.0f}ms，传输 {transfer * 1000:.0f}ms，"
          f"驱动缓冲 {driver * 1000:.0f}ms")
    print()
    print(f"  {'取哪个时刻':>22}{'相对 t0 的滞后':>18}{'抖动':>10}")
    print("  " + "-" * 52)
    rows = [
        ("t0 曝光开始", 0.0, "低"),
        ("t1 曝光结束", exposure, "低"),
        ("t2 传输完成", exposure + transfer, "中"),
        ("t3 驱动出队", exposure + transfer + driver, "高"),
    ]
    for name, lag, jit in rows:
        print(f"  {name:>20}{lag * 1000:>15.0f}ms{jit:>9}")

    print(f"\n  如果图省事用 t3（『收到数据就打个时间戳』），")
    print(f"  所有观测会系统性滞后 {(exposure + transfer + driver) * 1000:.0f}ms，而且抖动最大。")
    print()
    print("  为什么这比『随机误差』可怕：")
    print("  - 随机误差在训练时会被平均掉，系统性滞后会被学进去")
    print("  - 它不会报错，不会让 loss 变差，只是在真机上表现成「总是慢半拍」")
    print("  - 换一台机器（USB 更快、曝光更短）后滞后变了，模型立刻失效")
    print()
    print("  工程做法：")
    print("  - 相机用硬件触发，时间戳由触发信号生成，而不是软件打点")
    print("  - 所有传感器接入同一个时钟源（PTP / PPS / 触发线）")
    print("  - 时间戳在采集端就写好，随数据一起落盘，不依赖下游补")


def part5_clock_drift():
    print("\n[5] 晶振漂移：一小时后差多少")
    print("-" * 74)
    print("  每台设备用自己的晶振计时。标称 30.000Hz 的相机，实际可能是 30.00003Hz。")
    print("  这个 ppm 级的偏差，会随着时间线性累积成真实的毫秒误差。")
    print()
    print(f"  {'晶振偏差':>10}{'1 分钟':>12}{'10 分钟':>12}{'1 小时':>12}{'8 小时':>12}")
    print("  " + "-" * 58)
    for ppm in (1, 10, 50, 100):
        row = "".join(f"{t * ppm * 1e-6 * 1000:>10.1f}ms" for t in (60, 600, 3600, 28800))
        print(f"  {ppm:>8}ppm{row}")

    print("\n  消费级晶振常见 20–100 ppm。100 ppm 的相机，录 8 小时后")
    print("  和主机时钟差了 2.9 秒 —— 整段数据的时间轴全错位。")
    print()
    print("  这就是为什么长时采集必须做时钟同步，而不是『反正都记下来』：")
    print("  - 每段录制开始打一次同步信号（哪怕只是闪光灯闪一下）")
    print("  - 或者用 PTP 让所有设备跑在同一个时钟上")
    print("  - 事后校验：检查两路数据的帧数比是否等于标称频率比")


def part6_what_can_you_learn():
    print("\n[6] 对齐精度决定了你能学什么")
    print("-" * 74)
    print(f"  {'对齐误差':>12}{'末端 1m/s 时的空间误差':>24}{'能支持的任务':>30}")
    print("  " + "-" * 68)
    rows = [
        (0.001, "力控装配、精密插孔、高速抛接"),
        (0.005, "常规抓取操作（主流 VLA 的区间）"),
        (0.020, "慢速搬运、开门、按按钮"),
        (0.050, "只能学任务顺序，学不到精细动作"),
        (0.150, "学到的策略在真机上会「慢半拍」"),
    ]
    for dt, task in rows:
        print(f"  {dt * 1000:>9.1f}ms{dt * 1.0 * 1000:>21.1f}mm{task:>30}")

    print("\n  这是一条硬约束：**数据的对齐精度，就是模型能力的上限。**")
    print("  模型再大也补不回来 —— 因为信息在采集时就已经错位了。")
    print()
    print("  这也解释了一个常见现象：同一个开源模型，别人复现效果好，你复现效果差。")
    print("  差距往往不在训练配置，在你花在时间戳上的功夫。")


def part7_sync_methods():
    print("\n[7] 四种同步方案：误差差三个数量级")
    print("-" * 74)
    print("  『时间戳准不准』不是一个是/否问题，而是看你在哪一层打的点：")
    print()
    rows = [
        ("驱动里软件打点（收到即记）", "5–20 ms", "免费"),
        ("主机收到数据时打点", "2–8 ms", "免费"),
        ("触发线 + 硬件打点", "10–100 µs", "几百元"),
        ("PTP / gPTP 全网同钟", "1–10 µs", "几千元"),
    ]
    for a, b, c in rows:
        print(f"  {a:<14}{b:>10}{c:>10}")

    print()
    print("  注意最后两行的量级：微秒。这不是『好一点』，是换了一个数量级。")
    print("  前面两行的毫秒级误差，正好落在第 6 节那张表最糟糕的区间里。")
    print()
    print("  为什么软件打点救不回来：")
    print("  - 从『传感器产生数据』到『你的代码拿到数据』之间，有内核缓冲、")
    print("    驱动队列、调度延迟，三者的延迟都随负载波动，不是常数。")
    print("  - 你能测的是『我收到它的时刻』，不是『它发生的时刻』。")
    print("  - 补偿一个常数延迟是可以的（标定一次），但补偿抖动不行 ——")
    print("    抖动是随机的，你没法从数据本身知道这一帧到底晚了几毫秒。")
    print()
    print("  所以唯一可靠的做法是：让时间戳由硬件在『事件发生的那一刻』生成。")
    print("  相机用触发信号（strobe out）记曝光开始，雷达用 PPS 对齐角度起始，")
    print("  力传感器用同步采样时钟 —— 三者接同一个源，误差就只剩传播延迟。")


def part8_cross_correlation():
    print("\n[8] 事后怎么测出滞后：埋一个事件，然后做互相关")
    print("-" * 74)
    print("  前面说『系统性滞后你无法从数据本身发现』—— 这句话有个前提：")
    print("  数据里没有可用于对齐的共同事件。如果采的时候埋了，事后就能测。")
    print()

    fs = 1000.0          # Hz
    dur = 2.0
    n = int(fs * dur)
    delay_ms = 8.0       # 真实滞后
    rng = random.Random(3)

    # 事件：t=0.5s 开始的一段持续接触（阶跃），叠加噪声
    def signal(shift_ms):
        out = []
        for i in range(n):
            t = i / fs - shift_ms / 1000.0
            v = 1.0 if t >= 0.5 else 0.0
            out.append(v + rng.gauss(0, 0.04))
        return out

    a = signal(0.0)
    b = signal(delay_ms)     # 真实滞后 8 ms
    print(f"  两路信号：1 kHz 采 2 秒，t=0.5 s 处一个阶跃事件，σ=0.04 的噪声。")
    print(f"  真实滞后：{delay_ms:.1f} ms（B 比 A 晚）")
    print()

    # 归一化互相关
    def corr(lag):
        s = 0.0
        cnt = 0
        for i in range(max(0, -lag), min(n, n - lag)):
            s += (a[i] - 0.5) * (b[i + lag] - 0.5)
            cnt += 1
        return s / cnt if cnt else 0.0

    best_lag, best_val = 0, -1e18
    lags = range(-40, 41)
    for lag in lags:
        v = corr(lag)
        if v > best_val:
            best_val, best_lag = v, lag
    print(f"  互相关峰在 lag = {best_lag} 个采样点 = {best_lag / fs * 1000:.1f} ms")
    print(f"  分辨率 = 1 个采样 = {1000 / fs:.1f} ms")
    print()
    print("  三件事值得注意：")
    print(f"  1. 结果精度受采样率限制：1 kHz 只能给到 1 ms 分辨率。")
    print("     要更细，要么提高采样率，要么在峰附近做抛物线插值（能到 0.1 ms）。")
    print("  2. 事件必须『两路都看得见』。相机和关节没有共同的物理事件，")
    print("     所以工程上常埋一个可见/可测的同步信号：闪一下 LED、敲一下桌面。")
    print("  3. 每一段录制都要重新测。延迟会随温度、负载、USB 拓扑变。")
    print()
    print("  这就是数据平台该做的事：把『同步质量自检』做成采集流程的一部分，")
    print("  每次录制自动跑一遍互相关，把测出来的偏移写进这段数据的元数据。")


def main():
    print("=" * 74)
    print("实验 4：多传感器时间对齐")
    print("=" * 74)
    part1_nearest_neighbor()
    part2_interpolation()
    part3_spatial_equivalent()
    part4_timestamp_source()
    part5_clock_drift()
    part6_what_can_you_learn()
    part7_sync_methods()
    part8_cross_correlation()

    print("\n" + "=" * 74)
    print("结论（数据平台视角）：")
    print("  1. 对齐误差下限 = 被取值信号的半周期，靠算法补不回来")
    print("  2. 两路信号合成时，必须组织在低频信号的时间网格上，把高频信号插值过去")
    print("  3. 时间戳必须在采集端硬件打点，软件打点会引入系统性滞后")
    print("  4. 长时采集要做时钟同步，否则 ppm 级偏差会累积成秒级错位")
    print("  5. 数据集应该公布自己的时间对齐精度 —— 它比数据量更能预测可用性")
    print("  6. 每段数据都该自带一个同步质量指标，而不是靠用户去猜")
    print("=" * 74)


if __name__ == "__main__":
    main()

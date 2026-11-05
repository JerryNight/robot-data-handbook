#!/usr/bin/env python3
"""
实验 4：控制模式与数据形态
===========================

同一个机器人、同一个目标位置，用不同的控制方式，产生的数据完全不一样。
这个脚本对比三种控制，并显示它们在数据上的特征差异：

  1. P / PD / PID 位置控制 -> 超调、振荡、稳态误差
  2. 阻抗控制（impedance）-> 遇外力时主动让步
  3. 两者的数据曲线差异 -> 判断一批数据是哪种控制器产生的
  4. 积分的副作用：卡住两秒，松手后冲过头（windup）
  5. 控制频率 vs 采样频率 -> 记录下来的轨迹为什么会变成折线

为什么要关心这个：数据里看到轨迹有超调和振荡，不一定是采集质量差，
可能是控制器参数问题；看到遇阻就停，可能是阻抗控制而不是失败。

运行：
    python examples/01_robotics_primer/04_pid_and_impedance.py

只依赖 Python 标准库。
"""

import math
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------- 仿真

def simulate(kp, ki, kd, m=1.0, b=0.4, f_c=0.6, target=1.0, t_end=4.0, dt=0.0005,
             ext_force=0.0, ext_start=99.0, record_every=40,
             f_max=None, clamp_integ=False, block_x=None, block_until=0.0):
    """
    一维质量-阻尼系统，受 PID 控制力 + 可选外力。

        m*x'' = F_pid + F_ext - 粘性摩擦(b*v) - 库仑摩擦(f_c)

    为什么必须有库仑摩擦项：只有粘性摩擦 b*v 时，机器人一旦静止，
    摩擦就是 0，PD 控制能把稳态误差压到 0 —— 这与真实机器人不符。
    真实系统里造成稳态误差的是**静摩擦**：速度为零时它依然存在，
    大小是 f_c，方向与驱动力相反。

    这里用粘滑（stick-slip）模型：
      - 速度接近 0 且驱动力不足以克服 f_c -> 锁死不动
      - 否则按速度方向施加 f_c 的动摩擦

    可选实验参数（默认都不启用，行为与之前完全一致）：
      record_every  每多少个控制周期记录一个采样点（默认 40，即 500Hz 控制 / 50Hz 记录）
      f_max         执行器力上限（None = 不限）
      clamp_integ   True 时，力饱和期间冻结积分（抗饱和 / conditional integration）
      block_x       一个虚拟障碍：t < block_until 时位置不能超过 block_x（模拟撞墙/堵转）
      block_until   障碍消失的时刻

    返回 [(t, x, F_pid)] 采样序列。
    """
    x, v = 0.0, 0.0
    integ = 0.0
    prev_e = target - x
    hist = []
    steps = int(t_end / dt)

    for step in range(steps):
        t = step * dt
        e = target - x
        integ += e * dt
        de = (e - prev_e) / dt

        f_un = kp * e + ki * integ + kd * de
        f_pid = f_un
        if f_max is not None and abs(f_un) > f_max:
            f_pid = math.copysign(f_max, f_un)
            if clamp_integ:
                integ -= e * dt          # 饱和了就别再记账，否则账本会越过越高

        f_ext = ext_force if t >= ext_start else 0.0
        drive = f_pid + f_ext

        if abs(v) < 1e-4 and abs(drive) <= f_c:
            v = 0.0                      # 静摩擦锁死：推不动
        else:
            if abs(v) > 1e-12:
                direction = 1.0 if v > 0 else -1.0
            else:
                direction = 1.0 if drive > 0 else -1.0
            fric = b * v + f_c * direction
            a = (drive - fric) / m
            v += a * dt
            x += v * dt

        if block_x is not None and t < block_until and x > block_x:
            x = block_x                  # 撞上了，过不去
            v = 0.0

        prev_e = e

        if step % record_every == 0:     # 降采样到 50 Hz，对应约 500 Hz 控制
            hist.append((t, x, f_pid))

    return hist


def impedance(k, b_imp, m=1.0, target=1.0, ext_force=0.0, ext_start=99.0,
              t_end=4.0, dt=0.0005):
    """
    阻抗控制：不追求精确到达目标，而是让"偏离目标的程度"和"外力"成正比。

        m*x'' + b*x' + k*(x - target) = F_ext

    稳态偏移 = F_ext / k。k 越小越"软"（让步越多）。
    """
    x, v = 0.0, 0.0
    hist = []
    for step in range(int(t_end / dt)):
        t = step * dt
        f_ext = ext_force if t >= ext_start else 0.0
        a = (f_ext - b_imp * v - k * (x - target)) / m
        v += a * dt
        x += v * dt
        if step % 40 == 0:
            hist.append((t, x, 0.0))
    return hist


# ---------------------------------------------------------------- 文本绘图

def plot_series(series, width=56, height=13, y_min=None, y_max=None):
    """
    series: [(label, [(t, y), ...]), ...]

    每一列取该列所有采样点的均值，避免逐点覆盖导致曲线形状失真
    （采样点数远多于字符列数时，直接覆盖画不出真实包络）。
    """
    all_y = [y for _, pts in series for _, y in pts]
    lo = min(all_y) if y_min is None else y_min
    hi = max(all_y) if y_max is None else y_max
    if hi - lo < 1e-9:
        hi = lo + 1.0

    chars = ["#", "*", ".", "+"]
    grid = [[" "] * width for _ in range(height)]

    for idx, (label, pts) in enumerate(series):
        ch = chars[idx % len(chars)]
        t_max = pts[-1][0] if pts[-1][0] > 0 else 1.0

        # 按列聚合
        buckets = {}
        for (t, y) in pts:
            col = int(t / t_max * (width - 1))
            col = max(0, min(width - 1, col))
            buckets.setdefault(col, []).append(y)

        for col, ys in buckets.items():
            y = sum(ys) / len(ys)
            row = int((hi - y) / (hi - lo) * (height - 1))
            row = max(0, min(height - 1, row))
            grid[row][col] = ch

    for r, row in enumerate(grid):
        val = hi - (hi - lo) * r / (height - 1)
        print(f"  {val:>7.3f} |" + "".join(row))
    print("  " + " " * 7 + " +" + "-" * width)
    print("  " + " " * 8 + "t=0" + " " * (width - 12) + "t=end")
    legend = "  ".join(f"{chars[i % 4]} {s[0]}" for i, s in enumerate(series))
    print(f"  图例：{legend}")


def metrics(hist, target):
    """
    从时间序列算超调、稳态误差、调节时间。

    如果最后 10% 的时段内仍在振荡（极差超过目标值的 2%），
    说明系统还没稳定，稳态误差与调节时间都无意义 -> 返回 None。
    """
    xs = [x for _, x, _ in hist]
    peak = max(xs)
    overshoot = max(0.0, (peak - target) / target * 100)

    tail = xs[-max(1, len(xs) // 10):]
    settled = (max(tail) - min(tail)) < 0.02 * target

    if not settled:
        return overshoot, None, None

    steady = sum(tail) / len(tail)
    ss_err = abs(target - steady)

    band = 0.02 * target
    settle = 0.0
    for i in range(len(hist) - 1, -1, -1):
        if abs(hist[i][1] - target) > band:
            settle = hist[i + 1][0] if i + 1 < len(hist) else None
            break
    return overshoot, ss_err, settle


def show_metrics(tag, hist, target):
    ov, ss, st = metrics(hist, target)
    ss_txt = f"{ss * 100:.2f} cm" if ss is not None else "未稳定"
    st_txt = f"{st:.2f}s" if st is not None else "未收敛"
    print(f"  {tag} 超调 {ov:>5.1f}%   稳态误差 {ss_txt:>10}   调节时间 {st_txt}")


def chord_error(truth, sparse):
    """
    把稀疏采样点用直线连起来，和真实（500 Hz）曲线比，最大差多少。

    这正是"回放一条低采样率轨迹"时发生的事：控制器按折线走，
    而当时机器人走的是曲线。返回 (最大偏差 m, 出现时刻 s)。
    """
    worst, t_worst = 0.0, 0.0
    j = 0
    for t, x, _ in truth:
        while j + 2 < len(sparse) and sparse[j + 1][0] <= t:
            j += 1
        if j + 1 >= len(sparse):
            break
        t0, x0 = sparse[j][0], sparse[j][1]
        t1, x1 = sparse[j + 1][0], sparse[j + 1][1]
        if t1 <= t0:
            continue
        x_line = x0 + (x1 - x0) * (t - t0) / (t1 - t0)
        d = abs(x - x_line)
        if d > worst:
            worst, t_worst = d, t
    return worst, t_worst


def main():
    target = 1.0
    print("=" * 76)
    print("实验 4：控制模式与数据形态")
    print(f"目标位置 x = {target}，系统 m=1.0, b=0.4（摩擦）")
    print("=" * 76)

    # -------- 1. 只有 P 控制
    print("\n[1] 纯 P 控制：Kp = 20")
    h_p = simulate(kp=20, ki=0, kd=0, t_end=8.0)
    plot_series([("P", [(t, x) for t, x, _ in h_p])], y_min=-0.1, y_max=1.9)
    ov, ss, st = metrics(h_p, target)
    show_metrics("纯 P  ", h_p, target)
    print("  纯 P 有稳态误差：静摩擦占了便宜，必须靠一个恒定的位置误差")
    print("  产生的控制力来抗衡它。误差大小约等于 f_c / Kp = " f"{0.6 / 20 * 100:.1f} cm。")
    print("  注意它还在振荡、8 秒内没稳下来 —— Kp 小又没有微分项，")
    print("  系统严重欠阻尼。这是纯 P 的第二个问题：容易振。")

    # -------- 2. PD 控制
    print("\n[2] PD 控制：Kp = 60, Kd = 18（加大刚度 + 加阻尼）")
    h_pd = simulate(kp=60, ki=0, kd=18, t_end=8.0)
    plot_series([("PD", [(t, x) for t, x, _ in h_pd])], y_min=-0.1, y_max=1.3)
    ov, ss, st = metrics(h_pd, target)
    show_metrics("PD    ", h_pd, target)
    print("  加 Kd 抑制了超调、加快了响应。稳态误差缩小到 f_c / Kp，")
    print("  但只要没有积分项，误差就永远存在 —— 这是 P 和 PD 的结构性缺陷。")

    # -------- 3. PID
    print("\n[3] PID 控制：Kp = 60, Ki = 30, Kd = 25")
    h_pid = simulate(kp=60, ki=30, kd=25, t_end=8.0)
    plot_series([("PID", [(t, x) for t, x, _ in h_pid])], y_min=-0.1, y_max=1.3)
    ov, ss, st = metrics(h_pid, target)
    show_metrics("PID   ", h_pid, target)
    print("  积分项持续累积，最终把静摩擦造成的稳态误差压下去。")
    print("  代价有两个：")
    print("    - 超调变大（本例 12%），积分和稳定性是经典权衡")
    print("    - 收敛靠'爬行'，调节时间 5s+。工业上通常再加摩擦前馈补偿")

    # -------- 4. 三种对比
    print("\n[4] 三条曲线叠在一起看")
    plot_series([
        ("P", [(t, x) for t, x, _ in h_p]),
        ("PD", [(t, x) for t, x, _ in h_pd]),
        ("PID", [(t, x) for t, x, _ in h_pid]),
    ], y_min=-0.1, y_max=1.9)

    # -------- 5. 外力下的位置控制 vs 阻抗控制
    print("\n[5] 遇到外力：位置控制 vs 阻抗控制")
    print("  从 t=2.0s 开始施加 5 N 的外力（相当于有人推了一下）")
    print()

    h_pos = simulate(kp=60, ki=0, kd=25, ext_force=5.0, ext_start=2.0)
    pos_off = (h_pos[-1][1] - 1.0) * 1000
    print(f"  位置控制（Kp=60，无积分）：理论偏移 F/Kp = {5.0 / 60 * 1000:.0f} mm，"
          f"实测 {pos_off:+.1f} mm")
    print(f"  差的约 {5.0 / 60 * 1000 - pos_off:.0f} mm 被静摩擦吃掉了"
          f"（f_c/Kp = 0.6/60 = 10 mm）。")
    print("  '理论算出来和实测不一样'，这正是仿真与真机的差距来源之一。")
    print("  没有积分就不会自行恢复 —— 外力一直在，它就一直在偏移位。")
    plot_series([("pos-ctrl", [(t - 1.8, x) for t, x, _ in h_pos if t >= 1.8])],
                y_min=0.85, y_max=1.20, width=44, height=10)

    print("\n  换成阻抗控制，刚度可以自己选：")
    results = []
    for k_imp in (200.0, 50.0):
        b_imp = 2.0 * math.sqrt(k_imp)          # 取临界阻尼，纯看刚度影响
        h_imp = impedance(k=k_imp, b_imp=b_imp, ext_force=5.0, ext_start=2.0)
        off = (h_imp[-1][1] - 1.0) * 1000
        results.append((k_imp, off))
        print(f"    k = {k_imp:>5.0f} N/m -> 稳态偏移 F/k = {5.0 / k_imp * 1000:>5.0f} mm，"
              f"实测 {off:+.1f} mm")

    k_mid, off_mid = results[0]
    h_imp_mid = impedance(k=k_mid, b_imp=2.0 * math.sqrt(k_mid),
                          ext_force=5.0, ext_start=2.0)
    plot_series([("impedance k=200", [(t - 1.8, x) for t, x, _ in h_imp_mid if t >= 1.8])],
                y_min=0.85, y_max=1.20, width=44, height=10)

    print(f"\n  同一个 5 N 外力下的偏移量：")
    print(f"    位置控制 Kp=60  -> {pos_off:>6.1f} mm（等效刚度 60 N/m）")
    for k_imp, off in results:
        print(f"    阻抗控制 k={k_imp:<4.0f} -> {off:>6.1f} mm")
    print("\n  本质上是同一个公式：偏移 = 外力 / 刚度。")
    print("  位置控制把'刚度'藏在 Kp 里，阻抗控制把它显式暴露出来。")
    print("  所以'硬顶'和'让步'不是两种策略，是同一个参数取不同值。")

    print("\n  两种都不是'失败'：")
    print("  - 位置控制：适合无接触的移动任务，要求精确到达")
    print("  - 阻抗控制：适合装配、插拔、擦桌子这类有接触的任务，")
    print("    让步本身就是需求，硬顶回去反而会把东西弄坏")

    # -------- 6. 数据特征对照表
    print("\n[6] 怎么从数据里看出用的是哪种控制器")
    rows = [
        ("超调 + 振荡后收敛", "P / PD 参数偏小，或欠阻尼", "检查控制参数，勿当采集故障"),
        ("稳态误差恒定", "纯 P 控制，无积分项", "正常现象，归一化时要考虑"),
        ("遇外力立即偏移后保持", "阻抗 / 导纳控制", "接触任务的标准表现"),
        ("遇外力偏移后缓慢回位", "PID 在积分抗饱和", "关注恢复时间是否超出任务要求"),
        ("动作在变但状态不变", "顶到关节限位或堵转", "脏数据，应剔除"),
        ("锯齿状轨迹", "控制频率低于数据采样频率", "插值伪影，需重采样"),
    ]
    print(f"  {'数据特征':<24} {'可能原因':<26} {'处理建议'}")
    print("  " + "-" * 74)
    for r in rows:
        print(f"  {r[0]:<24} {r[1]:<26} {r[2]}")

    # -------- 7. 积分饱和：卡住两秒会怎样
    print("\n[7] 积分的副作用：卡住两秒，松手后冲过头")
    print("  场景：目标 1.0 m，但 0.55 m 处有堵墙，2 秒后墙消失。")
    print("  卡住期间位置误差一直是 0.45 m，积分项（'账本'）一直在往上记。")
    peaks = []
    for clamp_on, tag in ((False, "不抗饱和"), (True, "饱和时冻结积分")):
        h_blk = simulate(kp=60, ki=30, kd=25, t_end=6.0, f_max=40.0,
                         clamp_integ=clamp_on, block_x=0.55, block_until=2.0)
        xs = [x for _, x, _ in h_blk]
        peak = max(xs)
        peaks.append(peak)
        print(f"    {tag:<16} 最高冲到 {peak:.3f} m（超调 {(peak - 1) * 100:>5.1f}%）")
    print(f"  差 {(peaks[0] - peaks[1]) * 100:.0f} cm。账本记的是历史误差总和，")
    print("  墙没了它也不会自己清零，只能靠反方向的误差慢慢还 —— 所以冲过头再退回来。")
    print("  这就是 windup。真机上还有一个更常见的场景：")
    print("  机器人贴着限位等了几秒，一旦任务切换，它会'报复性'地冲出去。")

    # -------- 8. 控制频率 vs 采样频率
    print("\n[8] 控制周期 0.5 ms，数据记多密才够")
    print("  仿真内部按 0.5 ms 积分（相当于 2 kHz 的控制周期），")
    print("  记录间隔可以调：间隔越大，'折线'离真实曲线越远。")
    truth = simulate(kp=60, ki=30, kd=25, t_end=1.2, record_every=1)
    print(f"    {'记录频率':<10}{'采样点':>6}{'折线最大偏差':>14}{'单步最大跳变':>14}")
    print("    " + "-" * 42)
    for hz in (50, 25, 10):
        every = int(round(2000 / hz))
        sparse = simulate(kp=60, ki=30, kd=25, t_end=1.2, record_every=every)
        err, _ = chord_error(truth, sparse)
        step = max(abs(sparse[i + 1][1] - sparse[i][1]) for i in range(len(sparse) - 1))
        print(f"    {hz:>3} Hz{'':<6}{len(sparse):>6}{err * 1000:>12.1f} mm{step * 1000:>12.1f} mm")
    print("  这不是'噪声'，是系统性偏差：低采样率记录下来的轨迹，")
    print("  形状上就已经和真机执行的不是同一条了 —— 动作最快的那一段丢得最狠。")
    print("  入库校验：Δt 是否与声明的频率匹配；同一段位移里是否出现")
    print("  '一步跨很大、几步几乎不动'的阶梯特征（零阶保持的痕迹）。")

    print("\n" + "=" * 76)
    print("结论（数据平台视角）：")
    print("  1. 轨迹形状是控制器的指纹，异常不等于脏数据，先判断是哪种")
    print("  2. 控制频率 vs 采样频率不匹配会产生插值伪影，入库时必须校验")
    print("  3. '动作在变、状态不变'是限位/堵转的确定信号，可自动检测剔除")
    print("  4. 阻抗控制的'让步'是特性不是缺陷，标注时不要标成失败")
    print("=" * 76)


if __name__ == "__main__":
    main()

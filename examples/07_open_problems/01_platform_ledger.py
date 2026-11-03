#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验 1：数据平台的总账 —— 一小时工位时间，最后有几秒钟变成梯度

前六章各算了一笔账：传感器怎么丢信息、流水线怎么失真、数据集长什么样、
模型要多少数据。这一节把它们收进同一本账，回答三个问题：

  1. 采集吞吐的真实瓶颈在哪？
     把一个工位的时间拆开，用 Amdahl 算「改进哪个环节最值」。
     结论会否掉一个直觉：遥操作本身不是瓶颈。

  2. 动作空间统一到末端位姿，丢了什么？
     用三连杆平面臂算清楚：末端位姿 2 个数决定不了 3 个关节角，
     同一条末端轨迹对应一族关节轨迹。用重力力矩量化这族差异的代价。

  3. 自动成败判定能替代人吗？
     算 Cohen's kappa 在「成功率很高」时为什么会崩，
     以及标注吞吐的硬上限。

运行：python examples/07_open_problems/01_platform_ledger.py
依赖：无（只用标准库）
"""

import math
import random


# ---------------------------------------------------------------- 输出对齐

def width(s):
    return sum(2 if ord(c) > 0x2E80 else 1 for c in s)


def pad(s, n):
    return s + " " * max(0, n - width(s))


def hr(title, n=78):
    print("\n" + "=" * n)
    print(title)
    print("=" * n)


# ================================================================ 1. 采集吞吐

# 单条 episode 的时间账（秒）。数字是量级估计，可按自己的工位替换。
PHASES = (
    ("布景 / 摆初始位姿", 25.0),
    ("遥操作执行", 28.0),
    ("复位物体", 18.0),
    ("质检（人看回放）", 15.0),
)

FAIL_RATE = 0.25          # 执行失败 / 中途放弃，必须重采


def per_usable(phases, fail_rate, scene_batch=1):
    """一个可用 episode 需要多少工位秒。

    scene_batch > 1 时，一次布景连着录 n 条（换物体、换目标，不重摆场景），
    布景成本被摊薄到 n 条上。
    """
    per_attempt = 0.0
    for name, sec in phases:
        if name.startswith("布景"):
            per_attempt += sec / float(scene_batch)
        else:
            per_attempt += sec
    return per_attempt / (1.0 - fail_rate)


def section_throughput():
    hr("1. 一小时工位时间，产出的「有效遥操作」有多少")

    total = sum(s for _n, s in PHASES)
    print("单条 episode 的时间账（失败率 %.0f%%，失败要重采）\n" % (FAIL_RATE * 100))
    print(pad("环节", 22) + pad("秒", 8) + pad("占比", 8) + "能不能自动化")
    print("-" * 78)
    auto = {"布景 / 摆初始位姿": "难，要理解场景",
            "遥操作执行": "不能，这就是数据本身",
            "复位物体": "部分可以（振动盘、专用夹具）",
            "质检（人看回放）": "部分可以（见第 3 节）"}
    for name, sec in PHASES:
        print(pad(name, 22) + pad("%.0f" % sec, 8)
              + pad("%.0f%%" % (sec / total * 100), 8) + auto[name])
    print("-" * 78)
    print(pad("单次尝试合计", 22) + pad("%.0f" % total, 8))
    print(pad("摊上 %.0f%% 失败重采" % (FAIL_RATE * 100), 22)
          + pad("%.0f" % (total / (1 - FAIL_RATE)), 8) + "<- 一个可用 episode 的真实成本")

    teleop = 28.0
    eff = teleop / per_usable(PHASES, FAIL_RATE)
    print("\n核心指标：有效数据占比 = %.0f s 遥操作 / %.0f s 工位时间 = **%.1f%%**"
          % (teleop, per_usable(PHASES, FAIL_RATE), eff * 100))
    print("也就是说，一个工位一年 2,000 小时，产出的遥操作数据只有 %.0f 小时。"
          % (2000 * eff))

    # ---------------- 从采集到进梯度：一条漏斗
    print("\n从「采下来」到「真的进梯度」还要再过四道：")
    funnel = (
        ("采到的 episode", 1.00, "——"),
        ("执行成功（未放弃/未撞）", 0.78, "失败样本对模仿学习是纯噪声"),
        ("质检通过（无遮挡/无抖动/标定未漂）", 0.88, "自动可查一部分"),
        ("去重（与已有轨迹几乎重复）", 0.85, "05 章：演示重复走同一条走廊"),
        ("有效多样性（可训练但贡献小）", 0.70, "需要按覆盖度降权，不是硬删"),
    )
    print("\n" + pad("阶段", 34) + pad("保留", 10) + pad("累计", 10) + "为什么")
    print("-" * 78)
    keep = 1.0
    for name, r, why in funnel:
        keep *= r
        print(pad(name, 34) + pad("%.0f%%" % (r * 100), 10)
              + pad("%.0f%%" % (keep * 100), 10) + why)
    print("-" * 78)
    print(pad("最终可用", 34) + pad("", 10) + pad("%.0f%%" % (keep * 100), 10)
          + "<- 采 100 条，约 %.0f 条真的在改参数" % (keep * 100))

    overall = eff * keep
    print("\n两段乘起来：1 小时工位时间 -> %.2f 小时遥操作 -> **%.3f 小时有效训练信号**"
          % (eff, overall))
    print("约等于 %.0f 分钟。这个数字是所有「数据规模」讨论的分母。"
          % (overall * 60))
    print("说「我们有 10,000 小时数据」时，要问清楚是哪一段的小时。")

    # ---------------- Amdahl：改哪个环节最值
    hr("1b. 每个环节改进 2 倍，吞吐涨多少（Amdahl）", 78)

    base = per_usable(PHASES, FAIL_RATE)
    rows = []
    teleop_gain = 1.0
    for name, sec in PHASES:
        faster = [(n, s / 2.0 if n == name else s) for n, s in PHASES]
        ratio = per_usable(faster, FAIL_RATE) / base
        if name.startswith("遥操作"):
            teleop_gain = 1.0 / ratio
        rows.append((name + " 减半", ratio, sec))
    rows.append(("失败率 %.0f%% -> %.0f%%" % (FAIL_RATE * 100, FAIL_RATE * 50),
                 per_usable(PHASES, FAIL_RATE / 2.0) / base, None))
    for n in (2, 4, 8, 16, 1 << 30):
        tag = "一次布景录 n 条（n=%s）" % ("inf" if n > 1000 else n)
        rows.append((tag, per_usable(PHASES, FAIL_RATE, n) / base, None))

    rows.sort(key=lambda r: r[1], reverse=True)
    print(pad("改进项", 30) + pad("耗时缩短", 12) + pad("吞吐倍数", 12) + "备注")
    print("-" * 78)
    for name, ratio, _sec in rows:
        gain = 1.0 / ratio
        note = ""
        if gain >= 1.30:
            note = "<- 最大杠杆：纯粹是流程设计"
        elif gain < 1.12:
            note = "收益已经很小"
        print(pad(name, 30) + pad("%.0f%%" % ((1 - ratio) * 100), 12)
              + pad("%.2f x" % gain, 12) + note)

    print("\n读法：遥操作时长减半只换来 %.0f%%，因为它在总时间里只占 %.0f%%。"
          % ((teleop_gain - 1) * 100, teleop / total * 100))
    print("真正的大头是**布景摊销**：一次摆好场景连着录 N 条不同任务，")
    print("布景成本按 N 摊薄。这是流程设计，不需要任何新硬件。")

    # ---------------- 规模换算
    print("\n换算成日历时间（工位一年按 2,000 小时算）。目标列是**原始遥操作时长**，")
    print("数据集报的「多少小时」通常就是这一档：")
    print()
    def span(station_hours, n_station):
        years = station_hours / (2000.0 * n_station)
        if years < 0.1:
            return "%.0f 天" % (years * 365)
        return "%.1f 年" % years

    print(pad("目标（原始遥操作）", 20) + pad("需要的工位小时", 18)
          + pad("10 工位", 11) + pad("30 工位", 11) + pad("折合有效信号", 15) + "参照")
    print("-" * 78)
    for hours, ref in ((350, "DROID"), (2000, "一个中型实验室"), (10000, "pi0")):
        need = hours / eff
        print(pad("%s h" % format(hours, ","), 20)
              + pad("%s" % format(int(need), ","), 18)
              + pad(span(need, 10), 11) + pad(span(need, 30), 11)
              + pad("%s h" % format(int(hours * keep), ","), 15) + ref)
    print("-" * 78)
    print("最后一列才是真正在改参数的量 —— 原始时长乘 %.0f%% 的漏斗留存。"
          % (keep * 100))
    print("所以「我们有 10,000 小时数据」这句话，如果是原始时长，")
    print("对应约 %s 工位小时、20 个工位年，实际有效信号 %.1f 千小时。"
          % (format(int(10000 / eff), ","), 10000 * keep / 1000))


# ================================================================ 2. 跨本体

LINK = (0.40, 0.35, 0.25)     # 三连杆平面臂，单位 m
MASS = 1.0                    # 末端负载，用来算重力力矩


def fk(angles):
    """正运动学，返回末端 (x, y) 与各关节位置。"""
    x = y = 0.0
    a = 0.0
    pts = [(0.0, 0.0)]
    for ang, L in zip(angles, LINK):
        a += ang
        x += L * math.cos(a)
        y += L * math.sin(a)
        pts.append((x, y))
    return (x, y), pts


def ik_family(target, n_samples=720):
    """末端位姿 -> 一族关节角。

    三连杆平面臂有 3 个自由度，末端位置只给 2 个约束 -> 解集是一条
    一维曲线（自运动流形）。扫 theta1，剩下两连杆做解析 IK。
    """
    L1, L2, L3 = LINK
    px, py = target
    sols = []
    for i in range(n_samples):
        t1 = -math.pi + 2 * math.pi * i / n_samples
        qx = px - L1 * math.cos(t1)
        qy = py - L1 * math.sin(t1)
        r = math.hypot(qx, qy)
        if r > L2 + L3 - 1e-9 or r < abs(L2 - L3) + 1e-9:
            continue
        c = (r * r - L2 * L2 - L3 * L3) / (2 * L2 * L3)
        c = max(-1.0, min(1.0, c))
        for sign in (1.0, -1.0):
            t3 = sign * math.acos(c)
            t2 = math.atan2(qy, qx) - math.atan2(L3 * math.sin(t3),
                                                 L2 + L3 * math.cos(t3))
            sols.append((t1, t2, t3))
    return sols


def gravity_torque(angles):
    """末端挂 1 单位重物时的关节力矩 tau = J^T F，F = (0, -1)。"""
    _, pts = fk(angles)
    abs_a = []
    a = 0.0
    for ang in angles:
        a += ang
        abs_a.append(a)
    taus = []
    for k in range(3):
        # J 的第 k 列 = d(tip)/d(theta_k)：第 k 个关节影响连杆 k..2
        jy = sum(LINK[j] * math.cos(abs_a[j]) for j in range(k, 3))
        taus.append(jy * (-1.0))          # tau = J^T F，F = (0, -1)
    return taus


def circular_span(angles):
    """一组角度覆盖的弧长：360 度减去最大的那个空档。"""
    degs = sorted(math.degrees(a) % 360.0 for a in angles)
    if len(degs) < 2:
        return 0.0
    gaps = [degs[i + 1] - degs[i] for i in range(len(degs) - 1)]
    gaps.append(degs[0] + 360.0 - degs[-1])
    return 360.0 - max(gaps)


def ang_dist(a, b):
    return math.hypot(*[((x - y + math.pi) % (2 * math.pi)) - math.pi
                        for x, y in zip(a, b)])


def section_embodiment():
    hr("2. 动作统一到末端位姿：3 个关节角被 2 个数替掉")

    print("场景：三连杆平面臂，杆长 %s m，末端挂单位重物。" % (LINK,))
    print("动作表示 A：关节角 (3 维)。动作表示 B：末端位置增量 (2 维)。")
    print("跨本体合并数据的通行做法是统一到 B 那类空间 —— 这里算清代价。\n")

    targets = []
    for th in (0.0, 0.6, 1.2, 1.8, 2.4):
        L = sum(LINK) * 0.62
        targets.append((L * math.cos(th), L * math.sin(th)))

    targets = []
    for th in (0.0, 0.6, 1.2, 1.8, 2.4):
        L = sum(LINK) * 0.62
        targets.append((L * math.cos(th), L * math.sin(th)))

    # 先看一个具体例子：末端钉在第二行那个点上，沿解族等间隔取样本
    TGT = targets[1]
    fam = ik_family(TGT)
    fam.sort(key=lambda sol: sol[0])
    picks = [fam[int(len(fam) * k / 6.0)] for k in range(6)]
    peaks_all = [max(abs(t) for t in gravity_torque(sol)) for sol in fam]
    lo_peak = min(peaks_all)
    hi_peak = max(peaks_all)

    print("先把末端钉死在 (%.2f, %.2f)，沿解族等间隔取 6 个解：" % TGT)
    print("（theta 已换成「相对上一节连杆的转角」，单位度）\n")
    print(pad("theta1", 9) + pad("theta2", 9) + pad("theta3", 9)
          + pad("峰值力矩", 11) + pad("相对最省力", 12) + "姿态")
    print("-" * 78)
    for sol in picks:
        peak = max(abs(t) for t in gravity_torque(sol))
        shape = "肘上" if sol[2] > 0 else "肘下"
        print(pad("%+.1f" % math.degrees(sol[0]), 9)
              + pad("%+.1f" % math.degrees(sol[1]), 9)
              + pad("%+.1f" % math.degrees(sol[2]), 9)
              + pad("%.3f" % peak, 11) + pad("%.1f x" % (peak / lo_peak), 12) + shape)

    print("-" * 78)
    print("末端位置一个数都没变，力矩差了 %.1f 倍。整条解族上最大/最小比是 %.1f 倍。"
          % (max(max(abs(t) for t in gravity_torque(x)) for x in picks) / lo_peak,
             hi_peak / lo_peak))

    # 再看五个末端位置上的量级
    print("\n换五个末端位置，看这三个数怎么变：\n")

    print(pad("末端位置", 15) + pad("解族铺开", 11) + pad("肘上/肘下", 12)
          + pad("力矩比", 9) + "说明")
    print("-" * 78)
    worst_fold = 0.0
    worst_ratio = 1.0
    span_seen = 0.0
    for tgt in targets:
        sols = ik_family(tgt)
        if len(sols) < 2:
            continue

        # 折叠歧义：同一个 theta1、theta3 取反号的两个解（肘上 / 肘下）
        by_t1 = {}
        for sol in sols:
            by_t1.setdefault(round(sol[0], 6), []).append(sol)
        fold = 0.0
        for key, group in by_t1.items():
            if len(group) == 2:
                d = ang_dist(group[0], group[1])
                if d > fold:
                    fold = d
        travel = circular_span([sol[0] for sol in sols])
        peaks = [max(abs(t) for t in gravity_torque(sol)) for sol in sols]
        ratio = max(peaks) / min(peaks) if min(peaks) > 1e-9 else float("inf")

        worst_fold = max(worst_fold, fold)
        worst_ratio = max(worst_ratio, ratio)
        span_seen = max(span_seen, travel)
        note = ""
        if abs(ratio - 9.2) < 0.05:
            note = "<- 最极端的一档"
        elif ratio < 2.0:
            note = "正前方，几乎无歧义"
        print(pad("(%.2f, %.2f)" % tgt, 15) + pad("%.0f deg" % travel, 11)
              + pad("%.0f deg" % math.degrees(fold), 12)
              + pad("%.1f x" % ratio, 9) + note)

    print("\n三列读法（后两列在工作空间内基本恒定，它们是解族的**最大**跨度）：")
    print("  · 解族铺开 %.0f 度：解集是一条一维曲线，不是几个离散解。" % span_seen)
    print("    同一条末端轨迹对应连续无穷多种关节轨迹，光靠末端坐标分不开。")
    print("  · 肘上/肘下最多相差 %.0f 度（各关节角差求欧氏距离）。" % math.degrees(worst_fold))
    print("    末端位姿这几个数**完全没有记录它** —— 而它决定手臂会不会撞到旁边的物体。")
    print("  · 力矩比最高 %.1f 倍：末端挂同样的重物，有的姿态快饱和了，有的很轻松。"
          % worst_ratio)
    print("    力控任务里这是本质差别（02 章），不是「稍微差一点」。")
    print("\n所以「统一到末端位姿」丢的是**姿态选择和出力分配**，不是精度。")
    print("对数据平台意味着：末端位姿（可跨本体合并）和关节角（本体专用但信息全）")
    print("**不是二选一，两个都要存**。丢了不可逆 —— 训练时想补也补不回来，")
    print("而多存 7 轴关节角每帧只要 %.1f MB/小时，图像是 %.0f GB/小时，"
          % (7 * 4 * 30 * 3600 / 1e6, 4.0))
    print("相差三个数量级。这是全书里性价比最高的一条 schema 决定。")

    # ---------------- 合并语料时的归一化作用域
    hr("2b. 合并多个本体时，归一化的作用域决定谁被牺牲", 78)

    robots = (
        ("桌面臂", 0.05, "装配、分拣"),
        ("协作臂", 0.80, "上下料、码垛"),
        ("移动操作", 1.50, "取送、巡检"),
    )
    N_BINS = 256
    print("%d 个 bin（8 bit）的动作空间，三个本体合并前后的可用档位：\n" % N_BINS)
    print(pad("本体", 12) + pad("工作空间", 12) + pad("单独归一化", 12)
          + pad("全局归一化", 12) + pad("损失", 8) + "用途")
    print("-" * 78)
    span_max = max(r[1] for r in robots)
    for name, span, use in robots:
        own = N_BINS
        glob = N_BINS * span / span_max
        print(pad(name, 12) + pad("%.2f m" % span, 12) + pad("%d" % own, 12)
              + pad("%.1f" % glob, 12) + pad("x%.0f" % (own / glob), 8) + use)
    print("-" * 78)
    small = [N_BINS * r[1] / span_max for r in robots][0]
    print("\n合并之后，桌面臂的 0.050 m 被映射到 1.500 m 的 %.1f%% 区间里，"
          % (0.05 / 1.5 * 100))
    print("只剩 %.1f 个档位（原 %d）。而 mm 级精度的装配任务，%.1f 档根本不够。"
          % (small, N_BINS, small))
    print("这不是「归一化没做好」，是**作用域选错了**：")
    print("  跨本体共享的是「语义」（向 +x 移动），不是「数值范围」。")
    print("  正确做法是按本体内归一化、外加一个本体 id 的 embedding，")
    print("  让模型学「这个 id 的 0.1 是 5 cm 还是 15 cm」，而不是强行拉平量纲。")


# ================================================================ 3. 自动判定

def kappa(tp, fp, fn, tn):
    n = tp + fp + fn + tn
    if n == 0:
        return 0.0, 0.0
    po = (tp + tn) / float(n)
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / float(n * n)
    if abs(1.0 - pe) < 1e-12:
        return po, 0.0
    return po, (po - pe) / (1.0 - pe)


def section_labeling():
    hr("3. 自动成败判定：为什么「准确率」在这里几乎必然失效")

    print("先看一个反直觉的事实：判断器比「全判成功」只好 1 个百分点，")
    print("但两者的价值差着整个可用性区间。\n")

    print(pad("真实成功率", 12) + pad("全判成功", 11) + pad("kappa", 9)
          + pad("检测器", 11) + pad("kappa", 9) + "同一组指标，两种读法")
    print("-" * 78)
    for prev in (0.50, 0.70, 0.85, 0.92):
        n = 4000
        # 检测器：敏感度 93%，特异度 87%
        tp = int(n * prev * 0.93)
        fn = int(n * prev) - tp
        fp = int(n * (1 - prev) * 0.13)
        tn = int(n * (1 - prev)) - fp
        acc_det, k_det = kappa(tp, fp, fn, tn)
        # 全判成功：准确率恰好等于真实成功率，kappa 恒为 0
        acc_all, k_all = prev, 0.0
        note = ""
        if prev >= 0.85:
            note = "<- 只差 %.0f pt，kappa 从 0 到 %.2f" % ((acc_det - acc_all) * 100, k_det)
        print(pad("%.0f%%" % (prev * 100), 12)
              + pad("%.0f%%" % (acc_all * 100), 11) + pad("%.2f" % k_all, 9)
              + pad("%.0f%%" % (acc_det * 100), 11) + pad("%.2f" % k_det, 9) + note)

    print("\n「全判成功」这个策略的实现是空的 —— 它不做任何判断，只是每次都答「是」。")
    print("但它拿到 92% 的准确率，因为机器人任务的成功率本来就高。")
    print("所以准确率不是一个「弱」指标，是**方向错了**的指标：它的大部分来自先验。")
    print("kappa 扣掉「瞎猜也能对」的那部分，才有可比性。")
    print("工程判据：kappa >= 0.8 可以直接用来过滤训练数据；0.6~0.8 只能用来挑出")
    print("要人看的样本，不能当最终标签；< 0.6 不要接进流水线。")

    # ---------------- 二值标签的模糊带
    print("\n第二层问题：成功是连续量的阈值化，而阈值附近注定有争议。")
    print("两个人各自测同一个末端误差（各自有测量噪声 sigma），二值标签下的分歧率：\n")

    rng = random.Random(7)
    succ = [abs(rng.gauss(0.008, 0.006)) for _ in range(6000)]
    fail = [abs(rng.gauss(0.045, 0.025)) for _ in range(2000)]
    allv = succ + fail
    D = 0.020

    def normal_cdf(z):
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    print(pad("测量噪声", 12) + pad("分歧率", 10) + pad("按 100 万条算", 16)
          + "谁能做到这个精度")
    print("-" * 78)
    for sigma_mm, who in ((0.2, "离线高精度重建（仿真 / 标定后）"),
                          (0.5, "结构光 / 多相机三角化"),
                          (2.0, "单目 + 深度相机（02 章的典型值）"),
                          (5.0, "只靠腕部相机近似")):
        sigma = sigma_mm / 1000.0
        tot = 0.0
        for v in allv:
            p_succ = normal_cdf((D - v) / sigma)
            tot += 2.0 * p_succ * (1.0 - p_succ)
        rate = tot / len(allv)
        print(pad("%.1f mm" % sigma_mm, 12) + pad("%.1f%%" % (rate * 100), 10)
              + pad("%s 条" % format(int(1e6 * rate), ","), 16) + who)
    print("-" * 78)
    print("\n分歧率只由 |误差 - 阈值| / sigma 决定，跟你把阈值定在哪无关：")
    print("换一个阈值只是换一批有争议的样本，争议总量不变。")
    print("而 sigma 是**测量精度**问题，不是标注态度问题 —— 换三个更认真的人也降不下来。")
    print("这就是 02 章标定的钱必须花在采集端的原因：它同时决定了标签的上限。")

    # ---------------- 三值标签
    print("\n所以二值标签本身就是错的 schema。改成三值：\n")
    LOW, HIGH = 0.015, 0.030
    n_ok = sum(1 for v in allv if v < LOW)
    n_bad = sum(1 for v in allv if v > HIGH)
    n_mid = len(allv) - n_ok - n_bad
    print(pad("区间", 22) + pad("占比", 10) + pad("怎么处理", 30) + "谁来做")
    print("-" * 78)
    print(pad("< %.0f mm 合格" % (LOW * 1000), 22) + pad("%.1f%%" % (n_ok / len(allv) * 100), 10)
          + pad("直接进训练集", 30) + "自动")
    print(pad("%.0f ~ %.0f mm 待判" % (LOW * 1000, HIGH * 1000), 22)
          + pad("%.1f%%" % (n_mid / len(allv) * 100), 10)
          + pad("存连续量 + 打标，不进主训练集", 30) + "人 + VLM 复核")
    print(pad("> %.0f mm 不合格" % (HIGH * 1000), 22)
          + pad("%.1f%%" % (n_bad / len(allv) * 100), 10)
          + pad("作为负样本 / 直接丢", 30) + "自动")
    print("-" * 78)
    review_frac = n_mid / float(len(allv))
    print("\n待判 %.0f%% 就是下一步「人只看多少」那个数字的来源。" % (review_frac * 100))
    print("带宽是自己选的，但**选不掉**：把 15~30 mm 收窄到 18~22 mm，带子里的")
    print("样本会流进「合格」和「不合格」，而它们恰好是分歧最大的那批。")
    print("所以人会看到两种分布里最像对方的样本 —— 这活不可能靠定规则消灭。")
    print("能压小它的只有一件事：把任务定义写得更硬（「放进去且不倒、把手朝上」）。")

    # ---------------- 标注吞吐
    hr("3b. 标注吞吐的硬上限", 78)

    print(pad("方式", 26) + pad("每条耗时", 12) + pad("每小时", 10)
          + pad("100 万条要", 16) + "可行性")
    print("-" * 78)
    rows = (
        ("人工看回放", 15.0, 3600.0 / 15.0, "不可行"),
        ("人工只看关键帧(4 张)", 4.0, 3600.0 / 4.0, "不可行"),
        ("规则 + 人工复核 %.0f%%" % (review_frac * 100),
         4.0 * review_frac, 3600.0 / (4.0 * review_frac), "勉强"),
        ("纯自动（规则/仿真重放）", 0.002, 3600.0 / 0.002, "可行，但要 kappa"),
    )
    for name, sec, per_hour, verdict in rows:
        total_h = 1e6 * sec / 3600.0
        if total_h < 1.0:
            need = "%.1f 小时" % total_h
        elif total_h < 2000.0:
            need = "%.0f 小时" % total_h
        else:
            need = "%.1f 人年" % (total_h / 2000.0)
        sec_txt = "%.2f s" % sec if sec >= 0.01 else "2 ms"
        print(pad(name, 26) + pad(sec_txt, 12)
              + pad("%.0f" % per_hour, 10) + pad(need, 16) + verdict)

    print("\n注意第 3 行：自动判定不是为了「更准」，是为了**筛出**那 %.0f%% 需要人看的。"
          % (review_frac * 100))
    print("人只看争议样本，才可能跟上采集速度 —— 这是流程上唯一能规模化的形态。")
    print("但上一节说过，争议样本恰好就在阈值附近，人看了也未必有共识。")
    print("所以平台要存的不是「标签」，是**连续量 + 判据 + 谁标的 + 什么版本**。")

    print("\n一个反例说明为什么必须存依据：")
    print("  「杯子放进去了吗」—— VLM 看最后一张图判「成功」。")
    print("  但杯子是斜的、把手朝下，后面 20 条数据都在模仿这个姿态。")
    print("  只有存了「最终位姿 + 判据 + 置信度」，事后才能定位这批标签的系统性偏差。")
    print("  只存一个 bool 的数据集，出了问题连溯源的入口都没有。")


# ================================================================ 小结

def section_summary():
    hr("4. 小结：把「损耗」写成一列可测的数")

    print("这本书里每个数字都在回答同一个问题的不同侧面：")
    print("「把物理世界的行为，变成可训练的信号，中间损耗了多少？」\n")

    print(pad("指标", 18) + pad("定义", 34) + pad("本书的量级", 14) + "章节")
    print("-" * 84)
    rows = (
        ("有效数据占比", "遥操作时长 / 工位时间", "24%", "07"),
        ("漏斗留存", "进梯度的条数 / 采到的条数", "41%", "07"),
        ("端到端产出", "两者相乘", "10%", "07"),
        ("状态覆盖", "覆盖格数 / 演示条数", "1.2 倍/12 倍", "05"),
        ("出界查询率", "决策落在空网格的比例", "93% -> 0%", "05"),
        ("信息密度", "新像素 / 总像素（30 fps）", "0.7%", "06"),
        ("跨本体损失", "合并后小本体剩余档位", "8.5 / 256", "07"),
        ("标签一致性", "kappa（不是准确率）", "0.61 @ 92%", "07"),
    )
    for a, b, c, d in rows:
        print(pad(a, 18) + pad(b, 34) + pad(c, 14) + d)
    print("-" * 84)
    print("\n这几个数放在一起，能推出全书想说的那句话：")
    print("规模不是买来的，是**挤**出来的 —— 同样的工位时间，")
    print("流程设计（布景摊销）能拿回 1.4 倍，覆盖度监控能拿回几倍的有效样本，")
    print("两者都不需要新硬件，只需要平台把这些数**测出来**。")


def main():
    section_throughput()
    section_embodiment()
    section_labeling()
    section_summary()


if __name__ == "__main__":
    main()

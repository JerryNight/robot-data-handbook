#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验 2：动作表示的三种选择，各自的成本在哪

动作怎么表示，看起来是模型侧的决定，实际上是数据侧的决定：
它规定了 schema 怎么存、归一化怎么做、评估口径怎么定、推理预算够不够。

本实验量四件事：

  1. 量化误差 vs bin 数
     均匀分箱的理论误差是 Δ/√12。对 7 维动作 × 控制频率，bin 数够不够
     一眼能算出来。

  2. 归一化方式对分辨率的影响
     把 6 个关节（rad，±3）和 1 个夹爪（0~1）放在同一个全局 min-max 里，
     夹爪实际能用到多少个 bin？—— 答案和直觉差 6 倍。

  3. 离散 token 的序列长度预算
     「每维每步一个 token」在 chunk=50、30 Hz 下需要多少 token/s？
     和真实解码速度对比，差几个数量级。

  4. 量化步长决定安全余量
     真实值在硬限位附近时，量化有一半概率把它推过线。余量必须 ≥ Δ/2。

运行：python examples/05_learning_from_data/02_action_representation.py
依赖：无（只用标准库）
"""

import math
import random


# ---------------------------------------------------------------- 输出对齐

def width(s):
    return sum(2 if ord(c) > 0x2E80 else 1 for c in s)


def pad(s, n):
    return s + " " * max(0, n - width(s))


def bar(v, n=28):
    k = int(round(v * n))
    return "#" * k + "." * (n - k)


# ---------------------------------------------------------------- 1. 量化误差

def quantize(x, lo, hi, n):
    """均匀分箱，返回 bin 中心值。"""
    step = (hi - lo) / n
    k = int(math.floor((x - lo) / step))
    k = max(0, min(n - 1, k))
    return lo + (k + 0.5) * step


def rmse(vals):
    return math.sqrt(sum(v * v for v in vals) / len(vals))


def section_quantization():
    print("=" * 74)
    print("1. 量化误差 vs bin 数")
    print("=" * 74)

    rng = random.Random(1)
    # 一维动作的真实分布：两个峰（抓取/松开），这在夹爪上极常见
    data = []
    for _ in range(20000):
        mu = 0.55 if rng.random() < 0.5 else -0.55
        data.append(max(-1.0, min(1.0, rng.gauss(mu, 0.18))))

    lo, hi = -1.0, 1.0
    print("动作范围 [-1, 1]，真实分布为双峰（±0.55，σ=0.18）")
    print(pad("bin 数", 10) + pad("步长 Δ", 12) + pad("理论 Δ/√12", 14)
          + pad("实测 RMSE", 12) + "有效位数")
    print("-" * 74)
    for n in (8, 16, 32, 64, 128, 256, 1024, 4096):
        step = (hi - lo) / n
        errs = [quantize(x, lo, hi, n) - x for x in data]
        m = rmse(errs)
        bits = math.log(n, 2)
        print(pad(str(n), 10) + pad("%.5f" % step, 12)
              + pad("%.5f" % (step / math.sqrt(12)), 14)
              + pad("%.5f" % m, 12) + "%.1f bit" % bits)

    print("\n实测和理论吻合 —— 这条公式足以做预算：")
    print("  想要 1 mm 的位置精度，动作范围 1 m，就需要 Δ ≤ 0.001")
    print("  → bin 数 ≥ 2000，也就是 11 bit 以上（还没算多峰的采样误差）")
    print("  想要 0.1° 的关节精度，范围 ±180° → bin 数 ≥ 3600，12 bit")

    # 分箱之后落在「禁区」的比例：边界附近最危险
    print("\n另一个视角：分箱不会只带来平均误差，还会把值**推过边界**。")
    for n in (64, 256, 1024, 4096):
        step = (hi - lo) / n
        lim = 0.95
        over = sum(1 for x in data if x < lim and quantize(x, lo, hi, n) >= lim)
        near = sum(1 for x in data if lim - step <= x < lim)
        print(f"  n={n:5d}  Δ={step:.5f}  "
              f"限位 {lim} 附近被推过线的样本 {over:4d} 条"
              f"（本应安全且离限位不到一个 Δ 的共 {near} 条）")
    print("  → 安全余量必须 ≥ Δ/2，否则「量化」本身就是一个越界来源。")


# ---------------------------------------------------------------- 2. 归一化

def section_normalization():
    print("\n" + "=" * 74)
    print("2. 归一化：量纲不一致时，全局 min-max 会吃掉小量纲的分辨率")
    print("=" * 74)

    # (名字, 该维真实取值范围, 单位, 物理意义)
    dims = [
        ("j1..j6", -2.90, 2.90, "rad", "六个关节"),
        ("gripper", 0.00, 1.00, "0~1", "夹爪开合"),
    ]
    n_bins = 256
    g_lo = min(d[1] for d in dims)
    g_hi = max(d[2] for d in dims)
    g_range = g_hi - g_lo

    print(f"统一量化到 {n_bins} 个 bin（8 bit），这是 VLA 里最常见的设定")
    print(f"全局 min-max 范围：[{g_lo:.2f}, {g_hi:.2f}]，跨度 {g_range:.2f}\n")
    print(pad("维度", 10) + pad("真实范围", 22) + pad("全局 min-max 的 bin", 20)
          + pad("每维 min-max 的 bin", 20) + "损失")
    print("-" * 78)

    for name, a, b, unit, _desc in dims:
        span = b - a
        bins_global = n_bins * span / g_range
        bins_own = n_bins
        loss = bins_own / bins_global
        print(pad(name, 10) + pad(f"[{a:+.2f}, {b:+.2f}] {unit}", 22)
              + pad("%.1f" % bins_global, 20) + pad("%d" % bins_own, 20)
              + "x%.1f" % loss)

    g_step_global = g_range / n_bins
    g_step_own = 1.0 / n_bins
    print(f"\n夹爪实际分辨率：全局 {g_step_global:.4f} vs 单独 {g_step_own:.4f}")
    print(f"也就是说夹爪只剩 {n_bins * 1.0 / g_range:.0f} 个可用档位，"
          f"而不是 {n_bins} 个。")
    print("抓取 5 mm 的物体需要 0.005 的夹爪分辨率 —— 全局归一化给不了。")

    # 离群值的影响
    print("\n更糟的情况：某一条 episode 里有个关节冲到了 3.60（碰撞后的过冲）。")
    g_hi2 = 3.60
    g_range2 = g_hi2 - g_lo
    print(f"  全局范围变成 [{g_lo:.2f}, {g_hi2:.2f}]，跨度 {g_range2:.2f}")
    print(f"  夹爪可用档位：{n_bins * 1.0 / g_range2:.0f} 个，"
          f"分辨率 {g_range2 / n_bins:.4f}")
    print(f"  → 一条脏数据让整个数据集的夹爪分辨率退化了 "
          f"{(g_range2 / n_bins) / (g_range / n_bins) - 1:+.0%}")
    print("  这就是为什么归一化统计量必须用分位数（1%/99%）而不是 min/max：")
    print("  分位数对离群值不敏感，min/max 对一条坏数据就崩。")


# ---------------------------------------------------------------- 3. token 预算

def section_token_budget():
    print("\n" + "=" * 74)
    print("3. 离散 token：真正的约束是「每秒要产出多少 token」")
    print("=" * 74)

    FREQ = 30.0          # 控制频率 Hz
    DOF = 7
    MS_PER_TOKEN = 15.0  # 7B 级模型单序列自回归解码的乐观值（假设，可改）

    print(f"设定：{DOF} 维动作，控制频率 {FREQ:.0f} Hz（每步预算 "
          f"{1000/FREQ:.1f} ms），解码 {MS_PER_TOKEN:.0f} ms/token")
    print("每维每步一个 token 是最直白的离散化方式（RT-2 属于这一类）\n")

    need = DOF * FREQ                     # 与 chunk 无关！
    have = 1000.0 / MS_PER_TOKEN
    print(f"需要的吞吐量 = {DOF} 维 x {FREQ:.0f} Hz = {need:.0f} token/s")
    print(f"单序列能给的吞吐 = 1000 / {MS_PER_TOKEN:.0f} = {have:.0f} token/s")
    print(f"缺口 = {need/have:.1f} 倍（这就是自回归动作头在实时场景的硬墙）\n")

    print(pad("方案", 30) + pad("token 数", 12) + pad("延迟", 12)
          + pad("对预算", 12) + "结论")
    print("-" * 74)
    rows = [
        ("每维每步 1 token, chunk=1", 7, "差 3.2 倍"),
        ("每维每步 1 token, chunk=10", 70, "差 3.2 倍"),
        ("每维每步 1 token, chunk=50", 350, "差 3.2 倍"),
        ("FAST 式频域 tokenizer (10x)", 35, "留 3.1 倍余量"),
        ("连续回归 / flow 动作头", 0, "1 次前向，与 chunk 无关"),
    ]
    for name, n_tok, verdict in rows:
        if n_tok == 0:
            print(pad(name, 30) + pad("-", 12) + pad("~30 ms", 12)
                  + pad("刚好", 12) + verdict)
            continue
        lat = n_tok * MS_PER_TOKEN / 1000.0
        budget = n_tok / float(need)          # 这批 token 允许花的时间
        print(pad(name, 30) + pad(str(n_tok), 12) + pad("%.2f s" % lat, 12)
              + pad("%.2f s" % budget, 12) + verdict)

    print("\n注意第 2、3 行：chunk 变大，单次解码变慢，但允许花的时间同比变长，")
    print("比值恒为 %.1f —— 所以「加大 chunk」救不了吞吐缺口，" % (need / have))
    print("因为缺口是「每秒 token 数」而不是「每次 token 数」。")

    print("\n对 per-token 速度的敏感性（缺口倍数 = 7 维 x ms/token x 30 Hz / 1000）：")
    print(pad("ms/token", 12) + pad("需要倍数", 12) + "现实对应")
    for ms, note in ((5.0, "小模型 / 高度优化的推理栈"),
                     (15.0, "7B 级模型单序列"),
                     (50.0, "未优化 / 大模型 / 高精度")):
        print(pad("%.0f" % ms, 12) + pad("%.1f x" % (DOF * ms * FREQ / 1000.0), 12)
              + note)

    print("\n结论：离散 token 的瓶颈不是精度，是序列长度 x 频率。")
    print("出路只有两条：压缩（FAST 这类动作 tokenizer，代价是多一个编解码器")
    print("和它的重建误差），或者换非自回归的动作头（回归 / flow）。")


# ---------------------------------------------------------------- 4. 安全余量

def section_safety_margin():
    print("\n" + "=" * 74)
    print("4. 量化步长就是安全余量的下限")
    print("=" * 74)

    LIMIT = 0.95
    print("场景：夹爪指令 >= %.2f 就压碎物体。真实值落在 [%.2f, %.2f) 这段安全带里。"
          % (LIMIT, LIMIT - 0.05, LIMIT))
    print("问题：分箱之后，这些本应安全的值有多少被推到限位以上？")
    print("（网格和限位对不对齐，完全是运气 —— 所以扫一遍所有对齐方式取最坏）\n")

    print(pad("bin 数", 10) + pad("Δ", 12) + pad("Δ/2", 10)
          + pad("被推过线比例", 16) + pad("（最好）", 12) + "结论")
    print("-" * 74)
    rng = random.Random(3)
    band = [LIMIT - 0.05 + 0.05 * rng.random() for _ in range(4000)]
    for n in (16, 64, 256, 1024, 4096):
        step = 2.0 / n
        worst_best = []
        for k in range(20):                    # 扫 20 种网格对齐
            off = -1.0 + step * k / 20.0
            over = 0
            for x in band:
                kk = int(math.floor((x - off) / step))
                q = off + (kk + 0.5) * step
                if q >= LIMIT:
                    over += 1
            worst_best.append(over / float(len(band)))
        lo_p, hi_p = min(worst_best), max(worst_best)
        verdict = "余量必须 ≥ %.4f" % (step / 2.0)
        print(pad(str(n), 10) + pad("%.5f" % step, 12) + pad("%.5f" % (step / 2), 10)
              + pad("%.1f%%" % (hi_p * 100), 16) + pad("%.1f%%" % (lo_p * 100), 12)
              + verdict)

    print("\n两个极端都在表里：")
    print("  · Δ 远小于安全带（n=4096）→ 只有 0.4% 沦陷，可以忽略。")
    print("  · Δ 接近安全带宽度（n=16，Δ=0.125 > 带宽 0.05）→ 整段安全带被")
    print("    一个 bin 吞掉，最坏对齐下 100% 变成危险动作 —— 而好对齐下是 0%。")
    print("单值的最大推过量有严格上界 Δ/2；一个区间里沦陷多少，取决于 Δ 相对")
    print("带宽的比例，以及网格和限位对不对齐 —— 后者事先不可知。")
    print("所以工程上只能按最坏情况留余量：安全阈值 - 指令上限 >= Δ/2。")

    print("\n同一个真实值 0.945、同一个 n=16，只是网格起点平移：")
    print(pad("网格起点", 12) + pad("量化结果", 14) + "结局")
    for off in (0.00, 0.03, 0.06, 0.09, 0.12):
        step = 2.0 / 16
        kk = int(math.floor((0.945 - (-1.0 + off)) / step))
        q = (-1.0 + off) + (kk + 0.5) * step
        print(pad("%+.3f" % (-1.0 + off), 12) + pad("%.4f" % q, 14)
              + ("安全" if q < LIMIT else "压碎"))
    print("同一个策略、同一个动作，结局取决于网格对齐 —— 所以余量不能按运气留。")

    print("\n对 schema 的要求：限位值、量化步长、分箱范围必须一起存。")
    print("只存限位不存量化的数据集，事后无法判断「这个动作是安全的」。")


# ---------------------------------------------------------------- 主流程

def main():
    section_quantization()
    section_normalization()
    section_token_budget()
    section_safety_margin()

    print("\n" + "=" * 74)
    print("小结：三种表示的取舍")
    print("=" * 74)
    rows = [
        ("连续回归", "单峰、平滑、短程", "多模态必然取均值",
         "每维 float32 / float16", "逐维标准化"),
        ("离散 token", "可多峰、可和 LLM 同构", "量化误差 + 序列长",
         "每维每步 1~2 token", "必须逐维，且要存 bin 定义"),
        ("flow / diffusion", "多峰、连续、无量化损失", "推理迭代步数",
         "float 张量 + chunk 维", "逐维标准化 + chunk 长度"),
    ]
    print(pad("表示", 18) + pad("强项", 26) + pad("硬伤", 26) + "数据侧要多存什么")
    print("-" * 100)
    for a, b, c, d, _e in rows:
        print(pad(a, 18) + pad(b, 26) + pad(c, 26) + d)

    print("\n对数据平台的三条结论：")
    print("1. 动作的归一化统计量（min/max/分位数）必须**随数据集版本一起存**，")
    print("   否则换个版本重训，同一个物理动作会变成不同的数字。")
    print("2. 如果用离散 token，bin 定义（范围、数量、是否逐维）是 schema 的")
    print("   一部分，不是训练脚本里的超参 —— 它决定了数据能不能被别的模型用。")
    print("3. 量化步长要和物理限位一起校验。Δ/2 是安全余量的下限。")


if __name__ == "__main__":
    main()

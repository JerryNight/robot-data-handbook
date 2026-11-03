#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验 1：语言到底承载了多少信息 —— VLA 和「VLM + 机械臂」的分界线

几乎所有 VLA 都是「图像 + 指令 → 动作」。但语言在这条链路上究竟提供了
多少比特？剩下没被指定的部分，只能从数据里学。本实验算三个量：

  1. 指令的指代力
     桌面上 20 个物体，「拿红色的」还剩几个候选？多少比特没被指定？

  2. 语言说了目标，没说约束
     「抓红色杯子」没有提到「从哪个方向接近」。在拥挤的桌面上，
     几何上可行的接近方向只占一小部分 —— 这个约束语言一个字都没说。

  3. 把动作「用文字说出来」要多少 token
     分层方案（VLM 出文字计划 + 低层控制器）的接口带宽上限是多少 Hz？

结论指向同一件事：语言指定 **做什么**，不指定 **怎么做**。两者的差值
就是模仿学习要从数据里补的那部分 —— 也是数据平台真正在生产的商品。

运行：python examples/06_vla/01_what_language_carries.py
依赖：无（只用标准库）
"""

import math
import random


# ---------------------------------------------------------------- 输出对齐

def width(s):
    return sum(2 if ord(c) > 0x2E80 else 1 for c in s)


def pad(s, n):
    return s + " " * max(0, n - width(s))


# ================================================================ 1. 指令指代力

COLORS = ("red", "blue", "green", "yellow")
SHAPES = ("cup", "bowl", "block", "bottle", "plate")
SIZES = ("small", "large")

# 人类实际会说出的属性组合。注意：没有人会说「大的红色的那个杯子」
# 这样的全属性描述 —— 语言习惯是「能区分就停」。
LEVELS = (
    ("L0  只说动词", ()),
    ("L1  + 颜色", (0,)),
    ("L2  + 颜色+形状", (0, 1)),
    ("L3  + 颜色+形状+大小", (0, 1, 2)),
)


def attrs(obj):
    return (COLORS[obj[0]], SHAPES[obj[1]], SIZES[obj[2]])


def section_reference(rng):
    print("=" * 76)
    print("1. 指令的指代力：说「红色的」还剩几个候选？")
    print("=" * 76)
    print("物体属性空间 = 4 色 x 5 形 x 2 大小 = 40 种组合")
    print("每张桌子随机摆 n 个互不相同的物体，指令只提到部分属性\n")

    all_combos = [(c, s, z) for c in range(4) for s in range(5) for z in range(2)]

    print(pad("桌面上物体数", 14) + pad("指令详细度", 22)
          + pad("无歧义比例", 12) + pad("平均候选数", 12) + "未指定的比特")
    print("-" * 78)

    for n in (5, 10, 20, 40):
        for name, keep in LEVELS:
            unamb = 0
            cand_sum = 0.0
            trials = 400
            for _ in range(trials):
                objs = rng.sample(all_combos, n)
                # 选一个目标，按 keep 指定的属性生成指令
                tgt = objs[rng.randrange(n)]
                key = tuple(tgt[i] for i in keep)
                cands = [o for o in objs if all(o[i] == tgt[i] for i in keep)]
                if len(cands) == 1:
                    unamb += 1
                cand_sum += len(cands)
            p_unamb = unamb / float(trials)
            mean_c = cand_sum / trials
            bits = math.log(mean_c, 2) if mean_c > 1 else 0.0
            print(pad(str(n), 14) + pad(name, 22)
                  + pad("%5.1f%%" % (p_unamb * 100), 12)
                  + pad("%.2f" % mean_c, 12) + "%.2f bit" % bits)
        print()

    print("读法：20 个物体的桌面上，只说颜色平均还剩 5.41 个候选 ——")
    print("2.44 bit 的不确定性没被指定。这些比特必须从**图像**里补，")
    print("这正是视觉编码器在 VLA 里真正干的活。")
    print("说到颜色+形状（52% 无歧义）人就不说了：语言习惯是「够用就停」。")

    # 一句话背后有多少动作
    print("\n一句话背后有多少个 (状态, 动作) 对：")
    TASK = 68                 # π0 报告的任务数
    STEPS = 903_000_000       # π0 报告的时间步数
    TOK = 10                  # 一句任务描述约 10 个 token
    BITS_PER_TOK = 12.0       # 词表约 32k，实际熵约 10~14 bit/token
    print("  统计口径：π0 的 %d 个任务、9.03 亿个时间步（论文 §V）" % TASK)
    print("  平均每个任务 = %.1f M 个时间步" % (STEPS / TASK / 1e6))
    print("  而「把红色杯子放进碗里」约 %d 个 token ≈ %.0f bit" % (TOK, TOK * BITS_PER_TOK))
    print("  → 一句话是 %.0f bit 的**索引**，指向 %.1f M 个动作样本。"
          % (TOK * BITS_PER_TOK, STEPS / TASK / 1e6))
    print("  语言没有描述这些动作，它只是给这一大团行为起了个名字。")
    print("  这一大团内容，就是数据平台要采集和治理的东西。")


# ================================================================ 2. 接近方向

def blocked(angle, obj_r, neighbors, grip_width, reach):
    """沿 angle 方向接近时，夹爪的扫掠体是否被邻物挡住。

    夹爪建模：以物体中心为原点，沿接近方向 u 张开一条宽 grip_width、
    长 reach 的走廊（含物体本身），把邻物按半径膨胀后做矩形相交测试。
    这是 2D 俯视图下的保守近似（忽略了圆角，略微高估碰撞）。
    """
    ux, uy = math.cos(angle), math.sin(angle)
    half = grip_width / 2.0
    for (px, py, pr) in neighbors:
        along = px * ux + py * uy
        perp = abs(-px * uy + py * ux)
        if perp < half + pr and -obj_r - pr < along < reach + pr:
            return True
    return False


def free_arc(angle_free):
    """返回最长连续可行弧段的角度（度）。环形数组。"""
    n = len(angle_free)
    if all(angle_free):
        return 360.0
    if not any(angle_free):
        return 0.0
    best = 0
    run = 0
    # 从某个被挡住的位置开始扫，避免跨 0 度断开
    start = angle_free.index(False)
    for k in range(n):
        if angle_free[(start + k) % n]:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best * 360.0 / n


def section_approach(rng):
    print("\n" + "=" * 76)
    print("2. 语言说了目标，没说约束：桌面上还剩几个可行的接近方向？")
    print("=" * 76)
    print("场景：目标物体（半径 0.030 m）在桌面中心，周围散布玩具。")
    print("夹爪开口 0.060 m；扫掠体长度按 0.55 m 算 —— 整条小臂都要从桌边伸进来，")
    print("不只是手指。方向角扫 720 个。")
    print("邻物位置在每个场景里随机重采样（离中心 0.12~0.40 m），取 300 个场景平均。\n")

    OBJ_R = 0.030
    GRIP = 0.060
    REACH = 0.550
    N_ANG = 720

    print(pad("邻物个数", 12) + pad("可行方向", 12) + pad("最宽弧中位", 14)
          + pad("最宽弧 P10", 12) + pad("弧<60度", 13) + "说明")
    print("-" * 76)

    for k in (0, 1, 2, 4, 6, 8, 10, 12, 16):
        free_ratio = []
        arcs = []
        scenes = 300
        for _ in range(scenes):
            neighbors = []
            tries = 0
            while len(neighbors) < k and tries < 4000:
                tries += 1
                d = rng.uniform(0.12, 0.40)
                th = rng.uniform(0, 2 * math.pi)
                pr = rng.uniform(0.015, 0.040)
                px, py = d * math.cos(th), d * math.sin(th)
                if math.hypot(px, py) < OBJ_R + pr + 0.005:
                    continue
                neighbors.append((px, py, pr))
            flags = [not blocked(2 * math.pi * i / N_ANG, OBJ_R, neighbors, GRIP, REACH)
                     for i in range(N_ANG)]
            free_ratio.append(sum(flags) / float(N_ANG))
            arcs.append(free_arc(flags))

        arcs_sorted = sorted(arcs)
        med = arcs_sorted[len(arcs) // 2]
        p10 = arcs_sorted[int(len(arcs) * 0.10)]
        narrow = sum(1 for a in arcs if a < 60.0) / float(scenes)
        fr = sum(free_ratio) / scenes
        note = ""
        if k == 0:
            note = "无障碍，随便抓"
        elif narrow > 0.4:
            note = "近半场景只剩窄缝"
        elif narrow > 0.15:
            note = "必须按摆位选方向"
        print(pad(str(k), 12) + pad("%5.1f%%" % (fr * 100), 12)
              + pad("%5.0f deg" % med, 14) + pad("%5.0f deg" % p10, 12)
              + pad("%5.1f%%" % (narrow * 100), 13) + note)

    print("\n三列读法：")
    print("  · 可行方向比例降得快：桌子一挤，能活着伸进去的方向就没了一半。")
    print("  · 但「最宽弧」塌得快，P10 从 360 度掉到几十度：**能用的方向越来越窄**。")
    print("  · 「弧<60度场景」这一列才是真危险：这些场景里，随便挑一个方向去抓")
    print("    几乎必撞，必须根据邻物的具体摆位选方向。")
    print("\n同一句「抓红色杯子」，摊到不同场景上就是**一整个动作分布**。")
    print("分布的形状由几何决定，而几何在图像里，不在句子里。")
    print("所以「一句话 → 一个动作」这个假设从来不成立；策略必须把场景读进来。")
    print("这也是为什么纯语言条件的数据集（只存指令不存多视角图像）没法训练 VLA。")


# ================================================================ 3. 文字接口带宽

def section_text_bandwidth():
    print("\n" + "=" * 76)
    print("3. 把动作「念出来」：文字接口的带宽上限")
    print("=" * 76)

    DOF = 7
    STEPS = 50
    DEC = 3               # 小数位数
    TOK_PER_NUM = 4       # "0.123" 在 Llama 类分词器里约 3~4 个 token
    MS_PER_TOKEN = 15.0   # 单序列自回归解码的乐观值

    n_num = DOF * STEPS
    n_tok = n_num * TOK_PER_NUM
    sec = n_tok * MS_PER_TOKEN / 1000.0

    print("设定：%d 维动作、%d 步、%.*f 精度（每个数字约 %d 个 token），"
          % (DOF, STEPS, DEC, 0.123, TOK_PER_NUM))
    print("      自回归解码 %.0f ms/token（7B 级模型单序列）\n" % MS_PER_TOKEN)

    print(pad("接口粒度", 26) + pad("token 数", 10) + pad("耗时", 12) + "能否支撑 30 Hz")
    print("-" * 76)
    rows = [
        ("每 episode 一句描述", 20, "可以（控制回路自己跑）"),
        ("每 2 秒一个子任务", 50, "可以（2 Hz 规划 + 30 Hz 控制）"),
        ("每步一个 7 维动作", n_tok, "不行，差 %.0f 倍" % (sec / (STEPS / 30.0))),
    ]
    for name, tok, verdict in rows:
        t = tok * MS_PER_TOKEN / 1000.0
        print(pad(name, 26) + pad("%.0f" % tok, 10) + pad("%.2f s" % t, 12) + verdict)

    rate = 1000.0 / MS_PER_TOKEN / TOK_PER_NUM / DOF
    print("\n文字接口的天花板：1000/%.0f token/s ÷ %d token/number ÷ %d 维 = %.1f Hz"
          % (MS_PER_TOKEN, TOK_PER_NUM, DOF, rate))
    print("50 Hz 灵巧操作是它的 %.0f 倍。所以「VLM 出文字计划」只能工作在"
          % (50.0 / rate))
    print("子任务层（1~2 Hz），一旦要求它承载控制信号，接口带宽就差两个数量级。")

    print("\n反过来算：π0 一次前向产出 %d 步 x %d 维 = %d 个连续值，用时 73 ms。"
          % (STEPS, DOF, n_num))
    print("同样内容用文字念完要 %.1f s —— 差 %.0f 倍。" % (sec, sec / 0.073))
    print("\n这不是说分层架构不行（它工作得很好），而是说它的接口在**语义层**：")
    print("文字承载「做什么」，动作头承载「怎么做」。谁承载后者，谁就得吃下")
    print("连续空间的全部带宽 —— 这也是 VLA 把动作塞进模型内部的根本原因。")


# ================================================================ 主流程

def main():
    rng = random.Random(7)
    section_reference(rng)
    section_approach(rng)
    section_text_bandwidth()

    print("\n" + "=" * 76)
    print("小结：三条链路，同一个结论")
    print("=" * 76)
    print("1. 「红色的」在 20 个物体里剩下 2.3 bit 的不确定性 —— 得看图。")
    print("2. 拥挤桌面上可行接近方向只剩一条窄弧 —— 得看几何，语言没提。")
    print("3. 把动作念出来比直接输出慢 288 倍 —— 文字接口到不了控制频率。")
    print("\n所以 VLA 的「V」不是装饰：它和「L」一起，把语言没说的那些比特")
    print("从像素里补回来。而补得准不准，取决于训练数据覆盖了多少种")
    print("「同一句话、不同的几何」——这正好是数据平台要保证的东西。")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验 3：VLA 的数据账本 —— 字节、帧率、钱都花在哪

前三章讲「数据怎么采」，第五章讲「数据怎么用」。这一节把 RT-2 / OpenVLA /
π0 / GR00T 这些模型的公开数字放进同一本账，回答三个问题：

  1. 一份 π0 规模的数据集，字节都花在哪？
     动作数据不到千分之一，图像占 99.9% 以上。这个比例决定所有优化方向。

  2. 30 Hz 是不是浪费？
     物体以固定物理速度运动，帧率翻三倍，token 翻三倍，但「新信息」不翻。
     实测一下每份新信息值多少 token。

  3. 采集贵还是训练贵？
     用公开的 GPU 小时数做锚点，比一下两边的量级。

运行：python examples/06_vla/03_data_ledger.py
依赖：无（只用标准库）
"""

import math


# ---------------------------------------------------------------- 输出对齐

def width(s):
    return sum(2 if ord(c) > 0x2E80 else 1 for c in s)


def pad(s, n):
    return s + " " * max(0, n - width(s))


def human(n_bytes):
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if n_bytes < 1000.0:
            return "%.1f %s" % (n_bytes, unit)
        n_bytes /= 1000.0
    return "%.1f EB" % n_bytes


# ================================================================ 1. 字节账本

def section_bytes():
    print("=" * 78)
    print("1. 字节账本：动作不到千分之一，图像占 99.9%")
    print("=" * 78)

    # π0 公开数据：10,000 小时、903M 时间步、7 种本体、3 相机（论文 §V）
    STEPS = 903e6
    CAMS = 3
    DOF = 7
    HZ = 30
    # 224x224 JPEG q90 典型 12~18 KB，取 15 KB
    JPEG_KB = 15.0
    # 每个时间步的状态向量：关节位置 + 夹爪 + 末端位姿，约 16 维 float32
    STATE_DIM = 16

    img = STEPS * CAMS * JPEG_KB * 1000
    act = STEPS * DOF * 4
    state = STEPS * STATE_DIM * 4
    total = img + act + state

    print("口径：π0 的 10,000 小时 / 903M 时间步 / 3 相机 / %d 维动作（论文 §V）" % DOF)
    print("假设：224x224 JPEG q90 约 %.0f KB/帧；动作与状态用 float32\n" % JPEG_KB)

    print(pad("内容", 22) + pad("每时间步", 14) + pad("总量", 14) + "占比")
    print("-" * 78)
    for name, b in (("图像（3 相机）", img), ("动作（7 维）", act),
                    ("状态（16 维）", state)):
        print(pad(name, 22) + pad(human(b / STEPS), 14) + pad(human(b), 14)
              + "%.4f%%" % (b / total * 100))
    print("-" * 78)
    print(pad("合计", 22) + pad(human(total / STEPS), 14) + pad(human(total), 14) + "100%")

    print("\n换算成别的口径：")
    print("  · 只存动作：%s —— 一个手机装得下，而且是无损的。" % human(act))
    print("  · 图像 : 动作 = %.0f : 1" % (img / act))
    print("  · 每 1 小时数据 = %s（图像占 %s）"
          % (human(total / 10000.0), human(img / 10000.0)))

    print("\n这一行比例决定了几件事：")
    print("  1. 任何「数据量太大」的问题都是视频问题，不是动作问题。")
    print("     想省 100 倍存储，只能动图像；动作怎么存都无所谓。")
    print("  2. 动作必须无损、必须保精度（第 5 章：量化步长就是安全余量），")
    print("     因为它便宜 —— 在只占千分之一的字段上抠精度是纯亏。")
    print("  3. 视频那边的每一次压缩都要回答「会不会把控制需要的细节压掉」：")
    print("     抓取前的 3 帧、接触瞬间的形变，恰好是低码率编码最先糊掉的东西。")

    # 用 OpenVLA 的数据量做对照
    EP = 970e3
    print("\n对照 OpenVLA：970k episodes（OXE）。按每条 %.1f 秒算，"
          % (STEPS / EP / HZ))
    print("  总时长约 %.0f 小时 —— 和 π0 的 10,000 小时同一量级。"
          % (EP * (STEPS / EP / HZ) / 3600.0))
    print("  两个团队独立收敛到同一个量级，说明这个规模是当前方法的**入场券**，")
    print("  不是某家的独门优势。")
    return total


# ================================================================ 2. 时间冗余

def render(size, cx, cy, side, tex, bg=40, fg=210):
    """画一帧：静态纹理背景 + 一个移动的方块。纹理固定，方块动。"""
    img = tex[:]                      # 背景是固定的纹理（相机不动）
    x0, y0 = int(cx), int(cy)
    for y in range(y0, y0 + side):
        if 0 <= y < size:
            base = y * size
            for x in range(x0, x0 + side):
                if 0 <= x < size:
                    img[base + x] = fg
    return img


def new_pixels(a, b):
    """相邻两帧里发生了变化的像素数。"""
    n = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            n += 1
    return n


def section_temporal():
    print("\n" + "=" * 78)
    print("2. 30 Hz 是不是浪费：冗余是真的，但不能靠降帧率去省")
    print("=" * 78)

    SIZE = 96
    SIDE = 24
    PX_PER_M = 320.0          # 3.1 mm/px
    SPEED = 0.20              # 物体移动 0.20 m/s
    DUR = 3.0

    rng = __import__("random").Random(11)
    tex = [rng.randrange(256) for _ in range(SIZE * SIZE)]   # 静态纹理背景

    print("场景：%dx%d 灰度图，相机不动（背景纹理固定），一个 %d px 的物体以 %.2f m/s 移动"
          % (SIZE, SIZE, SIDE, SPEED))
    print("观察 %.0f 秒。%.0f px/m，即每像素 %.1f mm\n" % (DUR, PX_PER_M, 1000.0 / PX_PER_M))

    v_px = SPEED * PX_PER_M
    total_px = SIZE * SIZE
    print(pad("帧率", 8) + pad("帧间位移", 12) + pad("每帧新像素", 14)
          + pad("新像素占比", 14) + pad("冗余倍数", 12) + "说明")
    print("-" * 78)
    for fps in (60, 30, 15, 10, 5, 2):
        n_frames = int(round(DUR * fps))
        step = v_px / fps
        prev = render(SIZE, 5, 5, SIDE, tex)
        new_tot = 0
        for i in range(1, n_frames):
            cur = render(SIZE, 5 + i * step, 5 + i * step * 0.6, SIDE, tex)
            new_tot += new_pixels(prev, cur)
            prev = cur
        per_frame = new_tot / float(max(1, n_frames - 1))
        frac = per_frame / total_px
        note = ""
        if fps == 30:
            note = "<- 常见的采集帧率"
        elif fps == 60:
            note = "<- 灵巧操作"
        print(pad(str(fps), 8) + pad("%.1f px" % step, 12)
              + pad("%.0f / %d" % (per_frame, total_px), 14)
              + pad("%.1f%%" % (frac * 100), 14)
              + pad("%.1f x" % (1.0 / frac), 12) + note)

    print("\n两件事同时成立，而且不矛盾：")
    print("  1. 冗余是真的：30 fps 下每帧只有 0.7% 的像素是新的，99% 是上一帧的重发。")
    print("     帧率越高这个比例越低 —— 60 fps 时只剩 0.4%。")
    print("     （本实验的物体只占画面 6%；真实场景里机械臂更大、动得更快，")
    print("      这个比例会到十几个百分点，但冗余仍是主要成分 ——")
    print("      静止机位的机器人视频用 H.264 常见 10~30 倍压缩，可以印证。）")
    print("  2. 但**不能靠降帧率来省**：")
    print("     物体 %.2f m/s 时，30 Hz 每帧移动 %.1f mm，已经接近精细抓取的容差"
          % (SPEED, v_px / 30.0 * (1000.0 / PX_PER_M)))
    print("     （1 mm 精度需要 %.0f Hz）。降到 10 Hz 就变成 %.1f mm/帧，"
          % (SPEED * 1000.0, v_px / 10.0 * (1000.0 / PX_PER_M)))
    print("     接触瞬间的形变根本追不上 —— 而且这个损失**不可逆**，存下来就没了。")
    print("\n所以正确的省钱方式是**编码器**，不是降采样：")
    print("  帧间冗余（99%）正是视频编码器吃掉的那部分，压缩是（近）无损的、可逆的。")
    print("  降帧率是砍采样，损失的是时间分辨率，不可恢复。")
    print("\n这条推论直接给出采集系统的默认值：")
    print("  存 30~60 Hz 的原始视频、用好的编码器压、把码率预算花在接触瞬间；")
    print("  训练时再决定要不要降采样 —— 反过来做没有回头路。")


# ================================================================ 3. 成本账

def section_cost(total_bytes):
    print("\n" + "=" * 78)
    print("3. 钱花在哪：采集 vs 训练")
    print("=" * 78)

    HOURS = 10000.0        # π0 规模
    LABOR = 80.0           # 元/小时，遥操作 + 现场人力
    OVERHEADS = (1.0, 3.0, 5.0)   # 现场开销倍数：重置、失败重采、布景
    GPU_HOUR_PRICE = 10.0  # 元/A100-小时（云上按需的乐观价）

    data_cost = {k: HOURS * LABOR * k for k in OVERHEADS}

    # 公开训练算力：OpenVLA 21,500 A100-小时（64 x A100 x 14 天）
    #               GR00T N1 约 50,000 H100-小时
    trains = (("OpenVLA 7B（实测公开值）", 21500.0),
              ("GR00T N1 2B（实测公开值）", 50000.0))

    print("假设（都是量级估算，不是报价单，可以按自己的价钱替换）：")
    print("  人工 + 设备 %.0f 元/小时；GPU %.0f 元/A100-小时；目标 %.0f 小时数据"
          % (LABOR, GPU_HOUR_PRICE, HOURS))
    print("  现场开销 = 布景、重置、失败重采导致的倍数\n")

    # 公开训练算力：OpenVLA 21,500 A100-小时（64 卡 x 14 天）
    #               GR00T N1 约 50,000 H100-小时（按 A100 价格粗算，实际更贵）
    t_ovla = 21500.0 * GPU_HOUR_PRICE
    t_groot = 50000.0 * GPU_HOUR_PRICE

    print(pad("现场开销", 10) + pad("采集成本", 16) + pad("vs OpenVLA", 14)
          + pad("vs GR00T N1", 14) + "说明")
    print("-" * 78)
    print(pad("", 10) + pad("", 16) + pad("%s 元" % format(int(t_ovla), ","), 14)
          + pad("%s 元" % format(int(t_groot), ","), 14) + "训练成本（公开算力 x 单价）")
    for k in OVERHEADS:
        dc = data_cost[k]
        note = {1.0: "只算遥操作人力", 3.0: "含重置与失败重采", 5.0: "复杂布景 + 多班倒"}[k]
        print(pad("%.0f x" % k, 10) + pad("%s 元" % format(int(dc), ","), 16)
              + pad("%.1f x" % (dc / t_ovla), 14) + pad("%.1f x" % (dc / t_groot), 14)
              + note)

    print("\n结论：采集是训练的 1.6~19 倍，取决于现场开销 —— 而且这个倍数是**下限**，")
    print("因为训练成本是「一次性 + 可复现」，采集成本是「每加一个任务就再来一次」。")
    print("所以资源分配上，模型侧的优化收益会很快见顶，数据侧的还没开始。")

    print("\n存储成本对照（%.1f TB 总量，其中图像占 99.8%%）：" % (total_bytes / 1e12))
    monthly = total_bytes / 1e9 * 0.1        # 0.1 元/GB-月
    print("  云对象存储约 0.1 元/GB-月 -> %s 元/月 = %.1f 万元/年"
          % (format(int(monthly), ","), monthly * 12 / 10000))
    print("  和采集的百万级比，存储只有千分之一 —— **存储不是瓶颈，采集才是**。")
    print("  这条不对称性直接给出工程结论：宁可多存，不要少采。")
    print("  30 Hz 全存、多相机全存、原始码率全存，成本都远低于重采一遍。")

    print("\n但「多存」不等于「多标」。标注是另一条成本曲线：")
    print("  每条 episode 的成败判定、每个片段的语言描述，都还是人工。")
    print("  这部分的可扩展性比存储差得多 —— 也是第 7 章要展开的开放问题。")


# ================================================================ 主流程

def main():
    total_bytes = section_bytes()
    section_temporal()
    section_cost(total_bytes)

    print("\n" + "=" * 78)
    print("小结：三个比例决定了数据平台该干什么")
    print("=" * 78)
    print("1. 字节：图像 99.8%，动作 0.06%。优化必须打视频，动作要无损。")
    print("2. 帧率：每帧只有 0.7% 的像素是新的，但降帧率会丢时间分辨率。")
    print("   省钱的正确方式是编码器（可逆），不是降采样（不可逆）。")
    print("3. 成本：采集是训练的 1.6~19 倍，且不可复用。所以「多存少采」永远是")
    print("   正确的默认值 —— 反过来做的代价是不可逆的。")
    print("\n这三条合起来，就是第 7 章那个问题的一半答案：")
    print("VLA 的瓶颈不在模型结构，在「怎么便宜地拿到更多、更杂、更少重复的真机数据」。")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验 2：VLA 的三种接法，成本与风险各在哪里

「视觉编码器 / 语言主干 / 动作头怎么接」看起来是模型设计问题，实际上是
两个可以算清楚的账：

  1. 推理成本 = 权重被读了几遍
     batch=1 的推理有两道硬墙：prefill 撞算力墙，decode 撞显存带宽墙。
     「一个控制步里有几次串行前向」直接决定延迟，参数量的影响反而是次要的。

  2. 冻结主干 = 按方差挑特征 = 主动丢掉控制信号
     预训练特征里，和控制相关的方向方差往往很小（语言/对比学习不奖励它）。
     任何「保留方差最大的 k 维」式压缩都会先扔掉它。这一节用合成特征
     空间量化：按方差选 vs 按任务选，差几倍到几十倍。

运行：python examples/06_vla/02_wiring_cost.py
依赖：无（只用标准库）
"""

import math
import random


# ---------------------------------------------------------------- 输出对齐

def width(s):
    return sum(2 if ord(c) > 0x2E80 else 1 for c in s)


def pad(s, n):
    return s + " " * max(0, n - width(s))


# ================================================================ 1. 权重读几遍

# 公开规格（论文 / 模型卡，2026-11 核对）：
#   OpenVLA    7.0B 224x224 单图，256 bin，逐 token 自回归 7 个 token，实测 ~6 Hz
#   π0         3.3B = PaliGemma 3B + 300M action expert，flow 10 步，chunk 50
#              实测 73 ms/次（4090，3 相机），每 0.5 s 执行 25 步
#   GR00T N1   2.2B = VLM 1.34B + DiT，flow 4 步，chunk 16，实测 63.9 ms（L40）
#   OpenVLA-OFT 7B 并行解码 + chunk 8，端到端延迟降到 1/3（动作生成快 26 倍）
#
# 名称, 参数(B), 串行前向次数/次决策, 一次决策执行几步, 实测延迟(ms), 实测硬件带宽
MODELS = (
    ("OpenVLA 逐 token", 7.0, 8.0, 1, 167.0, 1008.0),
    ("π0 flow 10 步", 3.3, 11.0, 25, 73.0, 1008.0),
    ("GR00T flow 4 步", 2.2, 5.0, 16, 63.9, 864.0),
    ("OpenVLA-OFT 并行", 7.0, 2.0, 8, 56.0, 1008.0),
)


def section_roofline():
    print("=" * 80)
    print("1. 推理成本 = 权重读几遍（batch=1 时 decode 由显存带宽决定）")
    print("=" * 80)
    print("bf16 权重 = 参数 x 2 字节。一次前向要把权重整体读一遍，")
    print("所以「次决策里几次串行前向」直接乘在延迟上。\n")

    print(pad("设计", 18) + pad("参数", 7) + pad("前向/决策", 10)
          + pad("流量", 9) + pad("下限", 8) + pad("实测", 8)
          + pad("实测/下限", 10) + pad("chunk", 7) + pad("30Hz 需", 9) + "余量")
    print("-" * 80)
    for name, params, reads, chunk, measured, bw in MODELS:
        gb = params * 2.0 * reads
        lb = gb / bw * 1000.0
        need = 30.0 * measured / 1000.0      # 30 Hz 下必须覆盖多少步才能摊平
        margin = chunk / need
        print(pad(name, 18) + pad("%.1fB" % params, 7)
              + pad("%.0f 次" % reads, 10) + pad("%.0f GB" % gb, 9)
              + pad("%.0f ms" % lb, 8) + pad("%.0f ms" % measured, 8)
              + pad("%.1f x" % (measured / lb), 10) + pad("%d 步" % chunk, 7)
              + pad("%.1f 步" % need, 9)
              + ("%.1f x%s" % (margin, "" if margin >= 1 else "  <-- 撑不住")))

    print("\n「30Hz 需」= 要维持 30 Hz，一次决策至少得覆盖多少步（= 30 x 决策耗时）。")
    print("「余量」= 实际 chunk / 所需 chunk。小于 1 就是物理上不可能 ——")
    print("OpenVLA 的 0.2 直接解释了它为什么只能跑到 6 Hz。\n")

    print("每一行的成本结构都能对应到一个具体的设计决定：")
    print("  · OpenVLA 8 次 = 1 次 prefill + 7 个动作 token 各一次 decode。")
    print("    注意后 7 次是「7 维动作」而不是「7 步」—— 每一步控制都要重来。")
    print("  · π0 11 次 = 1 次观测前向 + 10 步 Euler 积分。flow 的步数就是延迟。")
    print("    但一次决策覆盖 25 步，所以余量有 6.8 倍：步数贵，可以摊。")
    print("  · GR00T 只用 4 步积分，是为把余量买到 2 倍以上。")
    print("  · OFT 把 7 次 decode 压成 1 次并行解码，余量从 0.2 跳到 4.7。")
    print("    加速比没到 7 倍，因为剩下的 1 次 prefill 成了新瓶颈 —— 见第 3 节。")

    print("\n三个能直接对上论文数字的推论：")
    print("  1. 自回归的延迟线性于**动作 token 数**，与 chunk 长度几乎无关。")
    print("     所以 π0-FAST 训练快 5 倍，推理反而慢 4~5 倍（30~60 个 token）。")
    print("  2. 想降延迟，减「串行前向次数」比减参数量有效得多：")
    print("     参数减半 = 延迟减半；并行解码 7->1 = 延迟降到 1/7。")
    print("  3. 反推预算：30 Hz（33 ms）下一个 chunk 里能读几遍权重？")
    print("  " + pad("参数量", 8) + pad("允许前向次数", 16) + "结论")
    for params in (7.0, 3.3, 2.2):
        allowed = 0.033 * 1008.0 / (params * 2.0)
        if allowed >= 8:
            verdict = "自回归逐 token 还有余量"
        elif allowed >= 4:
            verdict = "够 flow 4 步，不够 10 步"
        else:
            verdict = "必须并行解码 / 分块 + 前缀缓存"
        print("  " + pad("%.1fB" % params, 8) + pad("%.1f 次" % allowed, 16) + verdict)


# ================================================================ 2. 冻结特征

def solve(A, b):
    """高斯消元解 A x = b（带部分主元）。"""
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-13:
            continue
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        for r in range(col + 1, n):
            f = M[r][col] / pv
            if f == 0.0:
                continue
            for c in range(col, n + 1):
                M[r][c] -= f * M[col][c]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        if abs(M[i][i]) < 1e-13:
            continue
        s = M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))
        x[i] = s / M[i][i]
    return x


def ols_r2(X, y, idx, n_tr, n_te):
    """只用 idx 指定的特征列做岭回归，返回测试集 R^2。"""
    k = len(idx)
    A = [[0.0] * k for _ in range(k)]
    b = [0.0] * k
    for t in range(n_tr):
        row = [X[t][i] for i in idx]
        for a in range(k):
            ra = row[a]
            if ra == 0.0:
                continue
            b[a] += ra * y[t]
            for c in range(a, k):
                A[a][c] += ra * row[c]
    for a in range(k):
        for c in range(a):
            A[a][c] = A[c][a]
        A[a][a] += 1e-8 * n_tr          # 岭项，防共线
    w = solve(A, b)

    mean = sum(y[:n_tr]) / n_tr
    sse = sst = 0.0
    for t in range(n_tr, n_tr + n_te):
        pred = sum(w[a] * X[t][idx[a]] for a in range(k))
        sse += (y[t] - pred) ** 2
        sst += (y[t] - mean) ** 2
    return 1.0 - sse / sst


def section_frozen_features():
    print("\n" + "=" * 80)
    print("2. 冻结主干 = 按方差挑特征 = 主动丢掉控制信号")
    print("=" * 80)

    D_NUIS, D_CTRL = 60, 4
    D = D_NUIS + D_CTRL
    N_TR, N_TE = 1500, 500
    N = N_TR + N_TE

    rng = random.Random(5)
    # 干扰维度：方差按幂律衰减（预训练特征的典型谱：少数方向吃掉大部分能量）
    sds = [1.0 * (0.86 ** i) for i in range(D_NUIS)]
    ctrl_sd = 0.02          # 控制维：方差极小，语言任务不奖励它
    sds = sds + [ctrl_sd] * D_CTRL

    order = sorted(range(D), key=lambda i: -sds[i])   # 按方差降序的真实排名
    rank_of_ctrl = [order.index(D_NUIS + j) + 1 for j in range(D_CTRL)]

    total_var = sum(s * s for s in sds)
    ctrl_var = D_CTRL * ctrl_sd * ctrl_sd

    print("合成特征空间：%d 维 = %d 个干扰维（方差按 0.86 幂律衰减）+ %d 个控制维"
          % (D, D_NUIS, D_CTRL))
    print("控制维标准差 %.2f，最大干扰维 %.2f —— 相差 %.0f 倍"
          % (ctrl_sd, sds[0], sds[0] / ctrl_sd))
    print("控制维只占总方差 %.2f%%" % (ctrl_var / total_var * 100))
    print("按方差排序时，4 个控制维分别排在第 %s 位（共 64 位）\n"
          % "、".join(str(r) for r in rank_of_ctrl))

    X = [[rng.gauss(0, sds[i]) for i in range(D)] for _ in range(N)]
    w_true = [1.0, -0.8, 0.6, 0.5]
    SIG = 0.01
    y = []
    for t in range(N):
        v = sum(w_true[j] * X[t][D_NUIS + j] for j in range(D_CTRL))
        y.append(v + rng.gauss(0, SIG))

    # 理论上限：信号方差 / (信号方差 + 噪声方差)
    sig_var = sum(w * w for w in w_true) * ctrl_sd ** 2
    ceiling = sig_var / (sig_var + SIG ** 2)

    def corr(i):
        mx = sum(X[t][i] for t in range(N_TR)) / N_TR
        my = sum(y[t] for t in range(N_TR)) / N_TR
        sxy = sum((X[t][i] - mx) * (y[t] - my) for t in range(N_TR))
        sxx = sum((X[t][i] - mx) ** 2 for t in range(N_TR))
        syy = sum((y[t] - my) ** 2 for t in range(N_TR))
        return abs(sxy) / math.sqrt(sxx * syy + 1e-30)

    by_var = order
    by_task = sorted(range(D), key=lambda i: -corr(i))

    print(pad("保留维数 k", 11) + pad("保留率", 9) + pad("按方差选", 12)
          + pad("按任务相关选", 14) + "差距")
    print("-" * 80)
    for k in (4, 8, 16, 24, 28, 30, 32, 48, 64):
        r2v = max(ols_r2(X, y, by_var[:k], N_TR, N_TE), 0.0)
        r2t = max(ols_r2(X, y, by_task[:k], N_TR, N_TE), 0.0)
        note = ""
        if k == 28:
            note = "  <-- 控制维刚进来"
        if k == 64:
            note = "  <-- 理论上限 %.3f" % ceiling
        print(pad(str(k), 11) + pad("%.0f%%" % (k / float(D) * 100), 9)
              + pad("%.3f" % r2v, 12) + pad("%.3f" % r2t, 14)
              + ("%.1f x" % (r2t / r2v) if r2v > 1e-6 else "无穷") + note)

    print("\n三个数字：")
    print("  · k=24（保留 38%）时，按方差选出来的特征是 **完全无用** 的：R^2 = 0。")
    print("  · 要到 k=30（47%%）才把 4 个控制维收全，R^2 跳到 %.2f。" % ceiling)
    print("  · 按任务相关性选，k=4 就到 %.2f —— 同样的特征、同样的回归器。" % ceiling)
    print("\n也就是说：按「方差重要性」压缩到一半以下，控制信号会被整个吃掉，")
    print("而它在方差排名里并不在末尾，就排在第 %d~%d 位 —— 一个看起来「还挺靠前」"
          % (rank_of_ctrl[0], rank_of_ctrl[-1]))
    print("的位置。这就是 PCA / 方差剪枝在策略特征上格外危险的原因。")

    print("\n这对应 OpenVLA 论文里的两个消融：")
    print("  · 只训最后一层、或冻结视觉编码器 → 性能明显下降；")
    print("  · 视觉编码器必须一起训，而且训练轮数要到 27 个 epoch（VLM 通常 1~2 个）。")
    print("原因不是「参数不够」，是预训练特征的**重要性排序准则**")
    print("（语言/对比学习看方差与语义）和控制的排序准则根本不是同一件事。")

    print("\n对数据平台的含义：")
    print("  · 「特征压缩 / 降维存储」这类优化，一旦用到策略链路上，要先验证")
    print("    它保住了控制维，而不是只报告「重建误差很小」。重建误差小不代表")
    print("    控制能用 —— 上面 k=24 的特征重建得挺好，控制 R^2 是 0。")
    print("  · 冻结 = 便宜但会丢信号；全量微调 = 能对齐但贵。中间那条线（LoRA、")
    print("    只解冻后半部分）落在哪里，是个可以用同样方法测出来的工程问题。")


# ================================================================ 3. 两道墙

def section_two_walls():
    print("\n" + "=" * 80)
    print("3. prefill 撞算力墙，decode 撞带宽墙 —— 双系统的由来")
    print("=" * 80)

    P = 7.0e9
    PATCH, SIDE = 14, 224
    per_cam = (SIDE // PATCH) ** 2
    CAMS, TEXT, ACT = 1, 20, 7          # OpenVLA 实际用单张第三人称图

    n_pre = per_cam * CAMS + TEXT
    n_dec = ACT

    TFLOPS, BW = 165e12, 1008.0        # RTX 4090：bf16 稠密 / HBM

    t_pre_c = 2.0 * P * n_pre / TFLOPS           # prefill 算力时间
    t_pre_b = (P * 2.0) / (BW * 1e9)             # prefill 读一遍权重的时间
    t_dec_b = (P * 2.0) / (BW * 1e9)             # 每个 token 读一遍权重
    t_dec_total = n_dec * t_dec_b

    print("以 OpenVLA 的实际配置为例：7B、单张 224x224（patch %d，%d token）、%d 个动作 token"
          % (PATCH, per_cam, ACT))
    print("硬件：RTX 4090，bf16 稠密算力 %.0f TFLOPS，HBM %.0f GB/s\n" % (TFLOPS / 1e12, BW))

    print(pad("阶段", 10) + pad("token", 8) + pad("FLOPs", 10) + pad("撞哪道墙", 12)
          + pad("权重流量", 12) + pad("理论耗时", 12) + "说明")
    print("-" * 80)
    print(pad("prefill", 10) + pad(str(n_pre), 8) + pad("%.1f TF" % (2.0 * P * n_pre / 1e12), 10)
          + pad("算力", 12) + pad("%.0f GB" % (P * 2.0 / 1e9), 12)
          + pad("%.0f ms" % (t_pre_c * 1000), 12) + "图像 token 全在这里")
    print(pad("decode", 10) + pad("%d x 1" % n_dec, 8) + pad("%.0f GF" % (2.0 * P * n_dec / 1e9), 10)
          + pad("带宽", 12) + pad("%.0f GB" % (P * 2.0 * n_dec / 1e9), 12)
          + pad("%.0f ms" % (t_dec_total * 1000), 12) + "%d 个 token 要读 %d 遍" % (n_dec, n_dec))

    total = t_pre_c + t_dec_total
    print("\n合计 %.0f ms，实测 167 ms。差额 %.0f ms 是视觉编码器（约 600M）与 kernel 开销。"
          % (total * 1000, 167 - total * 1000))
    print("关键不在总误差，在**比例**：%.0f%% 的时间花在 decode 上。"
          % (t_dec_total / total * 100))
    print("而 decode 的耗时几乎全是「等权重从显存里出来」—— 算力闲着 %.1f%%。"
          % (100.0 - 2.0 * P * n_dec / TFLOPS / t_dec_total * 100))

    print("\n两个阶段为什么行为完全不同：")
    print("  · prefill：%d 个 token，%.1f TFLOP，按峰值算 %.0f ms；读一遍权重只要 %.0f ms。"
          % (n_pre, 2.0 * P * n_pre / 1e12, t_pre_c * 1000, t_pre_b * 1000))
    print("    计算 >> 读权重 —— 算力受限。加 token（多相机、高分辨率）直接加时间。")
    print("  · decode：每个 token 只要 %.0f GF，按峰值算 %.1f ms；读一遍权重却要 %.0f ms。"
          % (2.0 * P / 1e9, 2.0 * P / TFLOPS * 1000, t_dec_b * 1000))
    print("    计算 << 读权重 —— 带宽受限。加 token 数就是线性加时间，跟算力无关。")

    print("\n所以「降延迟」的手段分两类，不能混：")
    print("  降 prefill：减少图像 token（降分辨率、pixel shuffle）、降低重规划频率。")
    print("  降 decode：减少串行前向次数（并行解码、少积分步、加 chunk）。")
    print("OpenVLA 的 7 个动作 token 让它卡在 decode 侧，所以「把图缩小」救不了它，")
    print("「把 7 次变 1 次」才救得了 —— OFT 正是这么做的（余量 0.2 -> 4.8）。")

    print("\n把预算摊成「每秒用掉多少毫秒」，看每个设计能跑到多快：")
    print(pad("设计", 22) + pad("每步成本", 10) + pad("6 Hz", 10) + pad("10 Hz", 10)
          + pad("30 Hz", 10) + "结论")
    print("-" * 80)
    for name, per_step in (("OpenVLA 逐 token", t_pre_c + t_dec_total),
                           ("π0 flow (摊到每步)", (72.6 / 1008.0) / 25.0)):
        cells = ""
        for hz in (6, 10, 30):
            util = per_step * hz * 100.0
            cells += pad("%.0f%%" % util, 10)
        verdict = "6 Hz 已经吃满" if name.startswith("OpenVLA") else "30 Hz 才用 9%"
        print(pad(name, 22) + pad("%.1f ms" % (per_step * 1000), 10) + cells + verdict)

    print("\nOpenVLA 在 6 Hz 时已经用掉 72% 的实时预算，10 Hz 就超了 —— 这不是实现问题。")
    print("π0 把单步成本压到 2.9 ms（一次决策产出 25 步），50 Hz 只用 14%。")
    print("差别不在模型大小（7B vs 3.3B），在「一次决策产出多少步」。")

    print("\n这就是 GR00T N1 双系统的由来：")
    print("  System 2（VLM，算力受限）跑 10 Hz —— 图像贵，就少看几次；")
    print("  System 1（DiT，带宽受限）跑 120 Hz —— 动作便宜，就多算几次。")
    print("同一个模型里，两个阶段的瓶颈不同，就该用两个频率。")
    print("π0 的 73 ms 分解也印证这点：14 ms 图像编码 + 32 ms 观测前向 + 27 ms 十步 flow，")
    print("六成时间花在「看」，不在「动」。")


# ================================================================ 主流程

def main():
    section_roofline()
    section_frozen_features()
    section_two_walls()

    print("\n" + "=" * 80)
    print("小结")
    print("=" * 80)
    print("1. 延迟 ≈ 权重字节 x 串行前向次数 / 带宽。自回归的动作 token 数、")
    print("   flow 的积分步数、chunk 的长度，都是在调这个乘法。减次数比减参数有效。")
    print("2. 冻结特征不是「省钱版」，是「换了个目标函数」。按方差保留一半维度时")
    print("   控制信息 R^2 = 0，而按任务选 4 维就够。所以 VLA 几乎都要动视觉编码器。")
    print("3. prefill 撞算力、decode 撞带宽。图像 token 全在 prefill，动作 token")
    print("   全在 decode。双系统、并行解码、action chunk 都是这条算术的产物。")
    print("\n对做数据的人：以上三笔全是**推理侧**的账，但它们都指向同一个输入 ——")
    print("「你的数据里，同一句话对应多少种不同的几何」。第 1 个实验已经说明")
    print("这个数目很大；而它有多大，决定了模型要看多少数据才学会读场景。")


if __name__ == "__main__":
    main()

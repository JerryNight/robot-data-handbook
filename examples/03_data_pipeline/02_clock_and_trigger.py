#!/usr/bin/env python3
"""
实验 2：时钟漂移、时间戳抖动、PTP 偏移估计，以及一条硬件触发链的预算。

运行：python examples/03_data_pipeline/02_clock_and_trigger.py

这一篇要回答的是 02 章 [time-sync](../02-sensor-data/time-sync.md) 留下的那个问题：
对齐误差**已经**存在了，能不能事后补？答案是不能——但可以事前设计掉。

四个部分：
    [1] 漂移：两个晶振各差 20ppm，一小时之后错多少
    [2] 抖动：软件打戳和硬件打戳差一个量级，而且是**单边**的
    [3] PTP：随机抖动可以平均掉，路径不对称不行
    [4] 触发链：端到端不确定性的预算表，以及每一项该花多少钱
"""

import math
import random


# ---------------------------------------------------------------- 对齐

def width(s: str) -> int:
    return sum(2 if ord(c) > 0x2E80 else 1 for c in s)


def pad(s: str, n: int) -> str:
    return s + " " * max(0, n - width(s))


def head(*cells) -> str:
    return "  " + "".join(pad(c, w) for c, w in cells)


# ---------------------------------------------------------------- [1] 漂移

def part1_drift(drift_ppm=40.0):
    print("\n[1] 时钟漂移：ppm 是个很小的数，乘以时间就不小了")
    print("-" * 76)
    print("  普通晶振的标称精度是 ±20 ppm（百万分之二十）。")
    print("  两个独立时钟最坏情况相差 %.0f ppm = %.0e。" % (drift_ppm, drift_ppm * 1e-6))
    print()
    print("  %s%s%s" % (pad("持续时间", 14), pad("%.0fppm 累计偏差" % drift_ppm, 22), "对 30Hz 相机相当于"))
    print("  " + "-" * 68)
    for label, secs in (("1 秒", 1), ("1 分钟", 60), ("10 分钟", 600),
                        ("1 小时", 3600), ("8 小时工作日", 8 * 3600)):
        d = drift_ppm * 1e-6 * secs
        frames = d * 30
        print("  %s%s%s" % (pad(label, 14),
                            pad("%.3f ms" % (d * 1000) if d < 0.01
                                else "%.1f ms" % (d * 1000), 22),
                            "%.1f 帧" % frames if frames >= 0.1 else "可忽略"))
    print()
    print("  关键不是偏差大小，而是它**是斜坡，不是噪声**。")
    d8 = drift_ppm * 1e-6 * 8 * 3600
    print("  8 小时之后相机和关节的时钟差 %.3f 秒——这不是精度问题，是数据报废。"
          % d8 if d8 >= 0.5 else
          "  8 小时之后相机和关节的时钟差 %.0f ms（%.1f 帧）—— 好晶振只是把大问题变成中问题。"
          % (d8 * 1000, d8 * 30))
    print("  而且它不会自己回来：ppm 差是晶振的物理属性，恒定不变。")


# ---------------------------------------------------------------- [2] 抖动

def part2_jitter(tail_prob=0.02):
    print("\n[2] 软件打戳 vs 硬件打戳：差两个数量级，而且方向一致")
    print("-" * 76)

    rng = random.Random(7)
    n = 3000
    period = 1 / 30.0
    hw_err = []          # 硬件：触发到曝光只有固定延迟 + 极小时钟量化误差
    sw_err = []          # 软件：曝光 + 传输 + 驱动排队 + 用户态调度
    for i in range(n):
        t_true = i * period
        hw_err.append(rng.gauss(0, 0.00002) + 0.000_040)        # 40µs 固定延迟
        # 软件打戳：单边长尾，偶尔被调度器打断 10-20ms
        lat = rng.gauss(0.004, 0.0012) + 0.0025                 # 均值 6.5ms
        if rng.random() < tail_prob:
            lat += rng.uniform(0.008, 0.020)                    # 长尾：被调度器卡住
        sw_err.append(lat + rng.gauss(0, 0.00002))

    def stats(xs):
        m = sum(xs) / len(xs)
        var = sum((x - m) ** 2 for x in xs) / len(xs)
        s = sorted(abs(x) for x in xs)
        return m, math.sqrt(var), s[int(len(s) * 0.99)]

    for label, xs in (("硬件触发戳", hw_err), ("软件接收戳", sw_err)):
        m, sd, p99 = stats(xs)
        print("  %s  均值 %+8.3f ms   标准差 %6.3f ms   P99 %6.2f ms"
              % (pad(label, 12), m * 1000, sd * 1000, p99 * 1000))
    print()
    print("  两个都不能忽略的差别：")
    print("  · **均值**：软件戳有 6.5ms 的系统性滞后。如果你不做补偿，")
    print("    训练时会把它当成「观测晚于动作」的因果结构学进去。")
    print("  · **尾部**：2% 的帧被调度器卡到 10-20ms。均值看不出来，")
    print("    但它是训练集里最难处理的离群点——一帧错配污染一整个样本。")
    print()
    print("  所以时间戳必须在**采集端硬件**打点，越接近光子越准：")
    print("    曝光开始(硬件) < 帧起始 SOF(驱动) < 收到完整帧(用户态)")

    # 证明「均值消除不了」
    print("\n  一个常见误解：「多帧平均一下，滞后就抵消了」。")
    print("  对每个 N 随机抽 200 组，看「N 帧平均」这个估计量本身有多准：")
    print("  %s%s%s" % (pad("帧数 N", 10), pad("平均滞后", 14), "200 组之间的标准差"))
    print("  " + "-" * 56)
    for k in (1, 10, 100, 1000):
        means = []
        for _ in range(200):
            sub = rng.sample(sw_err, k)
            means.append(sum(sub) / k)
        m = sum(means) / len(means)
        sd = math.sqrt(sum((x - m) ** 2 for x in means) / len(means))
        print("  %s%s%s" % (pad(str(k), 10), pad("%.3f ms" % (m * 1000), 14),
                            "%.4f ms" % (sd * 1000)))
    print("  N 从 1 到 1000：标准差从 2.40 ms 缩到 0.06 ms（约 1/√N 的量级），")
    print("  但「平均滞后」一直停在 6.8 ms 附近 —— 偏差部分纹丝不动。")


# ---------------------------------------------------------------- [3] PTP

def part3_ptp():
    print("\n[3] PTP 的偏移估计：为什么随机抖动可以平均掉，路径不对称不行")
    print("-" * 76)
    print("  报文四次时间戳：")
    print("      master ──t1──────────────▶ slave       t2 = t1 + d_ms + offset")
    print("      master ◀──────────────t4── slave       t4 = t3 + d_sm - offset")
    print()
    print("    offset = ((t2 - t1) - (t4 - t3)) / 2")
    print("    delay  = ((t4 - t1) + (t3 - t2)) / 2")
    print()
    print("  注意 offset 的公式默认了 d_ms == d_sm。现实中它们不相等。")

    rng = random.Random(11)
    TRUE_OFFSET = 0.003_000        # slave 比 master 慢 3ms
    BASE_DELAY = 0.000_200         # 单向 200µs
    ASYM = 0.000_100               # 下行比上行慢 100µs（排队/交换机拥塞）

    for label, asym in (("理想对称", 0.0), ("下行多 100µs", ASYM)):
        est = []
        for _ in range(1000):
            d_ms = BASE_DELAY + asym + rng.gauss(0, 0.000_030)
            d_sm = BASE_DELAY + rng.gauss(0, 0.000_030)
            t1 = rng.uniform(0, 1)
            t2 = t1 + d_ms + TRUE_OFFSET
            t3 = t2 + 0.000_050                    # slave 处理耗时
            t4 = t3 + d_sm - TRUE_OFFSET
            est.append(((t2 - t1) - (t4 - t3)) / 2)

        print("\n  场景：%s（真实 offset = %.3f ms）" % (label, TRUE_OFFSET * 1000))
        print("  %s%s%s" % (pad("样本数", 10), pad("估计均值", 16), "估计标准差"))
        print("  " + "-" * 48)
        for k in (1, 10, 100, 1000):
            sub = est[:k]
            m = sum(sub) / k
            sd = math.sqrt(sum((x - m) ** 2 for x in sub) / k) if k > 1 else 0.0
            print("  %s%s%s" % (pad(str(k), 10),
                                pad("%+.4f ms" % (m * 1000), 16),
                                "%.4f ms" % (sd * 1000)))

    print("\n  两件事同时发生，但只有一件会变好：")
    print("  · 标准差随 √N 下降 —— 多测几次确实更准")
    print("  · 均值稳定偏 **+50µs** —— 100µs 的不对称被除以 2，永久留在结果里")
    print("  这是 PTP 的精度天花板：不是时钟精度，不是采样次数，")
    print("  而是「上下行路径有多不对称」。交换机排队、光模块收发延迟差、")
    print("  非对称的 PHY —— 全都在这个 50µs 里，而且测不出来。")


# ---------------------------------------------------------------- [4] 触发链

def part4_trigger_chain():
    print("\n[4] 一条触发链的预算：每一环抖多少，最后剩多少")
    print("-" * 76)
    print("  端到端不确定性 = 各环节误差的平方和开根（独立同分布假设下）。")
    print()
    # 单位统一为微秒；每一项都对应一个可以被单独测量的物理环节
    chain_a = [("相机曝光抖动", 2000), ("USB 传输排队", 3000),
               ("驱动到用户态", 1500), ("用户态处理", 500)]
    chain_b = [("相机曝光抖动", 2000), ("SOF→驱动读取", 200),
               ("主机时钟自身偏移", 500)]
    chain_c = [("触发信号抖动", 20), ("PTP 残余偏移", 50),
               ("相机内部延迟", 5)]

    for label, chain in (("A 全软件打戳（相机、关节都靠主机时钟）", chain_a),
                         ("B 驱动 SOF 打戳（固件给帧起始打点，仍用主机时钟）", chain_b),
                         ("C 硬件触发 + PTP（相机与关节共用同一触发源）", chain_c)):
        tot = math.sqrt(sum(v * v for _, v in chain))
        print("\n  %s：" % label)
        for k, v in chain:
            print("    %s%s µs" % (pad(k, 22), "%6d" % v))
        print("    %s%s µs  （= %.2f ms）" % (pad("合计（RMS）", 22), "%6.0f" % tot, tot / 1000))

    ta = math.sqrt(sum(v * v for _, v in chain_a))
    tc = math.sqrt(sum(v * v for _, v in chain_c))
    print("\n  C 相对 A 好了 %.0f 倍。这个差距直接决定了两件事：" % (ta / tc))
    print("    1. 你能不能做「动作与视觉毫秒级对应」的任务（插拔、装配、拧螺丝）")
    print("    2. 你的数据能不能拿去做精细力控 / 高频灵巧操作")
    print()
    print("  但 C 的代价是硬件的：触发线要走线、要留接口、机器人本体要改。")
    print("  一个务实的判断标准：")
    print("    · 任务周期 > 1s（抓取搬运、开抽屉）→ B 够用，误差 2ms 量级")
    print("    · 任务周期 100ms~1s（插拔、倒水）→ 必须 C")
    print("    · 只做语言条件的行为克隆（VLA 粗粒度）→ A 也能训，但别指望精细")


# ---------------------------------------------------------------- [5] 漂移能不能事后估
def part5_drift_estimate(drift_ppm=40.0, jitter_ms=2.5):
    """从一对时钟的观测里拟合相对漂移，看需要多长的窗口才够。

    这是对「漂移是斜坡，事后无法恢复」这句话的修正：斜坡**可以**事后拟合，
    只要你有足够的观测——问题在于「足够」是多长，以及抖动给斜率估计带来的
    误差有多大。
    """
    print("\n[5] 漂移能不能事后估出来：能，但要够长的窗口")
    print("-" * 76)
    print("  假设有 N 个共同事件（同一触发源、或通过内容匹配得到的对应点），")
    print("  每个事件的观测带 σ = %.1f ms 的抖动。拟合 t_slave = a·t_master + b，" % jitter_ms)
    print("  斜率 a 偏离 1 的量就是相对漂移。")
    print()
    print("  斜率估计的标准差 ≈ σ / (T · √(N/12))，T 是观测窗口长度。")
    print("  也就是说：**窗口越长、事件越多、抖动越小，斜率才估得准。**")
    print()

    rng = random.Random(23)
    print("  %s%s%s%s" % (pad("窗口 T", 12), pad("事件数 N", 12),
                        pad("一次估计的误差", 18), "折算成 ppm"))
    print("  " + "-" * 64)
    rows = []
    a_true = 1.0 + drift_ppm * 1e-6
    for T in (1.0, 10.0, 60.0, 600.0):
        N = int(30 * T)                      # 30 Hz 的共同事件
        ests = []
        for _ in range(200):
            xs = [rng.uniform(0, T) for _ in range(N)]
            ys = [a_true * x + rng.gauss(0, jitter_ms * 1e-3) for x in xs]
            mx = sum(xs) / N
            my = sum(ys) / N
            num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            den = sum((x - mx) ** 2 for x in xs)
            ests.append(num / den)
        # 报的是一次估计的标准差，不是 200 次平均后的偏差 ——
        # 平均会把误差再除以 √200，那不是「一次采集能有多准」的答案
        m = sum(ests) / len(ests)
        sd = math.sqrt(sum((e - m) ** 2 for e in ests) / len(ests))
        rows.append((T, N, sd * 1e6))
        print("  %s%s%s%s" % (pad("%g s" % T, 12), pad(str(N), 12),
                              pad("%.2e" % sd, 18), "± %.1f ppm" % (sd * 1e6)))

    print()
    print("  读法：真实漂移是 %.0f ppm。" % drift_ppm)
    print("  · 1 秒窗口：误差 ±%.0f ppm，是漂移本身的 %.0f 倍 —— 完全测不出" 
          % (rows[0][2], rows[0][2] / drift_ppm))
    print("  · 10 秒窗口：±%.0f ppm，仍然比漂移大 —— 只能说「可能有漂」"
          % rows[1][2])
    print("  · 60 秒：±%.1f ppm，漂移开始可辨" % rows[2][2])
    print("  · 600 秒：±%.2f ppm，能反过来校准" % rows[3][2])
    print()
    print("  这解释了一个常见现象：**「跑一小段验证一下」永远发现不了时钟问题**。")
    print("  短窗口里漂移被抖动完全淹没，看起来一切正常；等采到一小时，")
    print("  误差已经长到 144 ms，而这时数据已经录完了。")


# ---------------------------------------------------------------- [6] 不对称敏感性
def part6_asymmetry():
    print("\n[6] PTP 的残余偏移 = 不对称量的一半")
    print("-" * 76)
    print("  上一节的公式：offset = ((t2-t1) - (t4-t3)) / 2。")
    print("  如果下行比上行慢 δ，那么 (t2-t1) 里多了 δ、")
    print("  (t4-t3) 里少了 δ，两式相减再除以 2 —— 残余偏移正好是 δ/2。")
    print()
    print("  注意这个误差**与采样次数无关**：它不在噪声里，在公式的结构里。")
    print()
    print("  %s%s%s" % (pad("上下行不对称 δ", 18), pad("残余偏移", 14), "典型成因"))
    print("  " + "-" * 72)
    for d_us, cause in ((10, "同型号光模块，链路对称"),
                        (100, "交换机排队、收发延迟差"),
                        (1000, "非对称 PHY、上下行带宽不同")):
        print("  %s%s%s" % (pad("%d µs" % d_us, 18), pad("%d µs" % (d_us / 2), 14), cause))
    print()
    print("  所以 PTP 的定位要摆正：它把误差从「毫秒级」压到「微秒级」，")
    print("  但**天花板不是时钟精度，是链路对称性**。要亚微秒绝对对齐，")
    print("  只能放弃「测」，改成「共触发」—— 让两个传感器共用一根线。")


# ---------------------------------------------------------------- [7] 结论
def part7_conclusion():
    print("\n[7] 一张决策表")
    print("-" * 76)
    rows = [
        ("多机多传感器长时间采集", "PTP + 共触发源", "漂移是斜坡，事后无法恢复"),
        ("单机 10 分钟级采集", "驱动 SOF 打戳即可", "累计漂移 < 24ms，可接受"),
        ("相机与关节要对齐", "同一触发源，或 PTP", "软件戳的 6.5ms 滞后是系统性的"),
        ("相机与 IMU 要对齐", "IMU 戳 + 相机 SOF", "IMU 频率高，插值误差小"),
        ("跨天/跨场地数据合并", "必须绝对时间 + 时区记录", "否则 episode 顺序都排不出来"),
    ]
    print(head(("场景", 26), ("最低要求", 24), ("为什么", 0)))
    print("  " + "-" * 74)
    for a, b, c in rows:
        print("  %s%s%s" % (pad(a, 26), pad(b, 24), c))

    print("\n  最后一条最容易被忽略：跨天合并时，如果只有相对时间（从 0 开始），")
    print("  你连「这两批数据谁先录的」都答不上来，更别说对齐环境变化了。")
    print("  时间戳的存储格式必须是**绝对时间（ns since epoch）+ 时钟域标识**。")
    print("  " + "-" * 76)


if __name__ == "__main__":
    print("=" * 78)
    print("实验 2：时钟、抖动、PTP 与触发链的预算")
    print("=" * 78)
    part1_drift()
    part2_jitter()
    part3_ptp()
    part4_trigger_chain()
    part5_drift_estimate()
    part6_asymmetry()
    part7_conclusion()
    print("=" * 78)

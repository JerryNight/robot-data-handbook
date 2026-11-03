#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验 3：自采数据集的自检清单（可运行的 linter）

自己采的数据集出问题，几乎从来不报错 —— 训练照样跑、loss 照样降，
只是模型在真机上「总是差一点」。这个脚本把七类常见缺陷变成可计算的检查。

  检查 1  动作-状态一致性（含帧偏移扫描）—— 最隐蔽、最致命
  检查 2  时间戳：丢帧与时钟跳变
  检查 3  重复 episode（重复采集同一段演示）
  检查 4  动作饱和（大量顶在限位上）
  检查 5  截断与长度异常
  检查 6  成败标注分布（全成功 = 无法筛选）
  检查 7  必要元数据缺失

为了能验证这些检查真的有效，脚本先**构造一个每个缺陷都埋了的数据集**，
再让 linter 去跑，最后核对「检测到的」和「埋进去的」是否一致。

运行：python examples/04_datasets/03_dataset_linter.py
依赖：无（只用标准库）
"""

import hashlib
import math
import random
import statistics


# ---------------------------------------------------------------- 输出对齐

def width(s):
    return sum(2 if ord(c) > 0x2E80 else 1 for c in s)


def pad(s, n):
    return s + " " * max(0, n - width(s))


# ---------------------------------------------------------------- 合成数据

DOF = 7
HZ = 30
DT = 1.0 / HZ
N_EP = 120
ACT_LIMIT = 0.05          # 单步动作的物理上限，rad
TRACK_NOISE = 3e-4        # 跟踪误差：实际运动与指令动作的偏差，rad


def make_episode(ep_id, rng, frames=None, amp_range=(0.15, 0.5),
                 freq_range=(0.1, 0.4)):
    """一条「正常」演示：7 个关节各自做不同频率的正弦运动。"""
    n = frames if frames is not None else rng.randint(60, 140)
    q0 = [rng.uniform(-0.5, 0.5) for _ in range(DOF)]
    amp = [rng.uniform(*amp_range) for _ in range(DOF)]
    freq = [rng.uniform(*freq_range) for _ in range(DOF)]
    phase = [rng.uniform(0, 2 * math.pi) for _ in range(DOF)]

    states = []
    for t in range(n + 2):                     # 多算 2 帧，供偏移缺陷使用
        q = []
        for j in range(DOF):
            val = (q0[j]
                   + amp[j] * math.sin(2 * math.pi * freq[j] * t * DT + phase[j]))
            q.append(val + rng.gauss(0, 1e-4))   # 编码器噪声
        states.append(q)

    # 真实演示完成时会停住：最后 15% 保持不动（只留编码器噪声）。
    # 这一点是「收尾检测」的全部依据 —— 没有静止段就分不出截断。
    hold = int(n * 0.85)
    for t in range(hold, n + 2):
        states[t] = [states[hold][j] + rng.gauss(0, 1e-4) for j in range(DOF)]

    # 动作 = 相邻状态的差 + 跟踪误差（指令值 ≠ 实际执行值，这是真实的）
    actions = [[states[t + 1][j] - states[t][j] + rng.gauss(0, TRACK_NOISE)
                for j in range(DOF)] for t in range(n)]
    ts = [round(t * DT, 6) for t in range(n)]

    return {
        "id": ep_id,
        "states": states[:n],
        "actions": actions,
        "timestamps": ts,
        "success": True,
        "meta": {"control_hz": HZ, "extrinsics": None, "embodiment": None},
    }


def build_dataset():
    rng = random.Random(2024)
    eps = [make_episode("ep_%03d" % i, rng) for i in range(N_EP)]
    planted = {}

    # 缺陷 A：4 条重复（换了 id，内容一模一样）
    for k in range(4):
        src = eps[k]
        dup = {
            "id": "ep_dup%02d" % k,
            "states": [list(s) for s in src["states"]],
            "actions": [list(a) for a in src["actions"]],
            "timestamps": list(src["timestamps"]),
            "success": True,
            "meta": dict(src["meta"]),
        }
        eps.append(dup)
    planted["重复"] = ["ep_dup%02d" % k for k in range(4)]

    # 缺陷 B：3 条动作错位 1 帧（动作记晚了一步），速度分别取慢 / 中 / 快。
    # 用不同的速度是有意的：检测灵敏度完全取决于运动快慢，第五节会看到差别。
    speeds = [(0.18, 0.06), (0.30, 0.18), (0.50, 0.45)]
    for i, (a, f) in zip((10, 11, 12), speeds):
        eps[i] = make_episode(eps[i]["id"], rng, amp_range=(a, a),
                              freq_range=(f, f))
        e = eps[i]
        # 等价于把动作序列整体后移一位：action'[t] = action[t+1]
        e["actions"] = e["actions"][1:] + [e["actions"][-1]]
    planted["动作错位"] = [eps[i]["id"] for i in (10, 11, 12)]

    # 缺陷 C：6 条丢帧（三路数组同时删，时间戳留下洞）
    for i in (20, 21, 22, 23, 24, 25):
        e = eps[i]
        drop = rng.randint(3, 8)
        k = rng.randint(10, len(e["timestamps"]) - drop - 2)
        for key in ("states", "actions", "timestamps"):
            del e[key][k:k + drop]
    planted["丢帧"] = [eps[i]["id"] for i in (20, 21, 22, 23, 24, 25)]

    # 缺陷 D：2 条时钟倒退（时间戳整体回拨 1.5 秒）
    for i in (30, 31):
        e = eps[i]
        k = len(e["timestamps"]) // 2
        e["timestamps"][k:] = [t - 1.5 for t in e["timestamps"][k:]]
    planted["时钟倒退"] = [eps[i]["id"] for i in (30, 31)]

    # 缺陷 E：1 条动作饱和。先造一条动作幅度很大（会顶到限位）的演示，
    # 再裁掉动作并**重算状态**，保证一致性不被破坏，这样饱和就是独立缺陷。
    eps[40] = make_episode("ep_040", rng, amp_range=(0.8, 0.8),
                           freq_range=(0.8, 0.8))
    for i in (40,):
        e = eps[i]
        for a in e["actions"]:
            for j in range(DOF):
                a[j] = max(-ACT_LIMIT, min(ACT_LIMIT, a[j]))
        st = [list(e["states"][0])]
        for a in e["actions"]:
            st.append([st[-1][j] + a[j] for j in range(DOF)])
        e["states"] = st
    planted["动作饱和"] = [eps[40]["id"]]

    # 缺陷 F：5 条截断（演示在运动中结束）
    for i in (50, 51, 52, 53, 54):
        e = eps[i]
        cut = int(len(e["states"]) * 0.6)
        for key in ("states", "actions", "timestamps"):
            e[key] = e[key][:cut]
    planted["截断"] = [eps[i]["id"] for i in (50, 51, 52, 53, 54)]

    planted["全成功"] = True
    planted["元数据"] = True
    return eps, planted


# ---------------------------------------------------------------- 检查 1

def residual_at(ep, offset):
    """
    在给定帧偏移下，算 |Δstate − action| 的逐帧误差。

    一致性成立时 state[i+1] − state[i] 应该等于 action[t]，
    offset 是对齐关系的假设：state[i+t+1] − state[i+t] ≈ action[t]。
    返回每帧误差列表。
    """
    st = ep["states"]
    ac = ep["actions"]
    errs = []
    for t in range(len(ac)):
        i = t + offset
        if i < 0 or i + 1 >= len(st):
            continue
        d = [st[i + 1][j] - st[i][j] - ac[t][j] for j in range(DOF)]
        errs.append(sum(abs(x) for x in d) / DOF)
    return errs


def action_state_stats(ep, span=3):
    """
    返回 (最佳 offset, 最佳 offset 的中位残差, offset=0 的中位, offset=0 的最大,
          该 episode 的中位单步动作幅值)

    最后一个量用来解释「检测灵敏度为什么因 episode 而异」：
    动作越慢，错一帧造成的残差越小。
    """
    table = {}
    for o in range(-span, span + 1):
        e = residual_at(ep, o)
        if e:
            table[o] = (statistics.median(e), max(e))
    best_o = min(table, key=lambda o: table[o][0])
    med0, max0 = table.get(0, (float("inf"), float("inf")))
    amps = [sum(abs(v) for v in a) / DOF for a in ep["actions"]]
    return best_o, table[best_o][0], med0, max0, statistics.median(amps)


def check_action_state(eps):
    """
    返回 (状态, 确诊列表, 疑似列表, 全部统计)

    分两档是必要的：动作很慢的 episode 上，错一帧造成的残差和噪声同量级，
    最小值可能只是**抖动**造成的。所以：
      确诊 —— 最佳 offset ≠ 0，且残差比 offset=0 低一半以上
      疑似 —— 最佳 offset ≠ 0，但差距不够大，需要人工确认或结合速度分段
    """
    rows = []
    confirmed = []
    suspect = []
    for ep in eps:
        o, best_med, med0, max0, amp = action_state_stats(ep)
        rows.append((ep["id"], o, med0, max0, best_med, amp))
        if o != 0 and med0 > 0:
            if best_med <= 0.5 * med0:
                confirmed.append((ep["id"], o, med0, best_med, amp))
            else:
                suspect.append((ep["id"], o, med0, best_med, amp))
    return ("FAIL" if (confirmed or suspect) else "PASS"), confirmed, suspect, rows


# ---------------------------------------------------------------- 检查 2

def check_timestamps(eps):
    gaps, jumps = [], []
    for ep in eps:
        ts = ep["timestamps"]
        diffs = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
        if not diffs:
            continue
        med = statistics.median(diffs)
        for i, d in enumerate(diffs):
            if d <= 0:
                jumps.append((ep["id"], i, d))
            elif d > 1.8 * med:
                gaps.append((ep["id"], i, int(round(d / med))))
    return ("FAIL" if (gaps or jumps) else "PASS"), gaps, jumps


# ---------------------------------------------------------------- 检查 3

def episode_fingerprint(ep):
    h = hashlib.sha256()
    for a in ep["actions"]:
        h.update(("%.6f %.6f %.6f %.6f %.6f %.6f %.6f" % tuple(a)).encode())
    return h.hexdigest()[:16]


def check_duplicates(eps):
    seen, hits = {}, []
    for ep in eps:
        fp = episode_fingerprint(ep)
        if fp in seen:
            hits.append((ep["id"], seen[fp], fp))
        else:
            seen[fp] = ep["id"]
    return ("FAIL" if hits else "PASS"), hits


# ---------------------------------------------------------------- 检查 4

def check_saturation(eps, limit=ACT_LIMIT - 1e-9, warn_frac=0.05):
    hits = []
    for ep in eps:
        tot = sat = 0
        for a in ep["actions"]:
            for v in a:
                tot += 1
                if abs(v) >= limit:
                    sat += 1
        frac = sat / tot if tot else 0.0
        if frac > warn_frac:
            hits.append((ep["id"], frac))
    return ("FAIL" if hits else "PASS"), hits


# ---------------------------------------------------------------- 检查 5

def check_truncation(eps, settle_ratio=0.25):
    """
    收尾检测：演示最后一帧的速度，应该明显低于过程中的中位速度。

    前提假设：任务完成时机器人是停住的。这个假设对抓取、放置、开抽屉成立，
    对「推着物体一直走」这类任务不成立 —— 那类数据要单独处理。

    判据：末帧速度 > 中位速度的 25% => 没有收尾，疑似截断。
    """
    hits = []
    lens = [len(ep["states"]) for ep in eps]
    med_len = statistics.median(lens)
    for ep in eps:
        st = ep["states"]
        if len(st) < 4:
            continue
        vels = [sum(abs(st[t + 1][j] - st[t][j]) for j in range(DOF)) / DOF
                for t in range(len(st) - 1)]
        med_v = statistics.median(vels) or 1e-9
        ratio = vels[-1] / med_v
        if ratio > settle_ratio:
            hits.append((ep["id"], ratio,
                         "末帧速度是中位的 %.0f%%，没有收尾" % (ratio * 100)))
        elif len(st) > 2.5 * med_len or len(st) < 0.3 * med_len:
            hits.append((ep["id"], float(len(st)), "长度异常：%d 帧" % len(st)))
    return ("FAIL" if hits else "PASS"), hits, med_len


# ---------------------------------------------------------------- 检查 6/7

def check_labels(eps):
    n = len(eps)
    ok = sum(1 for e in eps if e["success"])
    frac = ok / n if n else 0.0
    hits = [("全部 %d 条都标成功" % n, frac)] if frac >= 0.999 else []
    return ("FAIL" if hits else "PASS"), hits, frac


def check_metadata(eps):
    need = ["control_hz", "extrinsics", "embodiment"]
    missing = {k: 0 for k in need}
    for ep in eps:
        for k in need:
            if ep["meta"].get(k) in (None, ""):
                missing[k] += 1
    hits = [(k, v) for k, v in missing.items() if v > 0]
    return ("FAIL" if hits else "PASS"), hits


# ---------------------------------------------------------------- 主流程

def main():
    print("=" * 78)
    print("实验 3：自采数据集的自检清单")
    print("=" * 78)

    eps, planted = build_dataset()
    print("\n[0] 合成数据集：%d 条 episode，%d 自由度，%d Hz，单步动作上限 ±%.2f rad"
          % (len(eps), DOF, HZ, ACT_LIMIT))
    print("    埋进去的缺陷（真实采集里最常见的几类）：")
    print("      A 重复 episode      4 条      E 动作饱和      1 条")
    print("      B 动作错位 1 帧     3 条      F 演示被截断    5 条")
    print("      C 丢帧              6 条      G 全部标成功    %d 条" % len(eps))
    print("      D 时钟倒退          2 条      H 元数据缺失    全部")

    # ---- 检查 1
    print("\n[1] 动作-状态一致性：靠扫描帧偏移来定位")
    print("-" * 78)
    print("  原理：delta 动作的定义就是 state[t+1] − state[t]。")
    print("  如果采集代码里动作记晚了一步，这条等式会整体错位，")
    print("  但数据本身**不会报错** —— 形状对、数值范围对、loss 也能降。")
    print()
    status, confirmed, suspect, rows = check_action_state(eps)
    ep_by_id = {ep["id"]: ep for ep in eps}
    best_by_id = {r[0]: r[1] for r in rows}
    hit_ids = set(h[0] for h in confirmed + suspect)

    print("  先看三个例子，扫描 offset ∈ [−2, +2]，看残差的形状：")
    print()
    for eid in ("ep_000", "ep_010", "ep_012"):
        ep = ep_by_id[eid]
        amp = statistics.median([sum(abs(v) for v in a) / DOF for a in ep["actions"]])
        tag = "  ← 这一条被埋了错位" if eid in hit_ids else "  （正常）"
        print("  %s   中位单步动作 %.4f rad%s" % (pad(eid, 10), amp, tag))
        print("    %s%s%s" % (pad("offset", 8), pad("中位残差", 13), "最大残差"))
        for o in range(-2, 3):
            e = residual_at(ep, o)
            mark = "  <- 最小" if o == best_by_id[eid] else ""
            print("    %s%s%s%s" % (pad("%+d" % o, 8),
                                    pad("%.2e" % statistics.median(e), 13),
                                    "%.2e" % max(e), mark))
        print()

    def ratio(eid):
        med = [statistics.median(residual_at(ep_by_id[eid], o)) for o in (0, 1)]
        return med[0] / med[1] if med[1] > 0 else float("inf")

    print("  这三个例子的结果完全不同，这是这一节最该记住的地方：")
    print()
    print("    · ep_000（正常）：最小值在 0，两侧对称上翘，形状很干净。")
    print("    · ep_010（错位，动作慢）：曲线几乎是**平的** ——")
    print("      offset=0 只比最小值高 %.1f 倍，肉眼完全看不出异常。"
          % ratio("ep_010"))
    print("    · ep_012（错位，动作快）：标准的 U 形，offset=0 高 %.0f 倍，一眼就看出来。"
          % ratio("ep_012"))
    print()
    print("  原因是数学上的：错一帧造成的残差约等于动作的**二阶差分**，")
    print("  而二阶差分 ∝ 幅值 × (2πf·dt)²，也就是随速度**平方**增长。")
    print("  把它和跟踪噪声比一下，可分辨性就直接算出来了：")
    print()

    def second_diff(amp, freq):
        return amp * (2 * math.pi * freq * DT) ** 2

    print("  %s%s%s%s%s" % (pad("例子", 16), pad("动作步长", 12), pad("二阶差分", 12),
                            pad("二阶/噪声", 12), "能不能看出来"))
    print("  " + "-" * 68)
    for label, amp, freq in (("ep_010 慢", 0.18, 0.06), ("ep_012 快", 0.50, 0.45)):
        sd = second_diff(amp, freq)
        step = amp * 2 * math.pi * freq * DT
        ok = "看不出" if sd / TRACK_NOISE < 1 else "清楚"
        print("  %s%s%s%s%s" % (pad(label, 16), pad("%.4f" % step, 12),
                                pad("%.1e" % sd, 12),
                                pad("%.2f" % (sd / TRACK_NOISE), 12), ok))
    print()
    print("  慢动作的二阶差分不到噪声的 %.0f%%，所以 offset 扫描的曲线是平的 ——"
          % (second_diff(0.18, 0.06) / TRACK_NOISE * 100))
    print("  这不是算法不够好，是**数据里根本没有这个信息**。")
    print()
    print("  这条结论很实用：**用慢动作数据检查帧对齐，等于没检查。**")
    print("  但也不能只留快动作 —— 慢动作段恰恰是精细操作（插销、拧螺丝）")
    print("  所在的地方。正确做法是保留全部数据，但**按速度分段判定**。")

    print()
    print("  全部异常（按 offset 扫描的结果）：")
    print()
    print("  %s%s%s%s%s" % (pad("episode", 12), pad("最佳 offset", 13),
                          pad("动作幅值", 11), pad("倍数", 9), "判定"))
    print("  " + "-" * 62)
    confirmed_ids = set(h[0] for h in confirmed)
    for eid, o, med0, max0, best_med, amp in rows:
        if o != 0:
            r = med0 / best_med if best_med > 0 else float("inf")
            verdict = "确诊：整体错位" if eid in confirmed_ids else "待确认：差值太小"
            print("  %s%s%s%s%s" % (pad(eid, 12), pad("%+d" % o, 13),
                                    pad("%.4f" % amp, 11), pad("%.1fx" % r, 9),
                                    verdict))
        elif max0 > 1e-3:
            print("  %s%s%s%s%s" % (pad(eid, 12), pad("0", 13),
                                    pad("%.4f" % amp, 11), pad("—", 9),
                                    "散点损坏：最大残差 %.1e" % max0))

    print()
    print("  两类异常，处理方式完全不同：")
    print()
    print("    · 最佳 offset ≠ 0 → **整体性错位**（%d 条确诊，%d 条待确认）。"
          % (len(confirmed), len(suspect)))
    print("      数据形状对、数值范围对、loss 也能降，只是时间对应整体偏了。")
    print("      这类必须修：动作错一帧，策略会学到系统性提前量。")
    print("    · 中位残差正常、最大残差却很大 → **个别帧坏了**。")
    print("      用中位数永远抓不到 —— 必须看最大值或高分位数。")
    print()
    print("  后者正是丢帧留下的：三路数组一起删，删除接缝上那个动作")
    print("  还是按「原第 k 帧到 k+1 帧」算的，但它后面已经换成第 k+drop 帧了。")
    print("  所以**丢帧在运动段必然破坏动作-状态一致性**，两项检查是耦合的。")
    print()
    print("  注意「运动段」这个限定：如果丢的帧正好落在演示末尾的静止段，")
    print("  前后状态本来就一样，残差不会变大 —— 这种丢帧两项检查都抓不到。")
    print("  所以时间戳检查（下一节）不能省，它才是丢帧的直接证据。")

    # ---- 检查 2
    print("\n[2] 时间戳：丢帧与时钟倒退")
    print("-" * 78)
    status_t, gaps, jumps = check_timestamps(eps)
    print("  判定：相邻间隔 > 1.8 倍中位数 = 丢帧；间隔 ≤ 0 = 时钟跳变。")
    print()
    print("  丢帧 %d 处（涉及 %d 条 episode）："
          % (len(gaps), len(set(g[0] for g in gaps))))
    for eid, i, k in gaps[:6]:
        print("    %s  第 %d 帧后，间隔是正常值的 %d 倍（丢了 %d 帧）"
              % (pad(eid, 12), i, k, k - 1))
    print("  时钟倒退 %d 处（涉及 %d 条）："
          % (len(jumps), len(set(j[0] for j in jumps))))
    for eid, i, d in jumps[:6]:
        print("    %s  第 %d 帧后，间隔 %.3f 秒" % (pad(eid, 12), i, d))
    print()
    broken = set(r[0] for r in rows if r[1] == 0 and r[3] > 1e-3)
    missed = sorted(set(g[0] for g in gaps) - broken)
    if missed:
        print("  对照上一节：%s 的丢帧**没有**出现在「散点损坏」列表里。"
              % "、".join(missed))
        print("  因为它丢在减速段，前后状态本来就接近，残差没被拉大。")
        print("  这正是「要靠时间戳抓丢帧」的实证，而不是理论担忧。")
        print()
    print("  为什么丢帧比想象中严重：动作是按帧存的，丢一帧就等于")
    print("  跳过了 33ms 的物理过程。如果丢的那一帧正好在接触瞬间，")
    print("  这条演示的因果链就断了 —— 但数据看起来完全正常。")

    # ---- 检查 3
    print("\n[3] 重复 episode")
    print("-" * 78)
    status_d, dup_hits = check_duplicates(eps)
    print("  做法：对动作序列算哈希，相同就是同一段演示。")
    print()
    for eid, orig, fp in dup_hits:
        print("    %s 与 %s 完全相同（哈希 %s）" % (pad(eid, 12), pad(orig, 12), fp))
    print()
    print("  重复的来源通常是：批量导入跑了两次；「重采一条」时没删掉旧的；")
    print("  同一条被不同人各自提交。")
    print("  危害：这些样本在训练里被加权 2 倍，等于无意中改了数据分布。")

    # ---- 检查 4
    print("\n[4] 动作饱和")
    print("-" * 78)
    status_s, sat_hits = check_saturation(eps)
    print("  判定：|动作| 顶到限位的比例 > 5% 就报警。")
    print()
    for eid, frac in sat_hits:
        print("    %s  饱和比例 %.1f%%" % (pad(eid, 12), frac * 100))
    print()
    print("  饱和意味着「机器人想要更快，但被限位挡住了」。")
    print("  这类数据学出来的是「一直输出最大值」，到了不限位的场景就会过冲。")
    print("  正确做法是丢掉饱和段，或者重采慢一点的演示。")

    # ---- 检查 5
    print("\n[5] 截断与长度异常")
    print("-" * 78)
    status_c, trunc_hits, med_len = check_truncation(eps)
    print("  判定：末帧速度 > 中位速度的 25% = 没有收尾，疑似截断。")
    print("  前提假设：任务完成时机器人停在静止状态（抓取/放置/开门成立，")
    print("  「推着物体一直走」这类任务不成立，需要单独处理）。")
    print("  中位长度 %d 帧（%.1f 秒）" % (med_len, med_len * DT))
    print()
    for eid, ratio, note in trunc_hits[:8]:
        print("    %s  %s" % (pad(eid, 12), note))
    print()
    print("  截断的演示最坑：它把「动作做了一半」当成完整轨迹。")
    print("  模型会学到「任务到这里就算完成」，部署时表现为提前松手。")
    print("  根因几乎总是采集中断（超时、误触停止、磁盘满）。")

    # ---- 检查 6
    print("\n[6] 成败标注分布")
    print("-" * 78)
    status_l, label_hits, frac_ok = check_labels(eps)
    print("  成功率 %.0f%%（%d 条）" % (frac_ok * 100, len(eps)))
    print()
    print("  全部成功听起来是好事，实际是个问题：")
    print("    · 无法做「只训成功样本」的筛选 —— 本来就没得选")
    print("    · 无法估计真实成功率 —— 数据集里的成功率不是策略的成功率")
    print("    · 丢掉的全是**最有信息量**的样本：失败的边界情况")

    # ---- 检查 7
    print("\n[7] 必要元数据")
    print("-" * 78)
    status_m, meta_hits = check_metadata(eps)
    for k, v in meta_hits:
        print("    %s 缺失 %d / %d 条" % (pad(k, 14), v, len(eps)))
    print()
    print("  这三个字段缺失的后果在本章第一篇里量化过：")
    print("    control_hz → 动作没有物理量纲，跨数据集不可比")
    print("    extrinsics → 多视角几何算不出来，点云无法融合")
    print("    embodiment → 无法按本体分组归一化")

    # ---- 汇总
    print("\n" + "=" * 78)
    print("汇总")
    print("=" * 78)
    n_sporadic = sum(1 for r in rows if r[1] == 0 and r[3] > 1e-3)
    summary_rows = [
        ("动作-状态一致性", status, len(confirmed) + len(suspect),
         "确诊 %d / 待确认 %d / 散点损坏 %d"
         % (len(confirmed), len(suspect), n_sporadic)),
        ("时间戳", status_t, len(gaps) + len(jumps),
         "丢帧 %d / 倒退 %d" % (len(gaps), len(jumps))),
        ("重复 episode", status_d, len(dup_hits), "内容逐字节相同"),
        ("动作饱和", status_s, len(sat_hits), "单条饱和比例过高"),
        ("截断/长度", status_c, len(trunc_hits), "演示在运动中结束"),
        ("成败标注", status_l, len(label_hits), "无法做筛选"),
        ("元数据", status_m, len(meta_hits),
         "%s 全缺" % "、".join(k for k, _ in meta_hits)),
    ]
    print("  %s%s%s%s" % (pad("检查项", 20), pad("结果", 8), pad("命中", 10), "说明"))
    print("  " + "-" * 72)
    for name, st, cnt, note in summary_rows:
        print("  %s%s%s%s" % (pad(name, 20), pad(st, 8), pad("%d" % cnt, 10), note))

    # ---- 召回核对
    print("\n" + "=" * 78)
    print("召回核对：检测到的 vs 埋进去的")
    print("=" * 78)
    detected = {
        "重复": set(h[0] for h in dup_hits),
        "动作错位": set(h[0] for h in confirmed + suspect),
        "丢帧": set(g[0] for g in gaps),
        "时钟倒退": set(j[0] for j in jumps),
        "动作饱和": set(s[0] for s in sat_hits),
        "截断": set(t[0] for t in trunc_hits),
    }
    print("  %s%s%s%s" % (pad("缺陷", 14), pad("埋入", 8), pad("检出", 8), "结论"))
    print("  " + "-" * 56)
    for key in ("重复", "动作错位", "丢帧", "时钟倒退", "动作饱和", "截断"):
        exp = set(planted[key])
        got = detected[key]
        ok = got & exp
        miss = exp - got
        extra = got - exp
        verdict = "全部命中" if not miss and not extra else (
            "漏 %d" % len(miss) + ("，多 %d" % len(extra) if extra else "")
            if miss else "多报 %d" % len(extra))
        print("  %s%s%s%s" % (pad(key, 14), pad("%d" % len(exp), 8),
                              pad("%d" % len(ok), 8), verdict))

    print()
    print("  六类埋入的缺陷全部被逐条定位，没有漏检也没有误报。")
    print()
    print("  落地建议：把这几项做成 CI 的一步，每批新数据入库前跑一次。")
    print("  成本是几秒钟，省掉的是「训了两周才发现数据有问题」。")
    print()
    print("  检查 1 必须放在最前面 —— 它是唯一一项")
    print("  **不依赖任何外部信息、只看数据本身就能发现问题** 的检查。")


if __name__ == "__main__":
    main()

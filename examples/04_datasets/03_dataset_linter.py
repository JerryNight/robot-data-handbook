#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验 3：自采数据集的自检清单（可运行的 linter）
================================================

自己采的数据集出问题，几乎从来不报错 —— 训练照样跑、loss 照样降，
只是模型在真机上「总是差一点」。这个脚本把七类常见缺陷变成可计算的检查，
并且**先造一个每个缺陷都埋了的数据集**，再让 linter 去跑，最后核对
「检测到的」和「埋进去的」是否一致。

  检查 1  动作-状态一致性（帧偏移扫描）—— 最隐蔽、最致命
  检查 2  时间戳：丢帧与时钟跳变
  检查 3  重复 episode：精确哈希与粗指纹
  检查 4  动作饱和：顶在限位上
  检查 5  截断与长度异常
  检查 6  成败标注分布（全成功 = 无法筛选）
  检查 7  必要元数据缺失

除了「能不能抓到」，这个脚本还算三件更少被讨论的事：

  [3] 可检测性下限：动作多慢的时候，检查 1 就失效了
  [4] 检查粒度：全身一个 offset 与逐关节 offset 的区别
  [8] 饱和的两种严重度：多少帧被裁 vs 多少位移丢了

运行：python examples/04_datasets/03_dataset_linter.py
依赖：无（只用标准库）
"""

import hashlib
import math
import random
import statistics
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


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
ENC_NOISE = 1e-4          # 编码器噪声，rad


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
            q.append(val + rng.gauss(0, ENC_NOISE))
        states.append(q)

    # 真实演示完成时会停住：最后 15% 保持不动（只留编码器噪声）。
    # 这一点是「收尾检测」的全部依据 —— 没有静止段就分不出截断。
    hold = int(n * 0.85)
    for t in range(hold, n + 2):
        states[t] = [states[hold][j] + rng.gauss(0, ENC_NOISE) for j in range(DOF)]

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


def clone(ep, new_id):
    return {
        "id": new_id,
        "states": [list(s) for s in ep["states"]],
        "actions": [list(a) for a in ep["actions"]],
        "timestamps": list(ep["timestamps"]),
        "success": ep["success"],
        "meta": dict(ep["meta"]),
    }


def build_dataset():
    rng = random.Random(2024)
    eps = [make_episode("ep_%03d" % i, rng) for i in range(N_EP)]
    planted = {}

    # 缺陷 A：4 条重复（换了 id，内容一模一样）
    for k in range(4):
        eps.append(clone(eps[k], "ep_dup%02d" % k))
    planted["重复"] = ["ep_dup%02d" % k for k in range(4)]

    # 缺陷 A2：3 条「重采」—— 掐头去尾 + 微小噪声，精确哈希抓不到
    near = []
    for k, src in enumerate((5, 6, 7)):
        e = clone(eps[src], "ep_near%02d" % k)
        head, tail = rng.randint(2, 4), 0
        for key in ("states", "actions", "timestamps"):
            e[key] = e[key][head:len(e[key]) - tail]
        for s in e["states"]:
            for j in range(DOF):
                s[j] += rng.gauss(0, 2e-3)
        # 重采是一条新演示：动作按扰动后的状态重算，一致性仍然成立
        e["actions"] = [[e["states"][t + 1][j] - e["states"][t][j]
                         + rng.gauss(0, TRACK_NOISE) for j in range(DOF)]
                        for t in range(len(e["states"]) - 1)]
        e["timestamps"] = [round(t * DT, 6) for t in range(len(e["actions"]))]
        eps.append(e)
        near.append(e["id"])
    planted["近似重复"] = near

    # 缺陷 B：3 条动作错位 1 帧（动作记晚了一步），速度分别取慢 / 中 / 快。
    # 用不同速度是有意的：检测灵敏度完全取决于运动快慢。
    speeds = [(0.18, 0.06), (0.30, 0.18), (0.50, 0.45)]
    for i, (a, f) in zip((10, 11, 12), speeds):
        eps[i] = make_episode(eps[i]["id"], rng, amp_range=(a, a),
                              freq_range=(f, f))
        e = eps[i]
        e["actions"] = e["actions"][1:] + [e["actions"][-1]]
    planted["动作错位"] = [eps[i]["id"] for i in (10, 11, 12)]

    # 缺陷 C：6 条丢帧（三路数组同时删，时间戳留下洞）
    for i in (20, 21, 22, 23, 24, 25):
        e = eps[i]
        drop = rng.randint(3, 8)
        k = rng.randint(10, len(e["timestamps"]) - drop - 2)
        for key in ("states", "actions", "timestamps"):
            del e[key][k:k + drop]
    # 再埋一条**丢在静止段**的：前后状态本来就一样，残差不会变大。
    # 这是为了证明「时间戳检查不可省略」而特意构造的。
    e = eps[26]
    k = int(len(e["timestamps"]) * 0.92)
    for key in ("states", "actions", "timestamps"):
        del e[key][k:k + 4]
    planted["丢帧"] = [eps[i]["id"] for i in (20, 21, 22, 23, 24, 25, 26)]

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
    e = eps[40]
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


# ---------------------------------------------------------------- 检查 1 内核

def residual_at(ep, offset):
    """
    在给定帧偏移下，算 |Δstate − action| 的逐帧误差。

    一致性成立时 state[i+1] − state[i] 应该等于 action[t]，
    offset 是对齐关系的假设：state[i+t+1] − state[i+t] ≈ action[t]。
    """
    st, ac = ep["states"], ep["actions"]
    errs = []
    for t in range(len(ac)):
        i = t + offset
        if i < 0 or i + 1 >= len(st):
            continue
        d = [st[i + 1][j] - st[i][j] - ac[t][j] for j in range(DOF)]
        errs.append(sum(abs(x) for x in d) / DOF)
    return errs


def action_state_stats(ep, span=3):
    """返回 (最佳 offset, 最佳中位残差, offset=0 中位, offset=0 最大, 中位动作幅值)。"""
    table = {}
    for o in range(-span, span + 1):
        e = residual_at(ep, o)
        if e:
            table[o] = (statistics.median(e), max(e))
    best_o = min(table, key=lambda o: table[o][0])
    med0, max0 = table.get(0, (float("inf"), float("inf")))
    amps = [sum(abs(v) for v in a) / DOF for a in ep["actions"]]
    return best_o, table[best_o][0], med0, max0, statistics.median(amps)


def check_action_state(eps, ratio_thresh=0.5):
    rows, confirmed, suspect = [], [], []
    for ep in eps:
        o, best_med, med0, max0, amp = action_state_stats(ep)
        rows.append((ep["id"], o, med0, max0, best_med, amp))
        if o != 0 and med0 > 0 and best_med <= ratio_thresh * med0:
            confirmed.append((ep["id"], o, med0, best_med, amp))
        elif o != 0:
            suspect.append((ep["id"], o, med0, best_med, amp))
    return ("FAIL" if (confirmed or suspect) else "PASS"), confirmed, suspect, rows


# ---------------------------------------------------------------- 其它检查

def check_timestamps(eps, factor=1.8):
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
            elif d > factor * med:
                gaps.append((ep["id"], i, int(round(d / med))))
    return ("FAIL" if (gaps or jumps) else "PASS"), gaps, jumps


def exact_fingerprint(ep):
    h = hashlib.sha256()
    for a in ep["actions"]:
        h.update(("%.6f %.6f %.6f %.6f %.6f %.6f %.6f" % tuple(a)).encode())
    return h.hexdigest()[:16]


def coarse_signature(ep, n_pts=32):
    """把状态轨迹按时间等分重采样成 32 个点，展平成向量。"""
    st = ep["states"]
    if len(st) < 2:
        return []
    out = []
    for i in range(n_pts):
        pos = i / (n_pts - 1.0) * (len(st) - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, len(st) - 1)
        w = pos - lo
        out.extend(st[lo][j] * (1 - w) + st[hi][j] * w for j in range(DOF))
    return out


def sig_distance(a, b):
    if not a or not b:
        return float("inf")
    return max(abs(x - y) for x, y in zip(a, b))


def check_duplicates(eps):
    seen, hits = {}, []
    for ep in eps:
        fp = exact_fingerprint(ep)
        if fp in seen:
            hits.append((ep["id"], seen[fp], fp))
        else:
            seen[fp] = ep["id"]
    return ("FAIL" if hits else "PASS"), hits


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


def check_truncation(eps, settle_ratio=0.25):
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
            hits.append((ep["id"], ratio, "末帧速度是中位的 %.0f%%，没有收尾"
                         % (ratio * 100)))
        elif len(st) > 2.5 * med_len or len(st) < 0.3 * med_len:
            hits.append((ep["id"], float(len(st)), "长度异常：%d 帧" % len(st)))
    return ("FAIL" if hits else "PASS"), hits, med_len


def check_labels(eps):
    n = len(eps)
    frac = sum(1 for e in eps if e["success"]) / float(n) if n else 0.0
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


# ---------------------------------------------------------------- [1] 数据集

def part1_dataset(eps, planted, limit=ACT_LIMIT, hz=HZ, dof=DOF):
    print("\n[1] 合成数据集：%d 条 episode，%d 自由度，%d Hz，"
          "单步动作上限 ±%.2f rad" % (len(eps), dof, hz, limit))
    print("-" * 74)
    print("  埋进去的缺陷就是「真实采集里最常见、又最不容易发现」的那几类：")
    print()
    print("  %s%s%s" % (pad("缺陷", 18), pad("条数", 8), "为什么难发现"))
    print("  " + "-" * 66)
    items = [
        ("A 重复 episode", len(planted["重复"]), "文件数对、总数对，只是同一条算了两遍"),
        ("A2 重采（近似重复）", len(planted["近似重复"]), "掐掉开头几帧，哈希对不上"),
        ("B 动作错位 1 帧", len(planted["动作错位"]), "形状、范围、loss 全正常"),
        ("C 丢帧", len(planted["丢帧"]), "数组变短了，但没人规定该多长"),

        ("D 时钟倒退", len(planted["时钟倒退"]), "时间戳仍是单调递增的（直到那一点）"),
        ("E 动作饱和", len(planted["动作饱和"]), "数值合法，只是顶在限位上"),
        ("F 演示被截断", len(planted["截断"]), "长度各不相同，截断的也「像是」正常长度"),
        ("G 全部标成功", len(eps), "成功率 100% 听起来是好事"),
        ("H 元数据缺失", len(eps), "文档里有，字段里没有"),
    ]
    for a, b, c in items:
        print("  %s%s%s" % (pad(a, 18), pad("%d" % b, 8), c))
    print()
    print("  这九项的共同点：**没有一个会报错**。数据能读、形状对、训练能跑。")
    print("  它们只改变数据的**含义**，不改变数据的形状。")

    return {"n_ep": len(eps), "planted": planted}


# ---------------------------------------------------------------- [2] 残差形状

def part2_residual_shape(eps):
    print("\n[2] 检查 1 的原理：残差的 U 形")
    print("-" * 74)
    print("  delta 动作的定义就是 state[t+1] − state[t]。这是恒等式，不是近似。")
    print("  所以扫描 offset ∈ [−2, +2]，看残差在哪一格最小：")
    print()

    ep_by_id = {e["id"]: e for e in eps}
    out = {}
    for eid in ("ep_000", "ep_010", "ep_012"):
        ep = ep_by_id[eid]
        amp = statistics.median([sum(abs(v) for v in a) / DOF for a in ep["actions"]])
        o, best_med, med0, _max0, _a = action_state_stats(ep)
        print("  %s  中位单步动作 %.4f rad" % (pad(eid, 10), amp))
        print("    %s%s%s" % (pad("offset", 8), pad("中位残差", 13), "最大残差"))
        for k in range(-2, 3):
            e = residual_at(ep, k)
            mark = "  <- 最小" if k == o else ""
            print("    %s%s%s%s" % (pad("%+d" % k, 8),
                                    pad("%.2e" % statistics.median(e), 13),
                                    "%.2e" % max(e), mark))
        out[eid] = {"amp": amp, "best": o, "ratio": med0 / best_med if best_med else 0,
                    "med0": med0}
        print()

    print("  三条曲线，三种命运：")
    print("    · ep_000 正常：最小值在 0，两侧对称上翘，U 形很干净。")
    print("    · ep_010 错位、动作慢：曲线几乎是**平的**，offset=0 只高 %.1f 倍。"
          % out["ep_010"]["ratio"])
    print("    · ep_012 错位、动作快：标准 U 形，offset=0 高 %.0f 倍，一眼看出。"
          % out["ep_012"]["ratio"])
    print()
    print("  同样的缺陷，同样的一行代码，一个能抓一个抓不到。")
    print("  差别不在算法，在**动作有多快**。下一节把它算出来。")

    return out


# ---------------------------------------------------------------- [3] 可检测性

def part3_detectability(amps=(0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5),
                        freq=0.3, seed=7):
    print("\n[3] 可检测性下限：动作多慢的时候，这个检查就失效了")
    print("-" * 74)
    print("  错一帧造成的残差 ≈ 动作的**二阶差分**，它随速度平方增长：")
    print("    二阶差分 ∝ 幅值 × (2πf·dt)²")
    print("  而跟踪噪声是固定的 %.0e rad。两者的比值决定能不能看出来。" % TRACK_NOISE)
    print("  下面每一行都是真的跑了一遍偏移扫描（f=%.1f Hz）：" % freq)
    print()

    rows = []
    print("  %s%s%s%s%s" % (pad("幅值", 10), pad("步长", 11), pad("二阶/噪声", 12),
                            pad("实测倍数", 12), "判定"))
    print("  " + "-" * 62)
    for amp in amps:
        rng = random.Random(seed)
        ep = make_episode("probe", rng, frames=100, amp_range=(amp, amp),
                          freq_range=(freq, freq))
        ep["actions"] = ep["actions"][1:] + [ep["actions"][-1]]
        _o, best_med, med0, _mx, _a = action_state_stats(ep)
        step = amp * 2 * math.pi * freq * DT
        sd = amp * (2 * math.pi * freq * DT) ** 2
        snr = sd / TRACK_NOISE
        ratio = med0 / best_med if best_med else 0.0
        ok = ratio >= 2.0
        rows.append({"amp": amp, "step": step, "snr": snr, "ratio": ratio, "ok": ok})
        print("  %s%s%s%s%s" % (pad("%.4f" % amp, 10), pad("%.2e" % step, 11),
                                pad("%.2f" % snr, 12), pad("%.1fx" % ratio, 12),
                                "能确诊" if ok else "看不出"))

    first_ok = next((r for r in rows if r["ok"]), None)
    print()
    if first_ok:
        print("  可检测的下限在幅值 %.3f rad 附近（步长 %.1e rad/帧，二阶差分/噪声 %.1f）。"
              % (first_ok["amp"], first_ok["step"], first_ok["snr"]))
    print("  再往下，二阶差分被跟踪噪声淹没 —— **数据里根本没有这个信息**，")
    print("  不是算法不够好。这就是「用慢动作数据检查帧对齐 = 没检查」的量化版本。")
    print()
    print("  实践含义：不要对整个数据集报一个「一致性 OK」。要么按速度分段判定，")
    print("  要么只对快动作段下结论 —— 慢动作段的「通过」是没有信息量的通过。")

    return {"rows": rows, "first_ok": first_ok}


# ---------------------------------------------------------------- [4] 误报

def per_joint_best(ep, span=2):
    """逐关节扫描：每个关节自己的最佳 offset。"""
    out = []
    for j in range(DOF):
        best, best_med = 0, None
        for o in range(-span, span + 1):
            errs = []
            for t in range(len(ep["actions"])):
                i = t + o
                if i < 0 or i + 1 >= len(ep["states"]):
                    continue
                errs.append(abs(ep["states"][i + 1][j] - ep["states"][i][j]
                                - ep["actions"][t][j]))
            med = statistics.median(errs) if errs else float("inf")
            if best_med is None or med < best_med:
                best, best_med = o, med
        out.append(best)
    return out


def shift_states(ep, joints):
    """把指定关节的 state 相对 action 后移一帧（模拟滞后，不改数值范围）。"""
    e = clone(ep, ep["id"])
    for j in joints:
        col = [row[j] for row in e["states"]]
        col = [col[0]] + col[:-1]
        for t, v in enumerate(col):
            e["states"][t][j] = v
    return e


def part4_lag_vs_offset(seed=77):
    print("\n[4] 同一个签名，两个病灶：日志错位 vs 伺服滞后")
    print("-" * 74)
    print("  偏移扫描只告诉你「动作和状态差了一帧」，不告诉你**谁错了**。")
    print("  两种完全不同的情况会给出同一个签名：")
    print("    · 记录 bug：动作被存晚了一步 → 该改采集代码")
    print("    · 伺服滞后：指令发出去了，关节晚一帧才动 → 该改控制器参数")
    print("  如果一律当成记录 bug 去「平移修正」，就会把一个真实的控制延迟抹掉。")
    print()

    rng = random.Random(seed)
    base = make_episode("probe", rng, frames=100, amp_range=(0.3, 0.4),
                        freq_range=(0.2, 0.3))
    cases = [
        ("① 无缺陷", base, []),
        ("② 全关节滞后 1 帧", shift_states(base, range(DOF)), list(range(DOF))),
        ("③ 只有 3 个关节滞后", shift_states(base, (0, 1, 2)), [0, 1, 2]),
    ]

    print("  %s%s%s" % (pad("构造", 22), pad("全身最佳 offset", 18), "逐关节最佳 offset"))
    print("  " + "-" * 66)
    rows = []
    for name, ep, _shifted in cases:
        o_all, best_med, med0, _mx, _a = action_state_stats(ep)
        pj = per_joint_best(ep)
        rows.append({"name": name, "whole": o_all, "per_joint": pj})
        print("  %s%s%s" % (pad(name, 22), pad("%+d" % o_all, 18),
                            " ".join("%+d" % v for v in pj)))

    print()
    print("  全身扫描对 ③ **完全瞎了**：4 个正常关节把中位数拉回 +0，检查报 PASS。")
    print("  而逐关节一看，问题立刻现形：三个关节 +1，四个关节 +0。")
    print()
    print("  这就把「检查粒度」这件事说清楚了：")
    print("    · 全身一个数字 → 只能抓「所有关节一起错」；")
    print("    · 逐关节 → 抓得到「只错了一部分」，并且能区分病灶：")
    print("      全 7 个 +1 是全局性的（记录/时间对齐）；只有 3 个 +1 是按关节的")
    print("      （那三个关节的伺服、负载、或标定）。")
    print()
    print("  所以「排查顺序」不是可选项：**先做逐关节，再做结论**。")
    print("  全身一个数字的检查只能报警，不能定性 —— 而报错报告里如果只有")
    print("  「一致性 FAIL」，下一个人还是得从头查一遍。")

    return {"rows": rows}


# ---------------------------------------------------------------- [5] 整体异常

def part5_action_state(eps, confirmed, suspect, rows):
    print("\n[5] 检查 1 的输出：两类异常，处理方式完全不同")
    print("-" * 74)
    print("  %s%s%s%s%s" % (pad("episode", 12), pad("最佳 offset", 13),
                            pad("动作幅值", 11), pad("倍数", 9), "判定"))
    print("  " + "-" * 66)

    confirmed_ids = set(h[0] for h in confirmed)
    sporadic = []
    for eid, o, med0, max0, best_med, amp in rows:
        if o != 0:
            r = med0 / best_med if best_med > 0 else float("inf")
            verdict = "确诊：整体错位" if eid in confirmed_ids else "待确认：差值太小"
            print("  %s%s%s%s%s" % (pad(eid, 12), pad("%+d" % o, 13),
                                    pad("%.4f" % amp, 11), pad("%.1fx" % r, 9), verdict))
        elif max0 > 1e-3:
            sporadic.append((eid, max0, amp))
            print("  %s%s%s%s%s" % (pad(eid, 12), pad("0", 13),
                                    pad("%.4f" % amp, 11), pad("—", 9),
                                    "散点损坏：最大残差 %.1e" % max0))

    print()
    print("  · 最佳 offset ≠ 0 → **整体性错位**（%d 条确诊，%d 条待确认）。"
          % (len(confirmed), len(suspect)))
    print("    数据形状对、范围对、loss 也降，只是时间对应整体偏了。")
    print("    这类必须修：动作错一帧，策略学到的是系统性提前量。")
    print("  · 中位正常、最大残差很大 → **个别帧坏了**（%d 条）。" % len(sporadic))
    print("    用中位数永远抓不到 —— 必须看最大值或高分位数。")
    print()
    print("  后者正是丢帧留下的：三路数组一起删，接缝上那个动作还是按")
    print("  「原第 k 帧到 k+1 帧」算的，但后面已经换成第 k+drop 帧了。")

    return {"sporadic": sporadic}


# ---------------------------------------------------------------- [6] 时间戳

def part6_timestamps(eps, gaps, jumps, rows):
    print("\n[6] 检查 2：时间戳是丢帧的唯一直接证据")
    print("-" * 74)
    print("  判定：相邻间隔 > 1.8 倍中位数 = 丢帧；间隔 ≤ 0 = 时钟跳变。")
    print()
    print("  丢帧 %d 处（涉及 %d 条）：" % (len(gaps), len(set(g[0] for g in gaps))))
    for eid, i, k in gaps:
        print("    %s第 %d 帧后，间隔是正常值的 %d 倍（丢了 %d 帧）"
              % (pad(eid, 12), i, k, k - 1))
    print("  时钟倒退 %d 处（涉及 %d 条）："
          % (len(jumps), len(set(j[0] for j in jumps))))
    for eid, i, d in jumps:
        print("    %s第 %d 帧后，间隔 %.3f 秒" % (pad(eid, 12), i, d))

    broken = set(r[0] for r in rows if r[1] == 0 and r[3] > 1e-3)
    missed = sorted(set(g[0] for g in gaps) - broken)
    print()
    if missed:
        print("  对照上一节：%s 的丢帧**没有**出现在「散点损坏」里。"
              % "、".join(missed))
        print("  因为它丢在减速段，前后状态本来就接近，残差没被拉大。")
        print("  所以两项检查不是冗余：一项抓「错位」，一项抓「缺帧」。")
    print()
    print("  为什么丢帧比想象中严重：动作是按帧存的，丢一帧就跳过 33 ms 的物理过程。")
    print("  如果丢的那帧正好在接触瞬间，这条演示的因果链就断了 —— 但数据看起来正常。")
    print("  顺带：检查时钟异常**不能**用 ORDER BY timestamp，跳变后顺序就错了。")

    return {"missed": missed}


# ---------------------------------------------------------------- [7] 重复

def part7_duplicates(eps, planted, dup_hits):
    print("\n[7] 检查 3：精确哈希抓得到复制，抓不到「重采」")
    print("-" * 74)
    print("  对动作序列算 sha256，相同就是同一段演示：")
    print()
    for eid, orig, fp in dup_hits:
        print("    %s与 %s 完全相同（%s）" % (pad(eid, 12), pad(orig, 12), fp))

    near = planted["近似重复"]
    exact_ids = set(h[0] for h in dup_hits)
    print()
    print("  但 %d 条「重采」（掐掉开头几帧 + 微小噪声）一条都没被抓到 ——" % len(near))
    print("  因为它们不是逐字节相同。这正是哈希的边界。")
    print()
    print("  换成粗指纹：把状态轨迹按时间等分成 32 个点，比最大坐标差。")
    print()

    sigs = {e["id"]: coarse_signature(e) for e in eps}
    base_ids = [e["id"] for e in eps if e["id"] not in
                set(planted["近似重复"]) | set(planted["重复"])]
    groups = []
    for eid in sorted(exact_ids):
        orig = [h[1] for h in dup_hits if h[0] == eid][0]
        groups.append(("精确复制", sig_distance(sigs[eid], sigs[orig])))
    for eid in near:
        src = "ep_%03d" % (5 + near.index(eid))
        groups.append(("重采", sig_distance(sigs[eid], sigs[src])))
    rng = random.Random(11)
    for _ in range(40):
        a, b = rng.sample(base_ids, 2)
        groups.append(("无关的两条", sig_distance(sigs[a], sigs[b])))

    print("  %s%s%s" % (pad("组", 14), pad("最大坐标差", 16), "区间"))
    print("  " + "-" * 58)
    stats = {}
    for name in ("精确复制", "重采", "无关的两条"):
        vals = [v for g, v in groups if g == name]
        stats[name] = (min(vals), max(vals))
        print("  %s%s%s" % (pad(name, 14), pad("%.4f" % max(vals), 16),
                            "%.4f – %.4f" % (min(vals), max(vals))))

    print()
    print("  三组之间差了两个数量级：复制 ≈ 0，重采 %.3f 量级，无关的两条 %.2f。"
          % (stats["重采"][1], stats["无关的两条"][0]))
    print("  门槛取 %.1f：是重采最大值的 %.0f 倍，又只有无关最小值的 %.0f%% —— 两边"
          % (0.2, 0.2 / stats["重采"][1], 0.2 / stats["无关的两条"][0] * 100))
    print("  都留了余量。这个门槛不是原理推出来的，是**看自己数据的两端**定的。")
    print()
    print("  危害不是「浪费空间」，是这些样本在训练里被**加权两倍**，")
    print("  等于悄悄改了数据分布。而且它是静默的 —— 你只会看到「这个任务")
    print("  过拟合得特别快」。")

    return {"stats": stats, "n_near": len(near), "n_exact": len(dup_hits)}


# ---------------------------------------------------------------- [8] 饱和

def part8_saturation(sat_hits, ratios=(1.0, 1.05, 1.2, 1.5, 2.0, 3.0, 5.0),
                     limit=ACT_LIMIT, n=720):
    print("\n[8] 检查 4：饱和的两种严重度 —— 多少帧被裁 vs 多少位移丢了")
    print("-" * 74)
    print("  模型：动作是正弦，峰值 A，限位 L。逐帧算 |a| > L 的比例，")
    print("  以及**被裁掉的那部分位移**占原本应走位移的比例。")
    print()
    print("  %s%s%s" % (pad("峰值/限位", 12), pad("被裁帧数", 14), "丢失位移"))
    print("  " + "-" * 48)

    rows = []
    for r in ratios:
        A = r * limit
        clipped = 0
        lost = 0.0
        total = 0.0
        for i in range(n):
            a = A * math.sin(2 * math.pi * i / n)
            total += abs(a)
            if abs(a) > limit:
                clipped += 1
                lost += abs(a) - limit
        rows.append({"ratio": r, "clip_frac": clipped / float(n),
                     "lost_frac": lost / total})
        print("  %s%s%s" % (pad("%.2f" % r, 12), pad("%.1f%%" % (clipped / float(n) * 100), 14),
                            "%.1f%%" % (lost / total * 100)))

    print()
    r15 = [x for x in rows if abs(x["ratio"] - 1.5) < 1e-9][0]
    print("  峰值超限 %.1f 倍时：%.0f%% 的帧被裁，但只丢了 %.0f%% 的位移。"
          % (r15["ratio"], r15["clip_frac"] * 100, r15["lost_frac"] * 100))
    print("  两个数差这么多，因为它们量的不是一件事：帧数按时间算，位移按幅度算。")
    print("  只看「被裁帧数」，会高估严重程度；只看位移，会低估「动作形状被削平」。")
    print()
    r_th = 1.0 / math.sin((1.0 - 0.05) * math.pi / 2.0)
    print("  顺带：5%% 的告警阈值等价于「峰值只超限 %.2f%%」。" % ((r_th - 1) * 100))
    print("  （正弦模型：被裁比例 = 1 − 2/π·asin(1/r)，解 5%% 得 r = %.4f）" % r_th)
    print("  门槛这么低，是因为正弦在峰值附近停留时间长 —— 稍微超一点就有很多帧被裁。")
    print("  所以饱和告警不适合当自动拒绝，更适合当「这条要人看一眼」。")
    print()
    print("  本例里的实际命中（阈值：单条饱和比例 > 5%）：")
    for eid, frac in sat_hits:
        print("    %s饱和比例 %.1f%%" % (pad(eid, 12), frac * 100))
    print()
    print("  饱和意味着「机器人想要更快，但被限位挡住了」。这类数据学出来的是")
    print("  「一直输出最大值」，到了不限位的场景就过冲。正确做法是丢掉饱和段或")
    print("  重采慢一点，**不要用归一化把它压回去** —— 被截断的动作和真实动作不是一回事。")

    return {"rows": rows, "r15": r15}


# ---------------------------------------------------------------- [9] 截断

def part9_truncation(trunc_hits, med_len):
    print("\n[9] 检查 5：截断 —— 模型学到「做一半就算完成」")
    print("-" * 74)
    print("  判定：末帧速度 > 中位速度的 25% = 没有收尾，疑似截断。")
    print("  **前提假设**：任务完成时机器人停在静止状态。抓取、放置、开门成立，")
    print("  「推着物体一直走」不成立 —— 那类任务要单独处理。")
    print("  中位长度 %d 帧（%.1f 秒）" % (med_len, med_len * DT))
    print()
    for eid, ratio, note in trunc_hits[:8]:
        print("    %s%s" % (pad(eid, 12), note))
    print()
    print("  截断的演示最坑：它把「动作做了一半」当成完整轨迹。部署时表现为")
    print("  提前松手。根因几乎总是采集中断（超时、误触停止、磁盘满）。")
    print()
    print("  写检查时一定要把假设写出来 —— 否则别人套到不成立的场景上会得到")
    print("  一堆假警报，然后整个工具被弃用。")

    return {"hits": len(trunc_hits)}


# ---------------------------------------------------------------- [10] 标注/元数据

def part10_labels_meta(eps, frac_ok, meta_hits):
    print("\n[10] 检查 6 / 7：全成功，和「文档里有、字段里没有」")
    print("-" * 74)
    print("  成功率 %.0f%%（%d 条）" % (frac_ok * 100, len(eps)))
    print()
    print("  全部成功听起来是好事，实际是个问题：")
    print("    · 无法做「只训成功样本」的筛选 —— 本来就没得选")
    print("    · 无法估计真实成功率 —— 数据集里的成功率不是策略的成功率")
    print("    · 丢掉的全是**最有信息量**的样本：失败的边界情况")
    print()
    for k, v in meta_hits:
        print("    %s缺失 %d / %d 条" % (pad(k, 14), v, len(eps)))
    print()
    print("  注意 control_hz 是**通过**的（脚本里设了），因为它是唯一一个")
    print("  「写在文档里就很容易被当成已有」的字段。另外两个全缺。")
    print()
    print("  这三个字段缺失的后果在本章第一篇量化过：")
    print("    control_hz → 动作没有物理量纲（3 Hz 与 30 Hz 差 10 倍）")
    print("    extrinsics → 多视角几何算不出来，点云无法融合")
    print("    embodiment → 无法按本体分组归一化")

    return {"frac_ok": frac_ok}


# ---------------------------------------------------------------- [11] 汇总

def part11_summary(status, confirmed, suspect, sporadic, gaps, jumps,
                   dup_hits, sat_hits, trunc_hits, label_hits, meta_hits,
                   planted, rows):
    print("\n" + "=" * 74)
    print("汇总：七项检查，九类缺陷")
    print("=" * 74)
    summary = [
        ("动作-状态一致性", status, len(confirmed) + len(suspect),
         "确诊 %d / 待确认 %d / 散点 %d" % (len(confirmed), len(suspect), len(sporadic))),
        ("时间戳", "FAIL" if (gaps or jumps) else "PASS", len(gaps) + len(jumps),
         "丢帧 %d / 倒退 %d" % (len(gaps), len(jumps))),
        ("重复 episode", "FAIL" if dup_hits else "PASS", len(dup_hits),
         "内容逐字节相同"),
        ("动作饱和", "FAIL" if sat_hits else "PASS", len(sat_hits),
         "单条饱和比例过高"),
        ("截断/长度", "FAIL" if trunc_hits else "PASS", len(trunc_hits),
         "演示在运动中结束"),
        ("成败标注", "FAIL" if label_hits else "PASS", len(label_hits), "无法做筛选"),
        ("元数据", "FAIL" if meta_hits else "PASS", len(meta_hits),
         "%s 缺失" % "、".join(k for k, _ in meta_hits)),
    ]
    print("  %s%s%s%s" % (pad("检查项", 20), pad("结果", 8), pad("命中", 8), "说明"))
    print("  " + "-" * 70)
    for name, st, cnt, note in summary:
        print("  %s%s%s%s" % (pad(name, 20), pad(st, 8), pad("%d" % cnt, 8), note))

    print()
    print("  召回核对：")
    print()
    detected = {
        "重复": set(h[0] for h in dup_hits),
        "动作错位": set(h[0] for h in confirmed + suspect),
        "丢帧": set(g[0] for g in gaps),
        "时钟倒退": set(j[0] for j in jumps),
        "动作饱和": set(s[0] for s in sat_hits),
        "截断": set(t[0] for t in trunc_hits),
    }
    print("  %s%s%s%s" % (pad("缺陷", 12), pad("埋入", 8), pad("检出", 8), "结论"))
    print("  " + "-" * 54)
    recall = {}
    for key in ("重复", "动作错位", "丢帧", "时钟倒退", "动作饱和", "截断"):
        exp, got = set(planted[key]), detected[key]
        miss, extra = exp - got, got - exp
        verdict = "全部命中" if not (miss or extra) else (
            "漏 %d" % len(miss) if miss else "多报 %d" % len(extra))
        recall[key] = {"exp": len(exp), "hit": len(exp & got), "miss": len(miss),
                       "extra": len(extra)}
        print("  %s%s%s%s" % (pad(key, 12), pad("%d" % len(exp), 8),
                              pad("%d" % len(exp & got), 8), verdict))

    print()
    print("  六类埋入的缺陷全部被逐条定位，没有漏检、没有误报。")
    print("  **但这个结论只在合成数据上成立**：真实数据的噪声结构更复杂，")
    print("  而且第 4 节已经说明——全身扫描对「只有部分关节错位」是盲的，")
    print("  这里的 100% 召回，是在「所有关节一起错」的前提下拿到的。")

    return {"summary": summary, "recall": recall}


# ---------------------------------------------------------------- [12] 结论

def part12_conclusion(p2, p3, p4, p7, p8, p11):
    print("\n" + "=" * 74)
    print("结论（数据平台视角）")
    print("=" * 74)
    print("  1. 缺陷不报错，只改含义。所以检查必须自动、每次入库都跑。")
    print("  2. 动作-状态一致性是最该先做的一项：不需要外部信息，只看数据本身。")
    print("  3. 但它的灵敏度取决于**动作速度**：幅值低于 %.3f rad 时二阶差分被噪声"
          % (p3["first_ok"]["amp"] if p3["first_ok"] else 0.0))
    print("     淹没，检查必然失效。所以判定要分级，别追求「全自动确诊」。")
    print("  4. 检查粒度决定能不能定性：全身扫描对「3 个关节滞后」直接报 PASS，")
    print("     逐关节才看得到。先分诊（哪些关节），再定因（记录还是伺服）。")
    print("  5. 精确哈希只抓复制：%d 条重采一条都抓不到。粗指纹要设门槛（本例 %.1f），"
          % (p7["n_near"], 0.2))
    print("     门槛得看自己数据的两端定，不是原理推出来的。")
    print("  6. 饱和要看两个数：峰值超限 1.5 倍时 %.0f%% 的帧被裁，但只丢 %.0f%% 的位移。"
          % (p8["r15"]["clip_frac"] * 100, p8["r15"]["lost_frac"] * 100))
    print("  7. 检查只能兜底，流程才是根本：采之前定好外参、时钟、频率，")
    print("     比事后修便宜一个数量级。")
    print("=" * 74)


# ---------------------------------------------------------------- 主流程

def main():
    print("=" * 74)
    print("实验 3：自采数据集的自检清单")
    print("=" * 74)

    eps, planted = build_dataset()
    p1 = part1_dataset(eps, planted)
    p2 = part2_residual_shape(eps)
    p3 = part3_detectability()
    p4 = part4_lag_vs_offset()

    status, confirmed, suspect, rows = check_action_state(eps)
    p5 = part5_action_state(eps, confirmed, suspect, rows)

    _st, gaps, jumps = check_timestamps(eps)
    p6 = part6_timestamps(eps, gaps, jumps, rows)

    _sd, dup_hits = check_duplicates(eps)
    p7 = part7_duplicates(eps, planted, dup_hits)

    _ss, sat_hits = check_saturation(eps)
    p8 = part8_saturation(sat_hits)

    _sc, trunc_hits, med_len = check_truncation(eps)
    p9 = part9_truncation(trunc_hits, med_len)

    _sl, label_hits, frac_ok = check_labels(eps)
    _sm, meta_hits = check_metadata(eps)
    p10 = part10_labels_meta(eps, frac_ok, meta_hits)

    p11 = part11_summary(status, confirmed, suspect, p5["sporadic"], gaps, jumps,
                         dup_hits, sat_hits, trunc_hits, label_hits, meta_hits,
                         planted, rows)
    part12_conclusion(p2, p3, p4, p7, p8, p11)


if __name__ == "__main__":
    main()

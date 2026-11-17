#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验 2：大规模采集的真实成本结构
================================

DROID 用 50 个人、12 个月、18 个实验室，采了 76k 条轨迹 / 350 小时数据。
这个规模看起来是「数据量」的胜利，实际是「人力时间」的账单。

这个脚本把账单拆开：

  [1] 数字对账：76k / 350h / 564 场景互相除一遍，看能不能对上
  [2] 人力效率：100,800 人时 → 350 小时，0.35%
  [3] 固定成本 vs 边际成本：多少时间花在「场景」上，多少在「每条」上
  [4] 每场景 100 条从哪来：不是成本定的，是场景内部的信息饱和
  [5] 场景覆盖曲线：后 93% 的数据只多覆盖 10% 的场景
  [6] 长尾敏感度：分布形状决定你要 1 万条还是 10 万条
  [7] 每场景预算的取舍：100 条/场景 vs 1000 条/场景
  [8] 失败数据与「售后」：0.6% 的追加投入决定另外 99% 能不能用

数据来源：DROID 论文与项目页（arXiv:2403.12945, 2024）。
凡是「模型假设」的地方都会标出来，并且给敏感度。

运行：python examples/04_datasets/02_collection_cost.py
依赖：无（只用标准库）
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def width(s):
    return sum(2 if ord(c) > 0x2E80 else 1 for c in s)


def pad(s, n):
    return s + " " * max(0, n - width(s))


# ---------------------------------------------------------------- 公开数字

TRAJ = 76000          # 成功轨迹数
FAIL = 16000          # 失败轨迹数
HOURS = 350.0         # 交互数据时长（小时）
SCENES = 564          # 场景数
TASKS = 86            # 任务（动词类别）数
BUILDINGS = 52        # 建筑数
PEOPLE = 50           # 采集员
MONTHS = 12           # 采集周期
HZ = 15               # 记录频率
CAMS = 3              # 相机路数
WORKDAY = 8.0         # 每天工作小时
WORKDAYS = 21         # 每月工作日


# ---------------------------------------------------------------- [1] 对账

def part1_reconcile(traj=TRAJ, hours=HOURS, scenes=SCENES, tasks=TASKS):
    print("\n[1] 数字对账：把 76k / 350h / 564 放在一起看")
    print("-" * 74)

    secs = hours * 3600.0 / traj
    frames = secs * HZ
    total_frames_single = traj * frames
    per_scene_traj = traj / scenes
    per_scene_min = hours * 60.0 / scenes
    per_task = traj / tasks
    per_building = scenes / BUILDINGS

    rows = [
        ("平均每条轨迹时长", "%.1f 秒" % secs, "350h ÷ 76k；和「抓一个杯子」的直觉一致"),
        ("每条轨迹的帧数", "%.0f 帧" % frames, "%.1f 秒 × %d Hz" % (secs, HZ)),
        ("总帧数（单路）", "%.1f M" % (total_frames_single / 1e6),
         "后面算存储和训练步数都要用"),
        ("总图像帧数", "%.1f M" % (total_frames_single * CAMS / 1e6),
         "× %d 路相机" % CAMS),
        ("每场景平均轨迹数", "%.0f 条" % per_scene_traj, "76k ÷ 564；官方描述是「最多 100 条」"),
        ("每场景平均数据量", "%.0f 分钟" % per_scene_min, "350h ÷ 564"),
        ("每任务平均轨迹数", "%.0f 条" % per_task, "长尾分布，平均值没有意义"),
        ("每栋建筑平均场景数", "%.1f 个" % per_building, "564 ÷ 52"),
    ]

    print("  %s%s%s" % (pad("指标", 20), pad("数值", 14), "怎么算的 / 注意什么"))
    print("  " + "-" * 72)
    for a, b, c in rows:
        print("  %s%s%s" % (pad(a, 20), pad(b, 14), c))

    implied = 20.0 * 60.0 / 100.0      # 20 分钟 / 100 条
    gap = abs(per_scene_min - per_scene_traj * implied / 60) / (per_scene_traj * implied / 60)
    print()
    print("  这里有一处**内部不一致**，值得单独说：")
    print("    官方：「每人每场景最多 100 条轨迹，约 20 分钟数据」")
    print("      → 隐含每条 %.0f 秒、每场景 20 分钟" % implied)
    print("    但 76k ÷ 564 = %.0f 条/场景，350h ÷ 564 = %.0f 分钟/场景，"
          % (per_scene_traj, per_scene_min))
    print("    且 350h ÷ 76k = %.1f 秒/条。按 %.0f 条算应为 %.0f 分钟。"
          % (secs, per_scene_traj, per_scene_traj * implied / 60))
    print()
    print("  这不是谁算错了，而是这类数字本来就是**估算量级**：")
    print("  350h 可能四舍五入，76k 是成功轨迹数，「20 分钟」是口语。")
    print("  差 %.0f%% 在存储预算上就是几十 TB。" % (gap * 100))
    print()
    print("  教训：拿公开数据集的统计量做容量规划，先自己对账一遍。")

    return {"secs": secs, "frames": frames, "per_scene_traj": per_scene_traj,
            "per_scene_min": per_scene_min, "total_frames": total_frames_single,
            "implied_secs": implied, "gap_frac": gap}


# ---------------------------------------------------------------- [2] 效率

def part2_human_cost(people=PEOPLE, months=MONTHS, hours=HOURS,
                     workday=WORKDAY, workdays=WORKDAYS):
    total = people * months * workdays * workday
    per_hour = total / hours
    eff = hours / total

    print("\n[2] 人力效率：每 %.0f 小时人力，产出 1 小时数据" % per_hour)
    print("-" * 74)

    per_day_traj = hours / (people * months * workdays) * 3600.0 / 16.6
    print("  名义投入：%d 人 × %d 个月 × %d 工作日 × %.0f 小时 = %s 人时"
          % (people, months, workdays, workday, format(int(total), ",")))
    print("  实际产出：%.0f 小时可用数据" % hours)
    print()
    print("  有效率 = %.0f / %s = **%.2f%%**" % (hours, format(int(total), ","), eff * 100))
    print("  也就是 **每 %.0f 小时人力，产出 1 小时数据**。" % per_hour)
    print()
    print("  换个说法：一名采集员一天产出 %.1f 条轨迹 = %.0f 秒有效数据，"
          % (per_day_traj, per_day_traj * 16.6))
    print("  占 8 小时工作时间的 %.2f%%。" % (eff * 100))
    print()
    print("  按场景摊：%s 人时 ÷ %d 个场景 = **%.0f 人时/场景**，"
          % (format(int(total), ","), SCENES, total / SCENES))
    print("  折合 %.1f 个工作日 —— 一个人干一个月，换来 %.0f 分钟数据。"
          % (total / SCENES / workday, hours * 60 / SCENES))
    print()
    print("  剩下 %.1f%% 的时间去哪了？看官方描述的流程：" % ((1 - eff) * 100))
    print()
    print("  %s%s" % (pad("开销项", 26), "说明"))
    print("  " + "-" * 72)
    for a, b in (("把平台推到新场景", "搬运、调平、布线"),
                 ("相机外参标定", "每个场景用棋盘格重标一次"),
                 ("录入任务清单", "手工输入或从列表选择"),
                 ("试采 + 调整视角", "找到能拍到有效行为的机位"),
                 ("失败重采", "约 16k 条失败轨迹，不进成功集"),
                 ("复位与场景增强", "移底座、改光照、增删物品")):
        print("  %s%s" % (pad(a, 26), b))
    print()
    print("  所以：存储和带宽**不是**瓶颈 —— 350 小时数据一块硬盘装得下。")
    print("  瓶颈是**人在场景里的时间**。")

    return {"total_hours": total, "eff": eff, "per_hour": per_hour,
            "hours_per_scene": total / SCENES,
            "days_per_scene": total / SCENES / workday}


# ---------------------------------------------------------------- [3] 固定/边际

def part3_fixed_vs_marginal(traj=TRAJ, scenes=SCENES,
                            total_hours=None, per_ep_options=(16.6, 60.0, 180.0)):
    if total_hours is None:
        total_hours = PEOPLE * MONTHS * WORKDAYS * WORKDAY

    print("\n[3] 固定成本 vs 边际成本：时间花在「场景」上还是「每条」上")
    print("-" * 74)
    print("  总时间 T = 场景数 × 每场景固定开销 + 轨迹数 × 每条边际开销")
    print("  公开数字只给了 T、场景数、轨迹数，**没给这两项的拆分**。")
    print("  但可以框住它：每条轨迹的边际开销不可能低于它的数据时长 16.6 秒。")
    print()
    print("  %s%s%s%s" % (pad("每条边际", 12), pad("边际合计", 12),
                          pad("场景级合计", 13), "占全部时间"))
    print("  " + "-" * 62)

    rows = []
    for sec in per_ep_options:
        marg = traj * sec / 3600.0
        fixed = total_hours - marg
        share = fixed / total_hours
        rows.append({"per_ep_s": sec, "marginal_h": marg, "fixed_h": fixed,
                     "fixed_share": share,
                     "per_scene_h": fixed / scenes,
                     "ratio": (fixed / scenes) / (sec / 3600.0)})
        print("  %s%s%s%s" % (pad("%.0f 秒" % sec, 12),
                              pad("%.0f h" % marg, 12),
                              pad("%.0f h" % fixed, 13),
                              "%.1f%%" % (share * 100)))

    lo = min(r["fixed_share"] for r in rows)
    mid = rows[1]
    print()
    print("  **无论怎么假设，场景级开销都占 %.0f%% 以上。**" % (lo * 100))
    print("  这不是精度问题，是量级问题：一条轨迹 16.6 秒，一个场景 3 天。")
    print()
    print("  取中间假设（每条 %.0f 秒，含复位）：每场景固定开销 %.0f 人时，"
          % (mid["per_ep_s"], mid["per_scene_h"]))
    print("  是单条边际开销的 **%s 倍**。" % format(int(mid["ratio"]), ","))
    print()
    print("  这个比值决定了所有采集策略：")
    print("    · 开一个新场景 = 放弃 %s 条已有场景的 episode"
          % format(int(mid["ratio"]), ","))
    print("    · 所以只要一个场景还能教出新东西，就该继续在它身上采")
    print("    · 一旦它教不出新东西，多留一分钟都是纯损失")
    print()
    print("  **瓶颈不在钱，在「这个场景还有没有新东西」。** 下一节量化它。")

    return {"rows": rows, "mid": mid, "lo_share": lo}


# ---------------------------------------------------------------- [4] 场景内饱和

def part4_within_scene(betas=(1.2, 1.5, 2.0, 2.5), ks=(50, 100, 200, 500, 1000),
                       k_inf=20000):
    print("\n[4] 每场景 100 条从哪来：不是成本，是场景内部的信息饱和")
    print("-" * 74)
    print("  模型假设（明确标出）：同一个场景里，第 k 条 episode 带来的**新增**")
    print("  物体-姿态组合按 k^-β 衰减。β 越大，场景越快被「采干」。")
    print()
    print("  那么前 k 条覆盖了该场景总多样性的多少？")
    print()
    head = "  " + pad("β", 8)
    for k in ks:
        head += pad("k=%d" % k, 10)
    print(head)
    print("  " + "-" * (8 + 10 * len(ks)))

    table = {}
    for beta in betas:
        tot = sum(i ** (-beta) for i in range(1, k_inf + 1))
        line = "  " + pad("%.1f" % beta, 8)
        row = {"beta": beta, "cov": []}
        for k in ks:
            part = sum(i ** (-beta) for i in range(1, k + 1))
            c = part / tot
            row["cov"].append(c)
            line += pad("%.1f%%" % (c * 100), 10)
        table[beta] = row
        print(line)

    key = 2.0 if 2.0 in betas else betas[len(betas) // 2]
    b2 = table[key]
    i100 = ks.index(100) if 100 in ks else 0
    i1000 = ks.index(1000) if 1000 in ks else len(ks) - 1
    print()
    print("  β=%.1f 时：100 条覆盖 %.1f%%，1000 条覆盖 %.1f%% —— 多 10 倍的采集，"
          % (key, b2["cov"][i100] * 100, b2["cov"][i1000] * 100))
    print("  只多买到 %.1f 个百分点。" % ((b2["cov"][i1000] - b2["cov"][i100]) * 100))
    print()
    print("  把两边放在一起看，结论有点反直觉：")
    print("    · **成本**说：场景贵上千倍，所以应该在一个场景里采到远超 100 条")
    print("    · **信息**说：100 条之后场景基本被采干，再采是同分布重复样本")
    print("    · 所以约束是**信息**，不是钱。DROID 的 100 条是信息边界。")
    print()
    print("  这也解释了为什么「每场景条数」没有公式：β 是你自己数据的性质，")
    print("  只能看数据估。但在 β=2 附近，100 这个量级是对的。")

    return {"table": table, "ks": list(ks), "betas": list(betas), "key": key,
            "cov100": b2["cov"][i100], "cov1000": b2["cov"][i1000]}


# ---------------------------------------------------------------- [5] 覆盖曲线

def zipf_probs(n, alpha):
    w = [1.0 / (i + 1) ** alpha for i in range(n)]
    tot = sum(w)
    return [x / tot for x in w]


def expected_distinct(probs, draws):
    """E[已见类别数] = Σ (1 - (1-p_i)^draws)。"""
    return sum(1.0 - (1.0 - p) ** draws for p in probs)


def _solve_for(probs, target):
    lo, hi = 1, 2000000
    while lo < hi:
        mid = (lo + hi) // 2
        if expected_distinct(probs, mid) / len(probs) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


def part5_coverage_curve(n_scenes=SCENES, alpha=1.0, traj=TRAJ,
                         checkpoints=(500, 1000, 2000, 5000, 10000, 20000, 40000, 76000)):
    print("\n[5] 场景覆盖曲线：采样收益是饱和的，不是线性的")
    print("-" * 74)
    print("  模型假设：%d 个场景的出现频率服从 Zipf(α=%.1f)。" % (n_scenes, alpha))
    print("  这**不是** DROID 的实测分布，但长尾这个形状在各类机器人数据集里")
    print("  是一致的：少数场景占掉大部分数据。")
    print()

    probs = zipf_probs(n_scenes, alpha)
    rows = []
    print("  %s%s%s" % (pad("已采 episode", 15), pad("期望覆盖场景", 15), "覆盖率"))
    print("  " + "-" * 46)
    for n in checkpoints:
        if n > 1000000:
            continue
        seen = expected_distinct(probs, n)
        rows.append({"n": n, "seen": seen, "cov": seen / n_scenes})
        print("  %s%s%s" % (pad(format(n, ","), 15),
                            pad("%.0f" % seen, 15), "%.1f%%" % (seen / n_scenes * 100)))

    print()
    print("  边际收益（每 1000 条 episode 带来多少个新场景）：")
    print()
    print("  %s%s%s" % (pad("区间", 24), pad("新增场景", 12), "每千条新增"))
    print("  " + "-" * 52)
    margin = []
    bounds = [0, 1000, 5000, 10000, 30000, traj]
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        seen_a = expected_distinct(probs, a) if a else 0.0
        seen_b = expected_distinct(probs, b)
        gain = seen_b - seen_a
        per_k = gain / ((b - a) / 1000.0)
        margin.append({"from": a, "to": b, "gain": gain, "per_k": per_k})
        print("  %s%s%s" % (pad("%s – %s" % (format(a, ",") if a else "0",
                                             format(b, ",")), 24),
                            pad("%.0f 个" % gain, 12), "%.1f 个" % per_k))

    n50 = _solve_for(probs, 0.5)
    n90 = _solve_for(probs, 0.9)
    print()
    print("  覆盖 50%% 需要 %s 条；覆盖 90%% 需要 %s 条。"
          % (format(int(n50), ","), format(int(n90), ",")))
    print("  实际采了 %s 条 —— 后 %s 条（占 %.0f%% 的数据）"
          % (format(traj, ","), format(int(traj - n90), ","),
             (traj - n90) / traj * 100))
    print("  只多覆盖了最后 %.0f%% 的场景。" % ((1 - 0.9) * 100))
    print()
    print("  这看起来很浪费，但它是**对的**：「覆盖多少场景」和「覆盖哪些场景」")
    print("  是两件事。尾部场景按数据量算不足 10%，按信息量算是最重要的。")

    return {"rows": rows, "margin": margin, "n50": n50, "n90": n90,
            "tail_frac": (traj - n90) / traj}


# ---------------------------------------------------------------- [6] 长尾敏感度

def part6_alpha(n_scenes=SCENES, alphas=(0.8, 1.0, 1.2, 1.5), target=0.9):
    print("\n[6] 长尾敏感度：分布形状决定你要 1 万条还是 10 万条")
    print("-" * 74)
    print("  α 越大，分布越集中（少数场景吃掉更多数据）。")
    print()
    print("  %s%s%s" % (pad("α", 8), pad("覆盖 90% 所需", 18), "含义"))
    print("  " + "-" * 58)

    rows = []
    labels = {0.8: "较均匀，好采", 1.0: "基准（Zipf）",
              1.2: "很集中，尾部难碰到", 1.5: "极集中"}
    for a in alphas:
        n = _solve_for(zipf_probs(n_scenes, a), target)
        rows.append({"alpha": a, "n": n})
        print("  %s%s%s" % (pad("%.1f" % a, 8), pad(format(int(n), ",") + " 条", 18),
                            labels.get(a, "")))

    ratio = max(r["n"] for r in rows) / min(r["n"] for r in rows)
    print()
    print("  α 从 %.1f 到 %.1f，所需数量差 **%.0f 倍**。"
          % (min(alphas), max(alphas), ratio))
    print("  所以在设计采集计划之前，**先摸清自己的场景分布有多长尾** ——")
    print("  这个数决定你需要 1 万条还是 10 万条。")

    return {"rows": rows, "ratio": ratio}


# ---------------------------------------------------------------- [7] 预算取舍

def part7_budget(traj=TRAJ, n_scenes=SCENES, alpha=1.0,
                 budgets=(50, 100, 200, 500, 1000, 2000)):
    print("\n[7] 每场景预算的取舍：100 条/场景 vs 1000 条/场景")
    print("-" * 74)
    print("  固定总量 %s 条，问：每个场景花多少条？" % format(traj, ","))
    print("  能去的场景数 = 总量 ÷ 每场景预算（上限 %d 个）。" % n_scenes)
    print()

    probs = zipf_probs(n_scenes, alpha)
    cum, acc = [], 0.0
    for p in probs:
        acc += p
        cum.append(acc)

    rows = []
    print("  %s%s%s" % (pad("每场景预算", 14), pad("能去场景数", 14), "覆盖的数据比例"))
    print("  " + "-" * 52)
    for b in budgets:
        m = min(n_scenes, traj // b)
        cov = cum[m - 1]
        rows.append({"budget": b, "scenes": m, "mass": cov})
        print("  %s%s%s" % (pad("%d 条" % b, 14), pad("%d 个" % m, 14),
                            "%.1f%%" % (cov * 100)))

    r100 = [r for r in rows if r["budget"] == 100][0]
    r1000 = [r for r in rows if r["budget"] == 1000][0]
    n90 = _solve_for(probs, 0.9)
    print()
    print("  从 100 条/场景 降到 1000 条/场景：场景数从 %d 掉到 %d，"
          % (r100["scenes"], r1000["scenes"]))
    print("  覆盖的数据比例从 %.0f%% 掉到 %.0f%%。"
          % (r100["mass"] * 100, r1000["mass"] * 100))
    print()
    print("  但这个百分比是**按出现频率加权**的：它衡量「你见过多少数据」，")
    print("  不衡量「你见过多少种罕见行为」。尾部 400 个场景按频率算不足 10%，")
    print("  但「在窄柜子里叠衣服」这种行为只存在于尾部。")
    print()
    print("  所以取舍不是「覆盖率最大化」，而是：")
    print("    · 头部场景 → 少量多条，把基础动作学扎实")
    print("    · 尾部场景 → 每个都要去，哪怕只采几十条")
    print("      （第 5 节：覆盖 90%% 场景只要 %s 条，不需要 %s 条）"
          % (format(int(n90), ","), format(traj, ",")))

    return {"rows": rows, "n90": n90}


# ---------------------------------------------------------------- [8] 失败与售后

def part8_failure_and_service(traj=TRAJ, fail=FAIL,
                              total_hours=None, per_ep_s=60.0,
                              n_lang=75000, sec_per_lang=10.0,
                              n_recal=36000, sec_per_recal=120.0):
    if total_hours is None:
        total_hours = PEOPLE * MONTHS * WORKDAYS * WORKDAY

    print("\n[8] 失败数据与「售后」：小钱决定大事")
    print("-" * 74)
    rate = fail / float(traj + fail)
    fail_h = fail * per_ep_s / 3600.0
    print("  失败轨迹 %s 条 / 总 %s 条 = **%.1f%%** 失败率。"
          % (format(fail, ","), format(traj + fail, ","), rate * 100))
    print()
    print("  它的**时间成本**：每条轨迹的边际开销 %.0f 秒，" % per_ep_s)
    print("  所以失败样本只多吃掉 %.0f 小时，占总投入 %.2f%%。"
          % (fail_h, fail_h / total_hours * 100))
    print("  **失败数据在时间上几乎免费。** 丢掉它省不下钱，")
    print("  但会永久失去「怎么失败」这唯一一类标签。")
    print()
    print("  两次「售后」的时间账：")
    print()
    print("  %s%s%s" % (pad("更新", 22), pad("规模", 18), "折算人时"))
    print("  " + "-" * 60)

    lang_h = n_lang * 3 * sec_per_lang / 3600.0
    recal_h = n_recal * sec_per_recal / 3600.0
    print("  %s%s%s" % (pad("2024-12 补语言标注", 22),
                        pad("%s 条 × 3 句" % format(n_lang, ","), 18),
                        "%.0f h" % lang_h))
    print("  %s%s%s" % (pad("2025-04 重做标定", 22),
                        pad("%s 条" % format(n_recal, ","), 18),
                        "%.0f h" % recal_h))
    print()
    frac = (lang_h + recal_h) / total_hours
    print("  两笔加起来 %.0f 小时，是采集投入（%s 小时）的 **%.2f%%**。"
          % (lang_h + recal_h, format(int(total_hours), ","), frac * 100))
    print()
    print("  但第一笔决定了这 %s 条 episode 能不能用于**语言条件**模型 ——"
          % format(n_lang, ","))
    print("  没有指令，整个数据集对 VLA 就是无用的。第二笔让 36k 条的")
    print("  多视角几何重新可用。")
    print()
    print("  两件事都不是「采更多数据」，而是**回头修已有数据**。前提是原始")
    print("  数据还在：如果在转换阶段就把未去畸变的原始图像丢了，")
    print("  重做标定这件事就永远做不了。")

    return {"fail_rate": rate, "fail_hours": fail_h,
            "lang_h": lang_h, "recal_h": recal_h, "service_frac": frac}


# ---------------------------------------------------------------- 结论

def part9_conclusion(p1, p2, p3, p4, p5, p6, p7, p8):
    print("\n" + "=" * 74)
    print("结论（数据平台视角）")
    print("=" * 74)
    print("  1. 公开统计量要自己对账：350h / 76k 隐含 %.1f 秒/条，与"
          % p1["secs"])
    print("     「20 分钟/100 条」的 %.0f 秒差 %.0f%%。容量规划会因此差几十 TB。"
          % (p1["implied_secs"], p1["gap_frac"] * 100))
    print("  2. 采集成本是**人力在场时间**：有效率 %.2f%%，一个场景 %.0f 人时。"
          % (p2["eff"] * 100, p2["hours_per_scene"]))
    print("     %.0f 个工作日换 %.0f 分钟数据 —— 这就是 350 小时的来历。"
          % (p2["days_per_scene"], p1["per_scene_min"]))
    print("  3. 时间几乎全在**场景级**（≥%.0f%%），不在每条上。所以「多开一个"
          % (p3["lo_share"] * 100))
    print("     场景」和「多加一条 episode」的代价差 %s 倍。"
          % format(int(p3["mid"]["ratio"]), ","))
    print("  4. 每场景采多少条由**信息饱和**定，不由成本定：成本说该采更多，")
    print("     信息说 100 条就够（β=%.1f 时覆盖 %.1f%%）。约束是信息。"
          % (p4["key"], p4["cov100"] * 100))
    print("  5. 场景覆盖是饱和曲线：后 %.0f%% 的数据只多覆盖 10%% 的场景。"
          % (p5["tail_frac"] * 100))
    print("  6. 长尾形状决定总量：α 从 %.1f 到 %.1f，所需条数差 %.0f 倍。"
          % (p6["rows"][0]["alpha"], p6["rows"][-1]["alpha"], p6["ratio"]))
    print("  7. 失败数据的时间成本只有 %.2f%%，却是唯一的「怎么失败」标签。"
          % (p8["fail_hours"] / 100800 * 100))
    print("  8. 售后只花 %.2f%% 的追加投入，却决定另外 99%% 能不能用。"
          % (p8["service_frac"] * 100))
    print("     前提是原始数据还在 —— 保留原始数据就是保留修复能力。")
    print("=" * 74)


def main():
    print("=" * 74)
    print("实验 2：大规模采集的真实成本结构")
    print("=" * 74)
    print("  数据来源：DROID 论文与项目页（arXiv:2403.12945, 2024）")
    print("  76k 轨迹 / 350h / 564 场景 / 86 任务 / 52 栋建筑 /")
    print("  %d 采集员 / 18 实验室 / 13 机构 / %d 个月 / %d Hz / %d 路相机"
          % (PEOPLE, MONTHS, HZ, CAMS))

    p1 = part1_reconcile()
    p2 = part2_human_cost()
    p3 = part3_fixed_vs_marginal()
    p4 = part4_within_scene()
    p5 = part5_coverage_curve()
    p6 = part6_alpha()
    p7 = part7_budget()
    p8 = part8_failure_and_service()
    part9_conclusion(p1, p2, p3, p4, p5, p6, p7, p8)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验 2：大规模采集的真实成本结构

DROID 的公开数字：76k 条轨迹 / 350 小时 / 564 场景 / 86 任务 /
50 名采集员 / 18 个实验室 / 13 个机构 / 12 个月。

这些数字单独看没感觉，放在一起算，会得出几个很反直觉的结论：

  [1] 数字对账：平均每条多长、每场景多少条
  [2] 人力效率：0.35% —— 这才是数据平台的真实瓶颈
  [3] 场景覆盖曲线：90% 的 episode 只换来 10% 的场景
  [4] 每场景预算的取舍：集中采 vs 铺开采

[3][4] 是模型计算，不是 DROID 实测分布，文中已标注。

运行：python examples/04_datasets/02_collection_cost.py
依赖：无（只用标准库）
"""

import math


# ---------------------------------------------------------------- 输出对齐

def width(s):
    return sum(2 if ord(c) > 0x2E80 else 1 for c in s)


def pad(s, n):
    return s + " " * max(0, n - width(s))


# ---------------------------------------------------------------- 公开数字

EPISODES = 76_000        # 76k 条成功轨迹（另有约 16k 条失败）
HOURS = 350.0            # 350 小时交互数据
SCENES = 564             # 564 个场景
TASKS = 86               # 86 个任务
BUILDINGS = 52           # 52 栋建筑
COLLECTORS = 50          # 50 名采集员
LABS = 18                # 18 个实验室
MONTHS = 12              # 12 个月
HZ = 15                  # 15 Hz 控制与记录频率
CAMS = 3                 # 3 路立体相机


# ---------------------------------------------------------------- [1] 对账

def part1_reconcile():
    print("\n[1] 数字对账：把 76k / 350h / 564 放在一起看")
    print("-" * 78)

    sec_per_ep = HOURS * 3600 / EPISODES
    ep_per_scene = EPISODES / SCENES
    min_per_scene = HOURS * 60 / SCENES
    frames_per_ep = sec_per_ep * HZ
    total_frames = EPISODES * frames_per_ep

    rows = [
        ("平均每条轨迹时长", "%.1f 秒" % sec_per_ep,
         "350h ÷ 76k；和「抓一个杯子」的直觉一致"),
        ("每条轨迹的帧数", "%.0f 帧" % frames_per_ep,
         "%.1f 秒 × %d Hz" % (sec_per_ep, HZ)),
        ("总帧数（单路）", "%.1f M" % (total_frames / 1e6),
         "后面算存储和训练步数都要用"),
        ("总图像帧数", "%.1f M" % (total_frames * CAMS / 1e6),
         "× %d 路相机" % CAMS),
        ("每场景平均轨迹数", "%.0f 条" % ep_per_scene,
         "76k ÷ 564；官方描述是「每场景最多 100 条」"),
        ("每场景平均数据量", "%.0f 分钟" % min_per_scene,
         "350h ÷ 564"),
        ("每任务平均轨迹数", "%.0f 条" % (EPISODES / TASKS),
         "76k ÷ 86；但任务分布是长尾，平均值没意义"),
        ("每栋建筑平均场景数", "%.1f 个" % (SCENES / BUILDINGS),
         "564 ÷ 52"),
    ]

    print("  %s%s%s" % (pad("指标", 22), pad("数值", 14), "怎么算的 / 注意什么"))
    print("  " + "-" * 76)
    for name, val, note in rows:
        print("  %s%s%s" % (pad(name, 22), pad(val, 14), note))

    print()
    print("  注意「每场景 135 条」和官方描述的**内部不一致**：")
    print("    官方说「每人每场景最多 100 条轨迹，约 20 分钟数据」")
    print("    → 隐含每条 12 秒、每场景 20 分钟")
    print("    但 76k ÷ 564 = %.0f 条/场景，350h ÷ 564 = %.0f 分钟/场景，"
          % (ep_per_scene, min_per_scene))
    print("    且 350h ÷ 76k = %.1f 秒/条。按 135 条算应为 %.0f 分钟。"
          % (sec_per_ep, ep_per_scene * 12 / 60))
    print()
    print("  这不是谁算错了，而是这类数字本来就是**估算量级**：")
    print("  350h 可能是四舍五入，76k 是成功轨迹数，20 分钟是口语描述。")
    print("  **教训：拿公开数据集的统计量做容量规划时，先自己对账一遍。**")
    print("  差 40% 在存储预算上就是几十 TB。")


# ---------------------------------------------------------------- [2] 人力

def part2_human_cost():
    print("\n[2] 人力效率：每 288 小时人力，产出 1 小时数据")
    print("-" * 78)

    workdays = MONTHS * 21
    hours_per_person = workdays * 8
    total_person_hours = COLLECTORS * hours_per_person
    duty = HOURS / total_person_hours

    print("  名义投入：%d 人 × %d 个月 × 21 工作日 × 8 小时 = %s 人时"
          % (COLLECTORS, MONTHS, format(int(total_person_hours), ",")))
    print("  实际产出：%.0f 小时可用数据" % HOURS)
    print()
    print("  有效率 = %.0f / %s = **%.2f%%**"
          % (HOURS, format(int(total_person_hours), ","), duty * 100))
    print("  也就是 **每 %.0f 小时人力，产出 1 小时数据**。" % (1 / duty))
    print()

    ep_per_day = EPISODES / (COLLECTORS * workdays)
    sec_per_day = ep_per_day * (HOURS * 3600 / EPISODES)
    print("  换个说法，一名采集员一天：")
    print("    %.1f 条轨迹 = %.0f 秒有效数据" % (ep_per_day, sec_per_day))
    print("    占 8 小时工作时间的 %.2f%%" % (sec_per_day / (8 * 3600) * 100))
    print()
    print("  剩下 99.6% 的时间去哪了？看官方描述的流程就明白了：")
    print()
    waste = [
        ("把可移动平台推到新场景", "场景之间要搬运、调平、布线"),
        ("相机外参标定", "每个场景用棋盘格重标一次"),
        ("录入该场景的任务清单", "手工输入或从列表选择"),
        ("试采 + 调整视角", "找到能拍到有效行为的机位"),
        ("失败重采", "另有约 16k 条失败轨迹，不进去重后的成功集"),
        ("复位与场景增强", "移动底座、改光照、增删物品"),
    ]
    print("  %s" % pad("开销项", 24) + "说明")
    print("  " + "-" * 68)
    for a, b in waste:
        print("  %s%s" % (pad(a, 24), b))

    print()
    print("  这几项加起来才是数据采集的真实成本结构。所以：")
    print("    · 存储和带宽**不是**瓶颈 —— 350 小时的数据，一块硬盘就装下了")
    print("    · 瓶颈是**人在场景里的时间**，包括标定和复位")
    print("    · 任何能省掉标定/复位的工程手段，价值都高于优化存储格式")
    print()
    print("  这也解释了为什么「自动化数据采集」（用策略自己采、人类只做")
    print("  干预和纠错）是当前最热的方向：它砍掉的不是人手动作的时间，")
    print("  而是**场景切换和标定的固定开销**。")


# ---------------------------------------------------------------- [3] 覆盖

def zipf_probs(n, alpha):
    """n 个场景的 Zipf 分布概率。alpha 越大越集中。"""
    w = [1.0 / (i ** alpha) for i in range(1, n + 1)]
    tot = sum(w)
    return [x / tot for x in w]


def expected_distinct(probs, draws):
    """抽 draws 次，期望看到多少种不同的场景。"""
    return sum(1.0 - (1.0 - p) ** draws for p in probs)


def part3_coverage_curve():
    print("\n[3] 场景覆盖曲线：采样收益是饱和的，不是线性的")
    print("-" * 78)
    print("  模型：564 个场景的出现频率服从 Zipf 分布（长尾）。")
    print("  这是**模型假设**，不是 DROID 的实测分布 —— 但长尾这个形状")
    print("  在各类机器人数据集里是一致的（少数场景占大部分数据）。")
    print()

    probs_a1 = zipf_probs(SCENES, 1.0)

    print("  %s%s%s" % (pad("已采 episode", 16), pad("期望看到的场景数", 18), "覆盖率"))
    print("  " + "-" * 56)
    for n in (500, 1000, 2000, 5000, 10_000, 20_000, 40_000, 76_000):
        d = expected_distinct(probs_a1, n)
        print("  %s%s%s" % (pad(format(n, ","), 16), pad("%.0f" % d, 18),
                            "%.1f%%" % (d / SCENES * 100)))

    print()
    print("  边际收益（每 1000 条 episode 带来多少个新场景）：")
    print()
    print("  %s%s%s" % (pad("区间", 24), pad("新增场景", 14), "每千条新增"))
    print("  " + "-" * 60)
    bounds = [(0, 1_000), (1_000, 5_000), (5_000, 10_000),
              (10_000, 30_000), (30_000, 76_000)]
    for lo, hi in bounds:
        d_lo = expected_distinct(probs_a1, lo) if lo else 0.0
        d_hi = expected_distinct(probs_a1, hi)
        new = d_hi - d_lo
        per_k = new / ((hi - lo) / 1000)
        print("  %s%s%s" % (pad("%s – %s" % (format(lo, ","), format(hi, ",")), 24),
                            pad("%.0f 个" % new, 14), "%.1f 个" % per_k))

    print()
    n90 = _solve_for(probs_a1, 0.90)
    n50 = _solve_for(probs_a1, 0.50)
    print("  覆盖 50%% 的场景需要 %s 条；覆盖 90%% 需要 %s 条。"
          % (format(int(n50), ","), format(int(n90), ",")))
    print("  而实际采了 %s 条 —— 后 %s 条（占 %.0f%% 的数据）"
          % (format(EPISODES, ","), format(int(EPISODES - n90), ","),
             (EPISODES - n90) / EPISODES * 100))
    print("  只多覆盖了最后 10% 的场景。")
    print()
    print("  这看起来很浪费，但它是**对的**。原因在下一节：")
    print("  「覆盖多少场景」和「覆盖哪些场景」是两件事。")

    print("\n  敏感度：如果长尾更极端（alpha 更大），结论会怎么变？")
    print()
    print("  %s%s%s" % (pad("alpha", 10), pad("覆盖 90% 场景所需", 20), "含义"))
    print("  " + "-" * 58)
    for a in (0.8, 1.0, 1.2, 1.5):
        p = zipf_probs(SCENES, a)
        n = _solve_for(p, 0.90)
        desc = {0.8: "分布较均匀，好采", 1.0: "基准",
                1.2: "很集中，尾部难碰到", 1.5: "极集中"}[a]
        print("  %s%s%s" % (pad("%.1f" % a, 10),
                            pad("%s 条" % format(int(n), ","), 20), desc))
    print()
    print("  alpha 从 0.8 到 1.5，所需数量差一个数量级。")
    print("  所以在设计采集计划之前，**先摸清自己的场景分布有多长尾** ——")
    print("  这个数决定你需要 1 万条还是 10 万条。")


def _solve_for(probs, target):
    """二分找出达到 target 覆盖率所需的采样次数。"""
    lo, hi = 0.0, 1e7
    for _ in range(60):
        mid = (lo + hi) / 2
        if expected_distinct(probs, mid) / len(probs) < target:
            lo = mid
        else:
            hi = mid
    return hi


# ---------------------------------------------------------------- [4] 取舍

def part4_budget_tradeoff():
    print("\n[4] 每场景预算的取舍：100 条/场景 vs 1000 条/场景")
    print("-" * 78)
    print("  固定总量 %s 条 episode，问：每个场景花多少条？" % format(EPISODES, ","))
    print("  场景数 = 总量 ÷ 每场景预算（上限 %d 个场景）。" % SCENES)
    print()

    probs = zipf_probs(SCENES, 1.0)
    print("  %s%s%s%s" % (pad("每场景预算", 14), pad("能去的场景数", 14),
                          pad("覆盖场景质量", 16), "含义"))
    print("  " + "-" * 74)
    for budget in (50, 100, 200, 500, 1000, 2000):
        k = min(SCENES, int(EPISODES // budget))
        # 按频率降序访问 k 个场景，覆盖了多少「场景质量」
        mass = sum(probs[:k])
        note = ("接近官方做法（实际 %d 个场景、%.0f 条/场景）"
                % (SCENES, EPISODES / SCENES)) if budget == 100 else ""
        print("  %s%s%s%s" % (pad("%d 条" % budget, 14), pad("%d 个" % k, 14),
                              pad("%.0f%%" % (mass * 100), 16), note))

    print()
    print("  从 100 条/场景 降到 1000 条/场景：场景数从 %d 掉到 %d，"
          % (SCENES, int(EPISODES // 1000)))
    print("  覆盖的「场景质量」从 100%% 掉到 %.0f%%。"
          % (sum(probs[:int(EPISODES // 1000)]) * 100))
    print()
    print("  但这里有个陷阱：上面的百分比是**按出现频率加权**的，")
    print("  它衡量「你见过多少数据」，不衡量「你见过多少种罕见行为」。")
    print("  尾部 400 个场景按出现频率算不足 10%，但按信息量算是**最重要的** ——")
    print("  「在窄柜子里叠衣服」这种行为只存在于尾部场景。")
    print()
    print("  所以真实的取舍不是「覆盖率最大化」，而是：")
    print("    · 头部场景（厨房台面、桌面）→ 少量多条，学好基础动作")
    print("    · 尾部场景（异形空间、特殊物体）→ 每个都要去，哪怕只采几十条")
    print("    · 每场景条数的下限由「动作多样性」定，不由「数据量」定")
    print()
    print("  DROID 的「每场景最多 100 条就换地方」正是这个逻辑的产物：")
    print("  100 条 ≈ 20 分钟，足够覆盖该场景的主要物体和姿态组合；")
    print("  再多采，新增的是同一个分布里的重复样本，边际价值急剧下降。")


# ---------------------------------------------------------------- 结论

def summary():
    print("\n" + "=" * 78)
    print("结论（数据平台视角）")
    print("=" * 78)
    print("  1. 公开数据集的统计量要自己对账。350h 和 76k 隐含 16.6 秒/条，")
    print("     与「20 分钟/100 条」的 12 秒差 40% —— 容量规划会因此差几十 TB。")
    print("  2. 采集的真实成本是**人力在场时间**，有效率约 0.35%。")
    print("     存储、带宽、算力都不是瓶颈 —— 别在错误的地方优化。")
    print("  3. 场景覆盖是**饱和曲线**：90% 的量只多覆盖 10% 的场景。")
    print("     所以「再加 10 万条同场景数据」和「再去 10 个新场景」")
    print("     是完全不同量级的收益，不能用一个「样本数」指标比较。")
    print("  4. 每场景条数的下限由动作多样性决定。这个下限只能靠看数据定，")
    print("     不能靠公式算 —— 这是数据平台里最需要人来判断的地方。")


def main():
    print("=" * 78)
    print("实验 2：大规模采集的真实成本结构")
    print("=" * 78)
    print("  数据来源：DROID 论文与项目页（arXiv:2403.12945, 2024）")
    print("  76k 轨迹 / 350h / 564 场景 / 86 任务 / 52 栋建筑 /")
    print("  50 采集员 / 18 实验室 / 13 机构 / 12 个月 / 15 Hz / 3 路相机")
    part1_reconcile()
    part2_human_cost()
    part3_coverage_curve()
    part4_budget_tradeoff()
    summary()


if __name__ == "__main__":
    main()

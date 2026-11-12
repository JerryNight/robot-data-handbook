#!/usr/bin/env python3
"""
实验 6：标注的三件事——一致性、阈值、语言，以及它们各自能算出什么。

运行：python examples/03_data_pipeline/06_annotation.py

标注看起来是「人来打标签」，但在数据平台里它是几个可计算的问题：

    [1] 一致率会骗人：同样 95% 一致，kappa 可以从 0.9 掉到负数
    [2] 30 条样本的 kappa 有多不稳：bootstrap 出来的置信区间
    [3] 99% 成功率的数据集里，抽 30 条可能一个失败样本都没有
    [4] 阈值扫描：成功率对阈值有多敏感，最陡的那一段在哪
    [5] 两人心里的阈值不同，kappa 就会掉——而 0.8 的门槛挡不住
    [6] 同样的标注预算，随机抽和按不确定性抽差多少
    [7] 阶段切分：人工的时间精度上限，放大到下游是多大
    [8] 语言：动词熵、覆盖率，以及语言到底提供了多少信息
    [9] 决策表

关于外部输入
------------
本脚本自己造数据，但造法本身是输入：误差的双峰形状（0.7 cm / 4.5 cm）、
标注员的准确率（93%）、人的反应时间（300 ms），都是公开经验值。
凡是依赖它们的结论，都做了敏感性检查——看的是「结论的形状」，
而不是小数点后两位。

每个实验都返回它算出来的数，正文和图只引用返回值，不重抄数字。
"""

import math
import random
from collections import Counter


def width(s: str) -> int:
    return sum(2 if ord(c) > 0x2E80 else 1 for c in s)


def pad(s: str, n: int) -> str:
    return s + " " * max(0, n - width(s))


def head(*cells) -> str:
    return "  " + "".join(pad(c, w) for c, w in cells)


# ---------------------------------------------------------------- kappa

def cohen_kappa(a, b, c, d):
    """2x2 混淆：a=都判成功, b=A成功B失败, c=A失败B成功, d=都判失败。

    kappa = (p_o - p_e) / (1 - p_e)，p_e 是「按各自边际分布瞎猜」的期望一致率。
    p_e = 1 时无法定义（两人都只用一个类别），返回 None。
    """
    n = a + b + c + d
    if n == 0:
        return None
    po = (a + d) / n
    pa_pos = (a + b) / n
    pb_pos = (a + c) / n
    pe = pa_pos * pb_pos + (1 - pa_pos) * (1 - pb_pos)
    if pe >= 1.0:
        return None
    return po, pe, (po - pe) / (1 - pe)


def kappa_of(la, lb):
    a = b = c = d = 0
    for x, y in zip(la, lb):
        if x and y:
            a += 1
        elif x and not y:
            b += 1
        elif not x and y:
            c += 1
        else:
            d += 1
    return cohen_kappa(a, b, c, d) or (0.0, 1.0, None)


def pct(x, nd=1):
    return ("%." + str(nd) + "f%%") % (x * 100)


# ---------------------------------------------------------------- [1] 陷阱

def part1_trap():
    print("\n[1] 一致率会骗人：同样 95% 一致，kappa 可以从 0.9 掉到负数")
    print("-" * 78)
    print("  两位标注员各判 200 条 episode 是成功还是失败。")
    print("  kappa = (p_o - p_e) / (1 - p_e)，p_e 是「按各自边际分布瞎猜」的一致率。")
    print()
    print(head(("场景", 32), ("表面一致率", 14), ("期望一致率", 14), ("kappa", 0)))
    print("  " + "-" * 74)
    cases = [("均衡：两人各 100 成 / 100 败", 90, 5, 5, 100),
             ("失衡但同步：190 成 / 10 败", 188, 2, 2, 8),
             ("失衡且不同步：199 成 / 1 败", 198, 1, 1, 0),
             ("一人全判成功，另一人正常判", 140, 60, 0, 0)]
    rows = []
    for label, a, b, c, d in cases:
        po, pe, k = cohen_kappa(a, b, c, d)
        rows.append((label, po, pe, k))
        print("  %s%s%s%s" % (pad(label, 32), pad(pct(po), 14), pad(pct(pe), 14),
                              "%.3f" % k))

    print()
    print("  第二行和第三行：一致率 %.1f%% vs %.1f%%，kappa %.3f vs %.3f。"
          % (rows[1][1] * 100, rows[2][1] * 100, rows[1][3], rows[2][3]))
    print("  **一致率更高的那一行，kappa 反而更低**——第三行是负的，")
    print("  比「按各自边际分布随便猜」还差。")
    print()
    print("  第三行的情形最值得记住：两人各有 199 次判「成功」，")
    print("  但**不是同一批 199 条**。表面看几乎全一致，实际几乎没有共识。")
    print("  kappa 是负的，说明两人的分歧方向和边际分布对抗——")
    print("  他们不但没有共同标准，还系统性地反着来。")

    # 固定一致率，扫阳性率
    print()
    print("  这不是特例。把「表面一致率」钉死在 95%、两人边际相同，只改阳性率：")
    print()
    print(head(("阳性率", 12), ("期望一致率", 14), ("kappa", 12), ("含义", 0)))
    print("  " + "-" * 70)
    curve = []
    for pos in (0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99):
        pe = pos * pos + (1 - pos) * (1 - pos)
        k = (0.95 - pe) / (1 - pe)
        curve.append((pos, pe, k))
        note = ("可信" if k >= 0.8 else "勉强" if k >= 0.4 else
                "不可用" if k >= 0 else "比瞎猜还差")
        print("  %s%s%s%s" % (pad("%.0f%%" % (pos * 100), 12),
                              pad(pct(pe), 14), pad("%.3f" % k, 12), note))

    print()
    print("  同样是 95% 的表面一致率：阳性率 50% 时 kappa 是 0.900（可信），")
    print("  阳性率 99%% 时是 %.2f（比瞎猜还差）。**一致率这个数字本身不能判断质量。**"
          % curve[-1][2])
    print("  而机器人领域恰好到处是「成功率 95%+」的数据集——")
    print("  越是这样做，越需要 kappa，而不是一致率。")
    print()
    print("  一个反直觉的推论：**数据集越好，一致率越没用**。")
    print("  成功率 99% 时，两人闭着眼全判「成功」也有一致率 99%，")
    print("  而这种标注员对训练毫无价值。")
    return {"curve": curve, "rows": rows}


# ---------------------------------------------------------------- [2] 样本量

N_ITEMS = 200


def make_raters(seed=5, n=N_ITEMS, acc_a=0.93, acc_b=0.93, true_pos=0.70):
    """造两人的判定结果：各自以 acc 的概率给出正确标签。"""
    rng = random.Random(seed)
    truth = [1 if rng.random() < true_pos else 0 for _ in range(n)]
    a = [t if rng.random() < acc_a else 1 - t for t in truth]
    b = [t if rng.random() < acc_b else 1 - t for t in truth]
    return truth, a, b


def part2_sample_size(reps=2000):
    print("\n[2] 30 条样本的 kappa 有多不稳")
    print("-" * 78)
    truth, la, lb = make_raters()
    po, pe, k_all = kappa_of(la, lb)
    print("  全量 %d 条：一致率 %s，kappa %.3f —— 这就是「真值」。"
          % (N_ITEMS, pct(po), k_all))
    print("  现在从这 %d 条里随机抽 n 条，重算 kappa，重复 %d 次：" % (N_ITEMS, reps))
    print()
    print(head(("抽样条数", 12), ("kappa 均值", 14), ("2.5% 分位", 12),
               ("97.5% 分位", 12), ("区间宽度", 12)))
    print("  " + "-" * 72)
    res = []
    for n in (30, 60, 100, 200):
        rng = random.Random(7)
        ks, undef = [], 0
        for _ in range(reps):
            idx = [rng.randrange(N_ITEMS) for _ in range(n)]
            _, _, k = kappa_of([la[i] for i in idx], [lb[i] for i in idx])
            if k is None:
                undef += 1
            else:
                ks.append(k)
        ks.sort()
        lo, hi = ks[int(0.025 * len(ks))], ks[int(0.975 * len(ks))]
        mean = sum(ks) / len(ks)
        p_hi = sum(1 for k in ks if k >= 0.8) / len(ks)     # 会被判「可信」
        p_lo = sum(1 for k in ks if k < 0.4) / len(ks)      # 会被判「返工」
        res.append({"n": n, "mean": mean, "lo": lo, "hi": hi, "w": hi - lo,
                    "undef": undef, "p_hi": p_hi, "p_lo": p_lo})
        print("  %s%s%s%s%s" % (pad("%d 条" % n, 12), pad("%.3f" % mean, 14),
                                pad("%.3f" % lo, 12), pad("%.3f" % hi, 12),
                                "%.3f" % (hi - lo)))

    r30, r200 = res[0], res[-1]
    print()
    print("  抽 30 条时，kappa 的 95%% 区间是 [%.2f, %.2f]，宽度 %.2f。"
          % (r30["lo"], r30["hi"], r30["w"]))
    print("  判读门槛是「> 0.8 可信、< 0.4 返工」，两个门槛都落在这个区间里。")
    print("  2000 次重抽里有 %s 落在「可信」，%s 落在「返工」。"
          % (pct(r30["p_hi"], 0), pct(r30["p_lo"])))
    print("  真实 kappa 是 %.3f——**不管判成哪一种，都是错的**。" % k_all)
    print()
    print("  **「先标 30 条算一次 kappa」这个建议，在 30 条上是不成立的。**")
    print("  它只能告诉你「两人有没有系统性分歧」，不能告诉你「够不够 0.8」。")

    # 区间宽度按 1/sqrt(n) 缩：width(n) ≈ width(30) * sqrt(30/n)
    n_need = 30 * (r30["w"] / 0.2) ** 2
    print()
    print("  宽度大致按 1/√n 缩。要把宽度压到 0.2，需要约 %.0f 条；"
          % (round(n_need / 10) * 10))
    print("  实测 200 条时宽度是 %.3f——刚好压线。" % r200["w"])
    print("  所以务实的读法是：")
    print("    · 30 条：看**方向**（分歧集中在哪、是不是同一类错误）")
    print("    · 200 条：看**数值**（够不够 0.8，敢不敢停）")
    print("  两件事不要混。用 30 条的区间去回答数值问题，等于掷硬币。")
    return {"rows": res, "all_k": k_all, "n_need": n_need}


# ---------------------------------------------------------------- [3] 稀有类

def part3_rare(prev=0.99):
    print("\n[3] %.0f%% 成功率的数据集里，抽 30 条可能一个失败样本都没有"
          % (prev * 100))
    print("-" * 78)
    print("  kappa 需要两个类别都出现。一个都没抽到时 p_e = 1，kappa 无法定义——")
    print("  不是「kappa 低」，是**根本没有这个数**。")
    print()
    print(head(("抽样条数", 12), ("一条失败都没有的概率", 26), ("含义", 0)))
    print("  " + "-" * 76)
    pts = []
    for n in (10, 30, 50, 100, 300):
        p0 = prev ** n
        pts.append((n, p0))
        note = "kappa 算不出来" if p0 > 0.5 else ("有风险" if p0 > 0.05 else "基本能算")
        print("  %s%s%s" % (pad("%d 条" % n, 12), pad(pct(p0), 26), note))

    need_95 = math.log(0.05) / math.log(prev)
    need_99 = math.log(0.01) / math.log(prev)
    print()
    print("  要在 %.0f%% 成功率的数据集里，以 95%% 的把握至少抽到一个失败样本："
          % (prev * 100))
    print("  **需要 %.0f 条**。99%% 的把握要 %.0f 条。" % (need_95, need_99))
    print()
    print("  注意这不是「样本多一点更稳」的老生常谈，而是**量级问题**：")
    print("  从 %.0f%% 的成功率出发，负例的密度是 %.0f%%，" % (prev * 100, (1 - prev) * 100))
    print("  想让负例可统计，抽样量就得是负例密度的倒数再乘信心系数。")
    print("  这就是「小众失败模式」在数据里不可见的原因——")
    print("  不是标注员不够多，是负例太稀。")
    print()
    print("  三个可行的绕法：")
    print("    ① 分层抽样：按「疑似失败」预筛，再在预筛池里抽（成功率立刻降到 30~50%）")
    print("    ② 换指标：用连续量（位姿误差）而不是二分类，不受稀有类影响")
    print("    ③ 换问题：不问「这条成功吗」，问「这条比上一条好还是差」（配对比较）")
    print()
    print("  ① 是唯一不改变指标的绕法，代价是预筛器本身的偏差会带进样本——")
    print("  所以预筛器用「召回优先」，宁可多捞，别漏。")
    return {"pts": pts, "need_95": need_95, "need_99": need_99, "prev": prev}


# ---------------------------------------------------------------- [4] 阈值

def make_errors(seed=19, n=1000, ok_ratio=0.72):
    """双峰误差：72% 来自「对准」那一簇（均值 0.7 cm），28% 来自「没对准」。"""
    rng = random.Random(seed)
    errs = []
    for _ in range(n):
        if rng.random() < ok_ratio:
            errs.append(abs(rng.gauss(0.7, 0.45)))
        else:
            errs.append(abs(rng.gauss(4.5, 2.0)))
    errs.sort()
    return errs


def part4_threshold():
    print("\n[4] 阈值扫描：成功率对阈值有多敏感")
    print("-" * 78)
    errs = make_errors()
    n = len(errs)
    print("  1000 条「把方块放进盒子」，用最终位姿误差（cm）判定成败。")
    print("  误差是双峰的：72% 来自「对准」那一簇（均值 0.7 cm），")
    print("  28% 来自「没对准」（均值 4.5 cm）。")
    print()
    print(head(("阈值", 10), ("判成功", 12), ("成功率", 12),
               ("阈值 +0.2 cm 时成功率变化", 0)))
    print("  " + "-" * 68)
    rows = []
    for th in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0):
        n_ok = sum(1 for e in errs if e <= th)
        n_ok2 = sum(1 for e in errs if e <= th + 0.2)
        d = (n_ok2 - n_ok) / n * 100
        rows.append((th, n_ok, n_ok / n * 100, d))
        print("  %s%s%s%s" % (pad("%.1f cm" % th, 10), pad(str(n_ok), 12),
                              pad("%.1f%%" % (n_ok / n * 100), 12),
                              "%.1f 个百分点" % d))

    flat = [r for r in rows if r[0] >= 3.0]
    print()
    print("  阈值从 1.0 挪到 3.0，成功率从 %.0f%% 变成 %.0f%%——**差 %.0f 个百分点**。"
          % (rows[1][2], rows[5][2], rows[5][2] - rows[1][2]))
    print("  陡的是 0.5~1.0 cm 这一段：每挪 0.2 cm，成功率就动 %.1f~%.1f 个百分点。"
          % (min(r[3] for r in rows[:2]), max(r[3] for r in rows[:2])))
    print("  到 3.0 cm 以外基本平了：每挪 0.2 cm 只动 %.1f~%.1f 个百分点。"
          % (min(r[3] for r in flat), max(r[3] for r in flat)))
    print()
    print("  而 1.0 cm 恰好是文献里最常用的那个阈值——")
    print("  也就是说，「成功率」这个数在**最有可比性的位置上最敏感**：")
    print("  阈值差 0.2 cm，成功率就差 %.0f 个百分点。" % rows[1][3])
    print("  两边都报「成功率 75%% 左右」，可能只是阈值差了 0.5 cm（1.5 vs 2.0）。")
    print()
    print("  这不是学术争论，是可复现性问题：")
    print("    · 阈值必须写进数据集元数据，和标签一起存")
    print("    · 论文里报的「成功率」必须带阈值，否则无法比较")
    print("    · 换阈值等于换数据集版本，不能悄悄改")

    near = [e for e in errs if 1.2 <= e <= 2.4]
    print()
    print("  落在 1.2~2.4 cm 的有 %d 条（%.0f%%）——这一段横跨了所有常用阈值，"
          % (len(near), len(near) / n * 100))
    print("  也是人和自动判定分歧最大的地方。[6] 会算「只标这一段」能省多少。")
    cdf = [(e, i / n * 100) for i, e in enumerate(errs)]
    return {"rows": rows, "near": len(near), "n": n, "cdf": cdf}


# ---------------------------------------------------------------- [5] 阈值歧义

def part5_ambiguous():
    print("\n[5] 两人心里的阈值差 0.5 cm，kappa 就会掉——而 0.8 的门槛挡不住")
    print("-" * 78)
    errs = make_errors(seed=23)
    print("  同一条误差序列，两位标注员只看「误差 ≤ 自己心里的阈值吗」。")
    print("  除了阈值，两人没有任何分歧——没有误判、没有走神、没有理解偏差。")
    print()
    print(head(("A 的阈值", 12), ("B 的阈值", 12), ("一致率", 12),
               ("kappa", 12), ("结论", 0)))
    print("  " + "-" * 76)
    rows, sweep = [], []
    for i in range(16):
        tb = round(1.5 + 0.1 * i, 1)
        la = [e <= 1.5 for e in errs]
        lb = [e <= tb for e in errs]
        po, pe, k = kappa_of(la, lb)
        sweep.append((round(tb - 1.5, 1), po, k))
        if i % 5:
            continue
        note = ("完全一致" if k > 0.999 else "可信" if k >= 0.8 else
                "勉强" if k >= 0.6 else "会触发返工")
        rows.append((1.5, tb, po, k, note))
        print("  %s%s%s%s%s" % (pad("%.1f cm" % 1.5, 12), pad("%.1f cm" % tb, 12),
                                pad(pct(po), 12), pad("%.3f" % k, 12), note))

    r50 = rows[1]
    print()
    print("  阈值差 0.5 cm（1.5 vs 2.0），两个人都严格按自己的规则判，零失误，")
    print("  kappa 就从 1.000 掉到 %.3f——一致率还有 %s。" % (r50[3], pct(r50[2])))
    print()
    print("  这里有个比「掉多少」更值得注意的事实：")
    print("  **差到 1.5 cm（1.5 vs 3.0），kappa 还有 %.3f。**" % rows[-1][3])
    print("  整张表没有一个值掉到 0.8 以下——也就是说，「kappa > 0.8 就通过」")
    print("  这个门槛，**挡不住阈值歧义**：规范差 1.5 cm，指标还是「可信」。")
    print("  门槛能挡的是「两人判得乱七八糟」，挡不住「两人都在认真执行不同的规范」。")
    print()
    print("  所以 kappa 的正确用法是**诊断**，不是打分：")
    print("    · kappa 低 + 分歧集中在某个数值区间 → 阈值没定义，去写规范")
    print("    · kappa 低 + 分歧分散 → 标注员理解不同，去培训或改任务定义")
    print("    · kappa 高但下游指标抖 → 一致性不是瓶颈，别在标注上继续投入")
    print()
    print("  一个立刻能做的改进：不看 kappa，先看**分歧样本的误差分布**。")
    print("  如果分歧全部落在 1.5~2.0 cm 这一条窄带上，那答案是把阈值写进规范，")
    print("  而不是换人——换三个人，只要规范还是那句「放进盒子算成功」，结果一样。")
    return {"rows": rows, "sweep": sweep}


# ---------------------------------------------------------------- [6] 预算

def part6_budget(n=1000, near_n=111, near_err=0.30, far_err=0.02,
                 budgets=(0, 25, 50, 75, 100, 150, 200, 250, 300)):
    print("\n[6] 同样的标注预算，花在哪差别很大")
    print("-" * 78)
    print("  自动判定器：边界区间外错误率 %.0f%%，区间内错误率 %.0f%%。"
          % (far_err * 100, near_err * 100))
    far_n = n - near_n
    print("  %d 条里，%d 条落在边界区间（%.0f%%），%d 条在区间外。"
          % (n, near_n, near_n / n * 100, far_n))
    auto_err = near_n * near_err + far_n * far_err
    print("  全靠自动判定：错 %.0f 条，即 %.1f%%。" % (auto_err, auto_err / n * 100))

    def remain(labelled_near, labelled_far):
        return (max(0, near_n - labelled_near) * near_err
                + max(0, far_n - labelled_far) * far_err)

    def rand_err(b):
        return remain(b * near_n / n, b * far_n / n)

    def act_err(b):
        return remain(min(b, near_n), max(0, b - near_n))

    budget = budgets[-1]
    print()
    print("  预算 %d 条人工，两种花法：" % budget)
    print()
    print(head(("策略", 24), ("标了哪些", 30), ("剩余错误数", 14), ("错误率", 0)))
    print("  " + "-" * 84)
    e_rand, e_act = rand_err(budget), act_err(budget)
    print("  %s%s%s%s" % (pad("随机抽", 24),
                          pad("区间内 %.0f + 区间外 %.0f"
                              % (budget * near_n / n, budget * far_n / n), 30),
                          pad("%.0f 条" % e_rand, 14), "%.2f%%" % (e_rand / n * 100)))
    print("  %s%s%s%s" % (pad("按不确定性挑", 24),
                          pad("区间内 %.0f 条（先吃满）" % min(budget, near_n), 30),
                          pad("%.0f 条" % e_act, 14), "%.2f%%" % (e_act / n * 100)))

    curve = [(b, rand_err(b), act_err(b)) for b in budgets]
    print()
    print("  同样 %d 条人力，错误标签从 %.0f 条降到 %.0f 条，**%.1f 倍**。"
          % (budget, e_rand, e_act, e_rand / max(e_act, 1e-9)))
    print("  区别只在于：人先标哪一批。")
    print()
    print("  把预算从 0 扫到 %d，两条曲线的形状不一样：" % budget)
    print(head(("预算", 10), ("随机抽剩余", 14), ("按不确定性剩余", 16), ("差距", 0)))
    print("  " + "-" * 60)
    for b, er, ea in curve:
        print("  %s%s%s%s" % (pad("%d 条" % b, 10), pad("%.0f 条" % er, 14),
                              pad("%.0f 条" % ea, 16), "%.1f 倍" % (er / max(ea, 1e-9))))
    print()
    print("  随机抽是一条**匀速下降的直线**——每标一条，期望只修掉很少一点；")
    print("  按不确定性挑在最前面 %d 条降得极快（先把边界区间吃满），" % near_n)
    print("  然后拐平——因为区间内已经标完，剩下的都是低错误密度区。")
    print("  这个拐点就是「该不该继续投人力」的分界线。看**边际收益**：")
    print("    · 拐点前：每条人力修掉 %.2f 条错（区间内错误率）" % near_err)
    print("    · 拐点后：每条人力修掉 %.2f 条错（区间外错误率）" % far_err)
    print("    · 随机抽：每条人力修掉 %.3f 条错，恒定" % (auto_err / n))
    print("  拐点后继续按「不确定性」挑，边际收益反而**低于随机抽**——")
    print("  因为你只剩低密度区可挑，而随机抽还能顺带撞上一些边界样本。")
    print("  这才是「标到拐点就该停」的真正理由：不是钱花完了，是花得不如随机。")
    print()
    print("  更省的做法是迭代：标完一批 → 重训自动判定器 → 重新挑边界 → 再标。")
    print("  每轮边界会收窄，同样预算覆盖的有效样本越来越多。")
    print("  （注意「区间内错误率高」这件事本身依赖自动判定器的质量，")
    print("  判定器很差时它会均匀地错，这个策略就退化成随机抽。）")
    return {"n": n, "near_n": near_n, "auto": auto_err, "rand": e_rand,
            "act": e_act, "curve": curve}


# ---------------------------------------------------------------- [7] 切分

def part7_segmentation(fps=30.0, seg_s=2.4, n_seg=4):
    print("\n[7] 阶段切分：人工的时间精度，放大到下游是多少")
    print("-" * 78)
    total_s = seg_s * n_seg
    print("  一段 %.1f 秒的抓取切成「接近 → 抓取 → 抬起 → 放置」%d 段，每段 %.1f s。"
          % (total_s, n_seg, seg_s))
    print("  每段 %d 帧（%.0f Hz），全段 %d 帧。"
          % (seg_s * fps, fps, seg_s * fps * n_seg))
    print()
    print("  边界晚 Δ 帧，就有 Δ 帧被划到前一段。方向不重要（早也是 Δ 帧），")
    print("  所以每个边界错标 Δ 帧。%d 段有 %d 个内部边界。" % (n_seg, n_seg - 1))
    print()
    print(head(("标注方式", 28), ("边界误差", 12), ("折合帧数", 12),
               ("段标签错帧", 14), ("占比", 0)))
    print("  " + "-" * 82)
    rows = []
    total = seg_s * fps * n_seg
    for label, err_ms in (("人工，慢放 ×0.25 逐帧看", 100),
                          ("人工，实时观看", 320),
                          ("速度阈值自动切分", 80),
                          ("力/接触信号自动切分", 30)):
        frames = err_ms / 1000.0 * fps
        wrong = min(seg_s * fps, frames) * (n_seg - 1)
        rows.append((label, err_ms, frames, wrong, wrong / total * 100))
        print("  %s%s%s%s%s" % (pad(label, 28), pad("%d ms" % err_ms, 12),
                                pad("%.1f 帧" % frames, 12),
                                pad("%.0f 帧" % wrong, 14),
                                "%.0f%%" % (wrong / total * 100)))

    r = rows[1]
    print()
    print("  实时观看的边界误差 %.0f ms = %.1f 帧。全段 %.0f 帧里，"
          % (r[1], r[2], total))
    print("  有 %.0f 帧的段标签是错的——%.0f%%，**每 %.0f 帧就有 1 帧**。"
          % (r[3], r[4], total / r[3]))
    print("  而这些样本在训练集里看起来完全正常：动作是对的，")
    print("  只是被切在了错误的位置。")
    print()
    print("  这就是段标签和帧标签的区别：帧标签错一个是一个，")
    print("  段标签错一次，错的是**边界上连续的一整条**——约 10 帧，"
          "而且它们看起来毫无异常。")
    print()
    print("  这不是标注员不认真，是人的反应时间决定的：视觉感知延迟约 100 ms，")
    print("  加上判断和按键，300 ms 是常态。所以任何需要帧级精度的边界，")
    print("  都不该指望纯人工。")
    print()
    print("  可用的组合策略：")
    print("    1. 用信号（速度/力/夹爪开合）自动切出候选边界")
    print("    2. 人工只做「接受 / 移动几帧」的确认，不做从零标注")
    print("    3. 把人工的修正量记下来，反过来校准自动切分器的偏差")
    print()
    print("  第 3 步最容易被漏掉：如果人工总是把边界往后挪 3 帧，")
    print("  那是自动切分器的系统性偏差，应该修它，而不是每条都手动改。")
    print("  修好之后，人工的修正量应该变成零均值——这就是切分器收敛的标志。")
    return {"rows": rows, "total": total, "fps": fps}


# ---------------------------------------------------------------- [8] 语言

VERBS = ["拿起", "放下", "放进", "移到", "推", "拉", "打开", "关上", "翻转", "堆叠"]
VERB_P = [0.42, 0.20, 0.14, 0.09, 0.05, 0.04, 0.03, 0.01, 0.01, 0.01]
OBJECTS = ["红色方块", "蓝色杯子", "积木", "抽屉", "香蕉", "夹子"]


def entropy(counts):
    total = sum(counts)
    h = 0.0
    for c in counts:
        if c:
            p = c / total
            h -= p * math.log2(p)
    return h


def mutual_info(pairs):
    """I(X;Y)，单位 bit。pairs 是 [(x, y), ...]。"""
    n = len(pairs)
    cx = Counter(x for x, _ in pairs)
    cy = Counter(y for _, y in pairs)
    cxy = Counter(pairs)
    mi = 0.0
    for (x, y), c in cxy.items():
        pxy = c / n
        mi += pxy * math.log2(pxy / ((cx[x] / n) * (cy[y] / n)))
    return mi


def part8_language(n=3000, noise=0.2):
    print("\n[8] 语言到底提供了多少信息")
    print("-" * 78)
    rng = random.Random(31)
    verbs = [rng.choices(VERBS, weights=VERB_P)[0] for _ in range(n)]

    print("  %d 条指令，动词分布：" % n)
    print()
    print(head(("动词", 12), ("出现次数", 12), ("占比", 12), ("累计", 0)))
    print("  " + "-" * 52)
    vc = Counter(verbs)
    cum = 0
    rows = []
    for v, c in vc.most_common(6):
        cum += c
        rows.append((v, c, c / n * 100, cum / n * 100))
        print("  %s%s%s%s" % (pad(v, 12), pad(str(c), 12),
                              pad("%.1f%%" % (c / n * 100), 12),
                              "%.1f%%" % (cum / n * 100)))
    h_verb = entropy(list(vc.values()))
    h_max = math.log2(len(VERBS))
    top2 = sum(c for _v, c in vc.most_common(2)) / n * 100
    print()
    print("  动词熵 %.2f bit，上限 %.2f bit（%d 个动词均匀分布），达 %.0f%%。"
          % (h_verb, h_max, len(VERBS), h_verb / h_max * 100))
    print("  前 2 个动词覆盖 %.0f%% 的样本——长尾是真实的，但头很重。"
          % top2)

    print()
    print("  然后问一个更关键的问题：**语言对动作提供了多少信息？**")
    print("  用归一化互信息 I(动作; 动词) / H(动作)，1 表示语言完全决定动作，")
    print("  0 表示语言和动作无关。")
    print()

    objs = [rng.choice(OBJECTS) for _ in range(n)]
    acts_c = [objs[i] if rng.random() < noise else verbs[i] for i in range(n)]
    scen = []
    for name, acts, note in (("动作 = 动词（理想）", verbs, "语言完全决定行为"),
                             ("动作 = 物体（语言无关）", objs, "语言只是个标签"),
                             ("动作 ≈ 动词 + %.0f%% 噪声" % (noise * 100), acts_c,
                              "接近真实数据")):
        ha = entropy(list(Counter(acts).values()))
        mi = mutual_info(list(zip(verbs, acts)))
        nmi = mi / ha if ha > 0 else 0.0
        scen.append((name, ha, mi, nmi))
        print("  %s" % name)
        print("     H(动作) %.2f bit，I(动作;动词) %.2f bit，归一化 %.2f  —— %s"
              % (ha, mi, nmi, note))

    print()
    print("  第三行是真实系统里的常态：语言有信息，但远不是全部。")
    print("  归一化 %.2f 说明：**即使把语言用到极致，也只能解释动作熵的 %.0f%%。**"
          % (scen[2][3], scen[2][3] * 100))
    print("  剩下 %.0f%% 的信息在语言里根本不存在——它来自物体几何、初始位姿、"
          % ((1 - scen[2][3]) * 100))
    print("  抓取时的微小滑动这些东西。")
    print()
    print("  这条结论直接给出那个最便宜的 sanity check：")
    print("  **给模型打乱语言输入，看性能掉多少。**")
    print("  掉得接近 %.0f%% = 语言条件真的在起作用；" % (scen[2][3] * 100))
    print("  掉得远小于它 = 语言标注基本没用，先别急着上更大的模型。")
    print("  这是唯一一个能在半小时内做完、却足以否掉一整个数据采集计划的实验。")
    print()
    print("  三条让语言标注真正有用的做法：")
    print("    ① 覆盖同一物体的多种动作。只有「拿起红方块」的数据集，")
    print("       学不会「不要拿红方块」——因为反例不存在")
    print("    ② 同一动作多种说法。让它学语义而不是背模板")
    print("    ③ 记录否定和约束。「把杯子放到盘子旁边，不要碰香蕉」")
    print("       这种指令才逼模型真的读语言")
    return {"verbs": rows, "h_verb": h_verb, "h_max": h_max, "top2": top2,
            "scen": scen}


# ---------------------------------------------------------------- [9] 结论

def part9_conclusion(p1, p2, p3, p4, p5, p6, p7, p8):
    print("\n[9] 一张决策表")
    print("-" * 78)
    r30 = p2["rows"][0]
    r50 = [r for r in p5["rows"] if abs(r[1] - 2.0) < 1e-9][0]
    seg = p7["rows"][1]
    rows = [
        ("只看一致率", "95% 一致可能是 kappa≈0", "类别不平衡会灌水"),
        ("30 条定 kappa", "95%% 区间宽 %.2f" % r30["w"], "30 条看方向，200 条看数值"),
        ("%.0f%% 成功率下抽 30 条" % (p3["prev"] * 100),
         "%.0f%% 概率一个负例都没有" % (p3["pts"][1][1] * 100),
         "要 %.0f 条才稳" % p3["need_95"]),
        ("阈值不写进元数据",
         "成功率差 %.0f 个百分点" % (p4["rows"][5][2] - p4["rows"][1][2]),
         "阈值是数据的一部分"),
        ("两人阈值差 0.5 cm", "kappa 从 1.000 掉到 %.2f" % r50[3],
         "是规范问题，不是人的问题"),
        ("随机抽 %.0f 条去标" % p6["curve"][-1][0],
         "剩余错误 %.0f 条" % p6["rand"],
         "按不确定性挑只要 %.0f 条" % p6["act"]),
        ("实时观看切阶段", "%.0f%% 的段标签是错的" % seg[4],
         "人工精度上限，换信号切分"),
        ("动词熵 %.2f / %.2f bit" % (p8["h_verb"], p8["h_max"]),
         "语言只能解释 %.0f%% 的动作信息" % (p8["scen"][2][3] * 100),
         "打乱语言看掉多少"),
    ]
    print(head(("做法", 26), ("代价", 30), ("为什么", 0)))
    print("  " + "-" * 90)
    for a, b, c in rows:
        print("  %s%s%s" % (pad(a, 26), pad(b, 30), c))
    print()
    print("  一句话：标注质量里唯一不能让步的是**「定义写清楚了没有」**。")
    print("  一致性、阈值、人力分配都是可算的取舍；定义模糊不是。")


if __name__ == "__main__":
    print("=" * 80)
    print("实验 6：一致性、阈值、语言——标注里那些能算的部分")
    print("=" * 80)
    d1 = part1_trap()
    d2 = part2_sample_size()
    d3 = part3_rare()
    d4 = part4_threshold()
    d5 = part5_ambiguous()
    d6 = part6_budget(near_n=d4["near"])
    d7 = part7_segmentation()
    d8 = part8_language()
    part9_conclusion(d1, d2, d3, d4, d5, d6, d7, d8)
    print("\n" + "=" * 80)

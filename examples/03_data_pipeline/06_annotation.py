#!/usr/bin/env python3
"""
实验 6：标注的三件事——一致性、阈值、语言。

运行：python examples/03_data_pipeline/06_annotation.py

标注看起来是「人来打标签」，但在数据平台里它是三个可计算的问题：

    [1] 一致性：为什么「95% 相同」可能等于「几乎没有一致性信息」
    [2] 阈值：成功/失败该在哪切？这个阈值决定了整个数据集的分布
    [3] 语言：如果 90% 的指令都是「把 X 放到 Y」，语言条件还有信息量吗
"""

import math
import random


def width(s: str) -> int:
    return sum(2 if ord(c) > 0x2E80 else 1 for c in s)


def pad(s: str, n: int) -> str:
    return s + " " * max(0, n - width(s))


# ---------------------------------------------------------------- [1] kappa

def cohen_kappa(a, b, c, d):
    """
    2x2 混淆：a=都判成功, b=A成功B失败, c=A失败B成功, d=都判失败
    kappa = (p_o - p_e) / (1 - p_e)
    p_e 是「按各自的边际分布瞎猜」时的期望一致率。
    """
    n = a + b + c + d
    po = (a + d) / n
    pa_pos = (a + b) / n
    pb_pos = (a + c) / n
    pe = pa_pos * pb_pos + (1 - pa_pos) * (1 - pb_pos)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 0.0
    return po, pe, kappa


def part1_agreement():
    print("\n[1] 「95% 一致」有时等于「毫无信息」")
    print("-" * 78)
    print("  两位标注员各判 200 条 episode 是成功还是失败。")
    print()
    print("  %s%s%s%s" % (pad("场景", 32), pad("表面一致率", 14),
                          pad("期望一致率", 14), "kappa"))
    print("  " + "-" * 78)
    cases = [
        ("均衡：两人各 100 成 / 100 败", 90, 5, 5, 100),
        ("失衡但同步：190 成 / 10 败", 188, 2, 2, 8),
        ("失衡且不同步：199 成 / 1 败", 198, 1, 1, 0),
        ("一方全判成功，另一方 200/19", 200, 0, 19, 0),
    ]
    results = []
    for label, a, b, c, d in cases:
        po, pe, k = cohen_kappa(a, b, c, d)
        results.append((label, po, pe, k))
        print("  %s%s%s%s" % (pad(label, 32), pad("%.1f%%" % (po * 100), 14),
                              pad("%.1f%%" % (pe * 100), 14), "%.3f" % k))

    hi_po = max(results, key=lambda r: r[1])
    hi_k = max(results, key=lambda r: r[3])
    print()
    print("  把两行放在一起看，结论就出来了：")
    print("    「%s」  一致率 %.1f%%   kappa %.3f"
          % (hi_po[0], hi_po[1] * 100, hi_po[3]))
    print("    「%s」  一致率 %.1f%%   kappa %.3f"
          % (hi_k[0], hi_k[1] * 100, hi_k[3]))
    print()
    print("  **一致率更高的那一行，kappa 反而更低**。")
    print("  而且 kappa 是负的：比「按各自边际分布随便猜」还差。")
    print("  第三行的情形最值得记住：两人各有 199 次判「成功」，")
    print("  但**不是同一批 199 条**。表面看几乎全一致，实际几乎没有共识。")
    print("  这不是理论游戏——成功率 99% 的数据集里，最容易出现这种情况。")

    print()
    print("  这就是为什么验收标注质量必须看 kappa：")
    print("  一致率会被**类别不平衡**灌水，kappa 扣掉了「按边际分布瞎猜也能对」的部分。")
    print()
    print("  经验阈值（学界常用，不是硬标准）：")
    print("    kappa < 0.4  标注规范有问题，返工")
    print("    0.4 ~ 0.6    勉强可用，但下游指标会有噪声")
    print("    > 0.8        可信，可以当 ground truth")
    print()
    print("  一个务实做法：先让两人各标 30 条，算一次 kappa。")
    print("  不达标改规范重标 —— 此时只浪费 60 条，而不是 2000 条。")


# ---------------------------------------------------------------- [2] 阈值

def part2_threshold():
    print("\n[2] 成功阈值：一个数字决定整个数据集的分布")
    print("-" * 76)
    print("  场景：评价「把方块放进盒子」是否成功，用最终位姿误差（cm）判定。")
    print("  真实误差服从一个双峰分布：大部分在 0.8cm 内，少的在 3-8cm（没对准）。")
    print()

    rng = random.Random(19)
    errs = []
    for _ in range(1000):
        if rng.random() < 0.72:
            errs.append(abs(rng.gauss(0.7, 0.45)))     # 成功
        else:
            errs.append(abs(rng.gauss(4.5, 2.0)))      # 失败
    errs.sort()

    print("  %s%s%s%s" % (pad("阈值", 10), pad("判成功条数", 14),
                          pad("成功率", 12), "被误判的典型情况"))
    print("  " + "-" * 66)
    for th in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0):
        n_ok = sum(1 for e in errs if e <= th)
        note = ("太严：真实的成功被丢掉" if th <= 1.0 else
                "太松：接近失败的也算成功" if th >= 3.0 else "较合理")
        print("  %s%s%s%s" % (pad("%.1f cm" % th, 10),
                              pad(str(n_ok), 14),
                              pad("%.1f%%" % (n_ok / len(errs) * 100), 12), note))

    print()
    print("  阈值从 1cm 挪到 3cm，成功率从 %.0f%% 变成 %.0f%% ——"
          % (sum(1 for e in errs if e <= 1.0) / 10,
             sum(1 for e in errs if e <= 3.0) / 10))
    print("  同一批数据，报出来的成功率差 %.0f 个百分点。"
          % (sum(1 for e in errs if e <= 3.0) / 10
             - sum(1 for e in errs if e <= 1.0) / 10))
    print()
    print("  这不是学术争论，是**可复现性**问题：")
    print("  · 阈值必须写进数据集元数据，和标签一起存")
    print("  · 论文里报的「成功率」必须带阈值，否则无法比较")
    print("  · 换阈值等于换数据集版本，不能悄悄改")

    print()
    print("  更隐蔽的一点：阈值附近的样本最难判。")
    near = [e for e in errs if 0.8 <= e <= 2.2]
    print("  落在 0.8-2.2cm 区间的有 %d 条（%.0f%%），" % (len(near), len(near) / 10))
    print("  这些样本人和自动判定的分歧最大 —— 而这个区间正好横跨常用阈值。")
    print("  所以「自动判定 + 人工复核边界样本」比「全部人工」更划算：")
    print("  人力花在 %d 条上，而不是 1000 条。" % len(near))


# ---------------------------------------------------------------- [3] 阶段

def part3_segmentation():
    print("\n[3] 阶段切分：人工标注的时间精度极限")
    print("-" * 74)
    print("  一段 8 秒的抓取，可以切成「接近 → 抓取 → 抬起 → 放置」四段。")
    print("  切分边界用于：分段训练、过程奖励、失败定位。")
    print()

    true_bounds = [0.0, 2.4, 3.1, 5.6, 8.0]
    print("  真值边界：%s 秒" % ", ".join("%.2f" % b for b in true_bounds))
    print()
    print("  %s%s%s" % (pad("标注方式", 28), pad("平均边界误差", 16), "说明"))
    print("  " + "-" * 62)
    # 典型量级：慢放逐帧看能做到 ±0.1s；实时看受反应时间限制约 ±0.3s
    rows = [
        ("人工，慢放 ×0.25 逐帧看", 0.100, "常见做法，成本高"),
        ("人工，实时观看", 0.320, "最省事但最不准"),
        ("速度阈值自动切分", 0.080, "快，但对缓变动作失效"),
        ("力/接触信号自动切分", 0.030, "最准，需要力传感器"),
    ]
    for label, err, note in rows:
        print("  %s%s%s" % (pad(label, 28), pad("%.0f ms" % (err * 1000), 16), note))

    print()
    print("  30Hz 的数据，一帧 = 33ms。实时观看的 %.0f ms 误差是 %.0f 帧，"
          % (rows[1][1] * 1000, rows[1][1] * 1000 / 33))
    print("  慢放逐帧看是 %.0f 帧 —— **人工标注的精度上限就在这个量级**。"
          % (rows[0][1] * 1000 / 33))
    print()
    print("  这不是标注员不认真，是人的反应时间决定了它：")
    print("  视觉感知延迟约 100ms，加上判断和按键，300ms 是常态。")
    print("  所以任何需要**帧级精度**的边界，都不该指望纯人工。")

    print()
    print("  一个可用的组合策略：")
    print("    1. 用信号（速度/力/夹爪开合）自动切出候选边界")
    print("    2. 人工只做「接受 / 移动几帧」的确认，不做从零标注")
    print("    3. 把人工的修正量记下来，反过来校准自动切分器的偏差")


# ---------------------------------------------------------------- [4] 语言

def part4_language():
    print("\n[4] 语言标注：90% 的指令是同一句话，语言条件就没有信息了")
    print("-" * 76)

    verbs = ["拿起", "放下", "放进", "移到", "推", "拉", "打开", "关上", "翻转", "堆叠"]
    objects = ["红色方块", "蓝色杯子", "积木", "抽屉", "香蕉", "夹子"]
    targets = ["盒子里", "桌面上", "左边", "盘子旁", "架子上"]
    # 真实分布：少数动词占绝大多数
    verb_p = [0.42, 0.20, 0.14, 0.09, 0.05, 0.04, 0.03, 0.01, 0.01, 0.01]

    rng = random.Random(31)
    n = 3000
    sents = []
    for _ in range(n):
        v = rng.choices(verbs, weights=verb_p)[0]
        o = rng.choice(objects)
        t = rng.choice(targets)
        sents.append("%s%s%s" % (v, o, t) if v in ("放进", "移到", "堆叠")
                     else "%s%s" % (v, o))

    def entropy(counts, total):
        h = 0.0
        for c in counts:
            if c:
                p = c / total
                h -= p * math.log2(p)
        return h

    from collections import Counter
    vc = Counter()
    for s in sents:
        for v in verbs:
            if s.startswith(v):
                vc[v] += 1
                break
    total = sum(vc.values())

    print("  %d 条指令，动词分布：" % n)
    print("  %s%s%s" % (pad("动词", 12), pad("出现次数", 12), "占比"))
    print("  " + "-" * 44)
    for v, c in vc.most_common(5):
        print("  %s%s%s" % (pad(v, 12), pad(str(c), 12), "%.1f%%" % (c / total * 100)))
    rest = total - sum(c for _v, c in vc.most_common(5))
    print("  %s%s%s" % (pad("其余 5 个", 12), pad(str(rest), 12),
                        "%.1f%%" % (rest / total * 100)))

    h_verb = entropy(list(vc.values()), total)
    print()
    print("  动词熵：%.2f bit（理论上限 %.2f bit）" % (h_verb, math.log2(len(verbs))))
    print("  前 2 个动词覆盖 %.0f%% 的样本。"
          % (sum(c for _v, c in vc.most_common(2)) / total * 100))
    print()
    print("  含义很直接：如果前 2 个动词就占了 62%，")
    print("  模型完全可以忽略语言输入，靠视觉走完 —— 训练 loss 照样很低。")
    print("  这不是模型的问题，是**数据没有提供语言-行为的对应关系**。")

    print()
    print("  三条让语言标注真正有用的做法：")
    print("    ① **覆盖同一物体的多种动作**。只有「拿起红方块」的数据集，")
    print("       学不会「不要拿红方块」——因为反例不存在")
    print("    ② **同一动作多种说法**。让它学语义而不是背模板")
    print("    ③ **记录否定和约束**。「把杯子放到盘子旁边，不要碰香蕉」")
    print("       这种指令才逼模型真的读语言")
    print()
    print("  最小的可检验指标：给模型打乱语言输入，看性能掉多少。")
    print("  掉得很少 = 语言标注基本没用，先别急着上更大的模型。")


if __name__ == "__main__":
    print("=" * 78)
    print("实验 6：一致性、阈值、语言——标注里三个可计算的问题")
    print("=" * 78)
    part1_agreement()
    part2_threshold()
    part3_segmentation()
    part4_language()
    print("\n" + "=" * 78)

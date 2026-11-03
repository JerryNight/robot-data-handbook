#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验 1：行为克隆的两种失败，以及 DAgger 只修得好其中一种

行为克隆（BC）就是把「一堆 episode」压成「一个函数」，然后祈祷这个函数
在它没见过的状态上也是对的。它有两条独立的死路：

  死路 A · 分布偏移（distribution shift）
     策略自己的小误差把它带到演示数据没覆盖的状态，在那里它的输出纯属
     外推。DAgger 就是为这条路准备的：让策略自己跑，跑到哪让专家标到哪。

  死路 B · 多模态动作（multimodal action）
     同一个状态下，专家做过向左也做过向右。用 MSE 回归去拟合，得到的是
     「平均动作」—— 一个专家从来没做过的动作，通常正好撞在障碍上。
     DAgger 修不了这个，加数据也没用，只能换输出表示。

本实验用两个独立的小世界分别演示：

  世界 1（2D 绕障）
      策略 = 状态网格查找表（非参数 BC），演示只从一个固定起点采。
      量：训练起点偏移 → 成功率曲线（能力边界）、出界访问比例、
          DAgger 迭代后边界外移了多少。

  世界 2（1D 双峰）
      同一状态两种正确动作。量：MSE 回归 / 分箱分类 / 采样 三种输出
      表示各自落在禁区的概率，以及「每步重采样」造成的抖动。

最后量分块执行（action chunking）的代价与收益。

运行：python examples/05_learning_from_data/01_bc_and_dagger.py
依赖：无（只用标准库）
"""

import math
import random


# ---------------------------------------------------------------- 输出对齐

def width(s):
    return sum(2 if ord(c) > 0x2E80 else 1 for c in s)


def pad(s, n):
    return s + " " * max(0, n - width(s))


def bar(v, n=24):
    k = int(round(v * n))
    return "#" * k + "." * (n - k)


# ---------------------------------------------------------------- 世界 1：2D 绕障

START = (0.0, 0.0)
GOAL = (1.0, 1.0)
OBST = (0.5, 0.5)
R_OBS = 0.18
R_BOT = 0.02
DT = 0.1
SPEED = 0.7           # 专家等速：动作 = 单位方向 × SPEED
MAX_STEPS = 70
SUBSTEP = 0.02
GOAL_TOL = 0.12
CELL = 0.06           # 策略表网格边长
DEMO_NOISE = 0.06     # 演示噪声（人手的抖动）
PROC_NOISE = 0.02     # 执行噪声（跟踪误差），真机永远有


def expert_action(p):
    """势场专家：目标吸引 + 障碍排斥（带切向分量打破对称）。"""
    px, py = p
    fx = (GOAL[0] - px) * 1.2
    fy = (GOAL[1] - py) * 1.2
    ox, oy = px - OBST[0], py - OBST[1]
    d = math.hypot(ox, oy) + 1e-9
    if d < 0.70:
        mag = 0.05 / (d * d)
        fx += mag * ox / d - 0.40 * mag * oy / d
        fy += mag * oy / d + 0.40 * mag * ox / d
    n = math.hypot(fx, fy)
    if n < 1e-9:
        return (0.0, 0.0)
    return (fx / n * SPEED, fy / n * SPEED)


def collide(p):
    return math.hypot(p[0] - OBST[0], p[1] - OBST[1]) < R_OBS + R_BOT


def at_goal(p):
    return math.hypot(p[0] - GOAL[0], p[1] - GOAL[1]) < GOAL_TOL


def rollout(policy, start=START, chunk=1, proc_noise=PROC_NOISE, rng=None,
            max_steps=MAX_STEPS, impulse=None):
    """跑一条轨迹，返回 (结局, 步数, 出界查询数, 决策次数)。

    chunk>1 时：每 chunk 步重新决策一次。决策时先用自己的策略在**无噪声**
    的预测轨迹上前推 chunk 步，得到动作序列，再把这串动作在真机上盲执行。
    这正是 action chunking 的定义 —— 不是「保持同一个动作」。

    impulse=(step, dx, dy)：在第 step 步之后给一个位置突变（被撞/打滑）。
    """
    p = (start[0], start[1])
    outside = 0
    replans = 0
    if collide(p):
        return ("collide", 0, 0, 0)
    queue = []
    for t in range(max_steps):
        if impulse is not None and t == impulse[0]:
            p = (p[0] + impulse[1], p[1] + impulse[2])
            # 注意：这里刻意**不**清空动作队列 —— 这个扰动是没被察觉的
            # （打滑、物体被碰走），策略仍然拿着旧的动作序列在盲执行。
            if collide(p):
                return ("collide", t, outside, replans)
        if not queue:
            ph = p
            replans += 1
            for _ in range(chunk):
                a, was_out = policy(ph)
                if was_out:
                    outside += 1
                queue.append(a)
                ph = (ph[0] + a[0] * DT, ph[1] + a[1] * DT)
        a = queue.pop(0)
        if rng is not None and proc_noise > 0:
            a = (a[0] + rng.gauss(0, proc_noise), a[1] + rng.gauss(0, proc_noise))
        dx, dy = a[0] * DT, a[1] * DT
        n_sub = max(1, int(math.ceil(math.hypot(dx, dy) / SUBSTEP)))
        for _ in range(n_sub):
            p = (p[0] + dx / n_sub, p[1] + dy / n_sub)
            if collide(p):
                return ("collide", t + 1, outside, replans)
            if at_goal(p):
                return ("success", t + 1, outside, replans)
        if not (-0.6 < p[0] < 2.1 and -0.6 < p[1] < 2.1):
            return ("lost", t + 1, outside, replans)
    return ("timeout", max_steps, outside, replans)


def ckey(p):
    return (int(math.floor(p[0] / CELL)), int(math.floor(p[1] / CELL)))


def make_demos(n_ep, rng, start_noise=0.0, noise=DEMO_NOISE):
    """专家演示。默认所有 episode 都从同一个起点开始 —— 这是真实采集的常态。

    注意：噪声同时进入「记录的动作」和「实际的轨迹」。也就是说每条演示
    的路径都略有不同，和真人复现同一动作时一样。
    """
    data = []
    for _ in range(n_ep):
        p = (START[0] + rng.gauss(0, start_noise),
             START[1] + rng.gauss(0, start_noise))
        for _t in range(MAX_STEPS):
            a0 = expert_action(p)
            a = (a0[0] + rng.gauss(0, noise), a0[1] + rng.gauss(0, noise))
            data.append((p, a))
            p = (p[0] + a[0] * DT, p[1] + a[1] * DT)
            if collide(p) or at_goal(p):
                break
    return data


def build_table(data):
    acc = {}
    for p, a in data:
        k = ckey(p)
        e = acc.get(k)
        if e is None:
            acc[k] = [a[0], a[1], 1]
        else:
            e[0] += a[0]
            e[1] += a[1]
            e[2] += 1
    return {k: (v[0] / v[2], v[1] / v[2]) for k, v in acc.items()}


def make_table_policy(tab, max_ring=6):
    """查找表 → policy。空格向外扩环找最近的非空格，这就是外推。"""
    def policy(p):
        ci, cj = ckey(p)
        hit = tab.get((ci, cj))
        if hit is not None:
            return (hit, False)
        best, bd = None, 1e18
        for r in range(1, max_ring + 1):
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    if max(abs(di), abs(dj)) != r:
                        continue
                    v = tab.get((ci + di, cj + dj))
                    if v is not None:
                        cx, cy = (ci + di + 0.5) * CELL, (cj + dj + 0.5) * CELL
                        dd = (cx - p[0]) ** 2 + (cy - p[1]) ** 2
                        if dd < bd:
                            bd, best = dd, v
            if best is not None:
                return (best, True)
        return ((0.0, 0.0), True)
    return policy


def eval_at_offset(policy, offset, n, rng, chunk=1):
    """起点在半径 offset 的圆上随机取，跑 n 条。"""
    ok = 0
    out_q = 0
    tot_q = 0
    for _ in range(n):
        th = rng.uniform(0, 2 * math.pi)
        s = (START[0] + offset * math.cos(th), START[1] + offset * math.sin(th))
        res, _steps, o, q = rollout(policy, start=s, chunk=chunk, rng=rng)
        if res == "success":
            ok += 1
        out_q += o
        tot_q += q
    return ok / n, out_q / max(1, tot_q)


def curve(policy, n, rng, offsets=(0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)):
    return [(r, eval_at_offset(policy, r, n, rng)) for r in offsets]


def print_curve(title, cur):
    print(title)
    print(pad("起点偏移 m", 12) + pad("成功率", 10) + pad("出界查询", 10) + "分布")
    for r, (rate, oq) in cur:
        print(pad(f"{r:.2f}", 12) + pad(f"{rate*100:5.1f}%", 10)
              + pad(f"{oq*100:5.1f}%", 10) + bar(rate))


def dump_map(tab, title):
    print(title)
    x0, x1, y0, y1 = -0.10, 1.10, -0.10, 1.10
    nx, ny = 30, 15
    for j in range(ny, -1, -1):
        y = y0 + (y1 - y0) * j / ny
        row = ""
        for i in range(nx):
            x = x0 + (x1 - x0) * i / nx
            if math.hypot(x - OBST[0], y - OBST[1]) < R_OBS:
                row += "@"
            elif ckey((x, y)) in tab:
                row += "#"
            else:
                row += "."
        print("  " + row)
    print("  # 有样本（策略有依据）   . 空格（只能外推）   @ 障碍")


# ---------------------------------------------------------------- 主流程

def main():
    rng = random.Random(20261103)

    print("=" * 76)
    print("世界 1：2D 绕障 —— 分布偏移能不能被 DAgger 修好")
    print("=" * 76)
    print(f"起点 {START}  目标 {GOAL}  障碍 {OBST} r={R_OBS}")
    print(f"专家：等速势场 {SPEED} m/s，控制周期 {DT}s，演示噪声 σ={DEMO_NOISE}，")
    print(f"      执行噪声 σ={PROC_NOISE}（真机的跟踪误差，仿真里也保留）")
    print(f"策略：{CELL} m 网格查找表（非参数 BC），空格向外扩环外推")
    print("采集：所有演示 episode 都从同一个起点开始（真实采集的常态）\n")

    exp_pol = lambda p: (expert_action(p), False)
    rate0, oq0 = eval_at_offset(exp_pol, 0.0, 150, rng)
    print(f"专家自身：起点无偏移成功率 {rate0*100:.1f}%，"
          f"起点偏移 0.30 m 时 {eval_at_offset(exp_pol, 0.30, 150, rng)[0]*100:.1f}%")
    print("（专家是闭式控制器，偏移不影响它 —— 这是后面所有对比的基准）\n")

    print("-" * 76)
    print("BC：只用演示数据训练，然后在不同起点偏移下测")
    print("-" * 76)
    tab_bc = None
    for n_ep in (10, 40, 120):
        data = make_demos(n_ep, rng)
        tab = build_table(data)
        pol = make_table_policy(tab)
        cur = curve(pol, 200, rng)
        print(f"\n[{n_ep:3d} 条演示 / {len(data):5d} 样本 / {len(tab):3d} 个非空格]")
        print_curve("", cur)
        if n_ep == 120:
            tab_bc = tab
            cur_bc = cur

    # ------------------------------------------------------------ DAgger
    print("\n" + "-" * 76)
    print("DAgger：让策略在自己会去的地方跑，让专家给这些状态标动作")
    print("-" * 76)
    rng_d = random.Random(4242)
    data = make_demos(120, rng_d)
    tab = build_table(data)
    cur0 = curve(make_table_policy(tab), 200, rng_d)
    print_curve("第 0 轮（纯 BC）", cur0)

    for it in range(1, 6):
        new = []
        for _ in range(100):
            # 采集分布 = 部署分布：起点带 ±0.30 m 随机偏移
            th = rng_d.uniform(0, 2 * math.pi)
            r = 0.30 * math.sqrt(rng_d.random())
            s = (START[0] + r * math.cos(th), START[1] + r * math.sin(th))
            p = s
            pol = make_table_policy(tab)
            queue = []
            for t in range(MAX_STEPS):
                if not queue:
                    ph = p
                    for _ in range(1):
                        a, _ = pol(ph)
                        queue.append(a)
                        ph = (ph[0] + a[0] * DT, ph[1] + a[1] * DT)
                a = queue.pop(0)
                # 探一点噪声，否则策略永远走同一条线，采不到别处
                a = (a[0] + rng_d.gauss(0, 0.20), a[1] + rng_d.gauss(0, 0.20))
                new.append((p, expert_action(p)))   # 专家给这个状态标动作
                p = (p[0] + a[0] * DT, p[1] + a[1] * DT)
                if collide(p) or at_goal(p):
                    break
        data = data + new          # 聚合，不丢历史数据
        tab = build_table(data)
        cur = curve(make_table_policy(tab), 200, rng_d)
        print_curve(f"第 {it} 轮（+{len(new)} 个纠正样本，累计 {len(data)}）", cur)
    tab_dagger = tab

    # 汇总对比
    print("\n" + pad("起点偏移", 10) + pad("BC(120条)", 14) + pad("DAgger(5轮)", 14)
          + "提升")
    for (r, (b, _)), (_, (d, _)) in zip(cur_bc, cur):
        print(pad(f"{r:.2f} m", 10) + pad(f"{b*100:5.1f}%", 14)
              + pad(f"{d*100:5.1f}%", 14) + f"{(d-b)*100:+5.1f} pt")

    dump_map(tab_bc, "\n覆盖图 · BC 的表（演示只覆盖一条窄走廊）")
    dump_map(tab_dagger, "\n覆盖图 · DAgger 之后的表（纠正数据把走廊撑宽了）")

    # ------------------------------------------------------------ 世界 2
    print("\n" + "=" * 76)
    print("世界 2：1D 双峰动作 —— MSE 回归必然掉进两峰之间")
    print("=" * 76)

    rng2 = random.Random(7)
    n = 4000
    p_left = 0.55
    acts = []
    for _ in range(n):
        base = -1.0 if rng2.random() < p_left else +1.0
        acts.append(base + rng2.gauss(0, 0.10))
    mean_a = sum(acts) / n
    var_a = sum((a - mean_a) ** 2 for a in acts) / n
    in_forbidden_expert = sum(1 for a in acts if abs(a) < 0.3) / n

    print(f"专家动作分布：{p_left*100:.0f}% 向左(-1.0) + {(1-p_left)*100:.0f}% 向右(+1.0)，σ=0.10")
    print(f"（对应「绕过障碍的左侧 / 右侧」，|a|<0.3 就是从障碍正中穿过去）")
    print(f"专家动作均值 {mean_a:+.3f}　方差 {var_a:.3f}　"
          f"专家自己落禁区比例 {in_forbidden_expert*100:.2f}%\n")

    nb = 40
    lo, hi = -1.6, 1.6
    bins = [0] * nb
    for a in acts:
        k = int((a - lo) / (hi - lo) * nb)
        if 0 <= k < nb:
            bins[k] += 1
    best_bin = max(range(nb), key=lambda k: bins[k])
    pred_cls = lo + (best_bin + 0.5) * (hi - lo) / nb
    sample_bad = sum(1 for _ in range(4000)
                     if abs((-1.0 if rng2.random() < p_left else 1.0)
                            + rng2.gauss(0, 0.10)) < 0.3) / 4000

    print(pad("输出表示", 26) + pad("单次预测", 12) + pad("落禁区概率", 14) + "代价")
    print(pad("-" * 26, 26) + pad("-" * 12, 12) + pad("-" * 14, 14) + "-" * 22)
    rows = [
        ("MSE 回归（条件均值）", f"{mean_a:+.3f}", 1.0, "每次都撞"),
        ("分箱分类（argmax）", f"{pred_cls:+.3f}", 0.0, "永远只走一侧"),
        ("采样（mix / diffusion）", "-", sample_bad, "每步可能换边"),
    ]
    for name, v, pb, cost in rows:
        print(pad(name, 26) + pad(v, 12) + pad(f"{pb*100:5.1f}%", 14) + cost)

    # 每步重采样的抖动
    def flip_rate(k, trials=4000):
        """每 k 步重新采样一次，统计相邻步之间「换边」的比例（每步口径）。"""
        flips = 0
        prev = None
        for t in range(trials):
            if t % k == 0:
                side = -1.0 if rng2.random() < p_left else 1.0
                if prev is not None and side != prev:
                    flips += 1
                prev = side
        return flips / float(trials)

    print(f"\n每步独立重采样：{flip_rate(1)*100:.1f}% 的相邻决策会换边 —— "
          f"在真机上就是来回犹豫。")
    for k in (4, 8, 16):
        print(f"  每 {k:2d} 步才重采一次：{flip_rate(k)*100:4.1f}%")
    print("所以 chunking 的第一价值不是预测更准，是让多峰的采样结果保持")
    print("一段时间 —— 动作连贯性来自「采样频率」而不是「模型精度」。")

    # ------------------------------------------------------------ 分块执行
    print("\n" + "=" * 76)
    print("分块执行（action chunking）：把「反应延迟」单独量出来")
    print("=" * 76)
    print("为了不让「覆盖不足」混淆结果，这里用完美的专家策略，只改 chunk。")
    print("在第 3 步（离障碍还远）给一次横向位移，幅度从 0.04 到 0.16 m，")
    print("看多大的扰动、在多长的 chunk 下会撞上去。")
    print("这个扰动是「没被察觉」的：策略不会因此重新决策，只是原来的动作序列")
    print("从一个偏移了的位置开始执行 —— 打滑、物体被碰走都是这一类。\n")

    # 名义路径（无障碍、无噪声）
    nom = []
    p = START
    for _t in range(MAX_STEPS):
        nom.append(p)
        a = expert_action(p)
        p = (p[0] + a[0] * DT, p[1] + a[1] * DT)
        if at_goal(p):
            nom.append(p)
            break

    exp_pol = lambda q: (expert_action(q), False)
    # 偏移方向：垂直于第 3→5 步的运动方向，朝障碍那一侧
    ax_, ay_ = nom[3]
    bx_, by_ = nom[5]
    ux, uy = bx_ - ax_, by_ - ay_
    un = math.hypot(ux, uy)
    ux, uy = ux / un, uy / un
    nx, ny = -uy, ux                      # 法向
    if (nx * (OBST[0] - ax_) + ny * (OBST[1] - ay_)) < 0:
        nx, ny = -nx, -ny                 # 指向障碍那一侧

    # 名义路径上第 3 步到障碍表面的余量，决定了「盲走多远会撞」
    d3 = math.hypot(ax_ - OBST[0], ay_ - OBST[1]) - R_OBS - R_BOT
    print(f"名义路径在第 3 步时距障碍表面 {d3*100:.1f} cm，"
          f"每步前进 {SPEED*DT*100:.0f} cm")
    print(f"→ 粗算：盲走超过 {d3/(SPEED*DT):.1f} 步就接近障碍，"
          f"所以 chunk 的安全上限就在这个量级（实测 6~8 之间）。\n")

    rng3 = random.Random(11)
    mags = (0.04, 0.07, 0.10, 0.13, 0.16)
    head = pad("chunk", 8) + pad("决策次数", 10)
    for m in mags:
        head += pad("%.2f m" % m, 10)
    print(head + "滞后")
    for k in (1, 2, 4, 6, 8, 12, 16):
        line = pad(str(k), 8) + pad("%d" % max(1, int(round(len(nom) / float(k)))), 10)
        for m in mags:
            col = 0
            for _ in range(80):
                res, _st, _o, _rp = rollout(exp_pol, chunk=k, rng=rng3,
                                            impulse=(3, nx * m, ny * m))
                if res != "success":
                    col += 1
            line += pad("%3.0f%%" % (col / 80.0 * 100), 10)
        line += ("%d 步" % k) if k > 1 else "无"
        print(line)

    print("\n同一张表读三件事：")
    print("  · 往左：扰动越小越容易扛住 —— chunk 的安全边界取决于扰动幅度。")
    print("  · 往下：chunk 越大越扛不住，因为纠偏要等到下一个决策点。")
    print("  · 决策次数 = 步数/chunk，算力省得是线性的，延迟涨得也是线性的。")
    print("这就是 ACT 那类方法必须配 temporal ensembling 的原因：")
    print("单靠加大 chunk 换平滑，会把反应能力一起卖掉。")

    print("\n" + "=" * 76)
    print("小结")
    print("=" * 76)
    print("1. 分布偏移：BC 的成功率随演示条数上升，但在「起点偏移」这条轴上")
    print("   存在一条明确的悬崖。悬崖的位置由采集覆盖决定，不由模型容量决定。")
    print("2. DAgger 能做的：把策略自己访问到的状态补进训练集，把悬崖往外推。")
    print("3. 多模态：MSE 回归必然给出两峰之间的均值，落禁区概率 100%。这不")
    print("   是数据不够，是目标函数错了。加数据、加迭代都没用，只能换表示。")
    print("4. 对数据平台的含义：episode 条数不是能力指标，覆盖范围才是。而覆盖")
    print("   范围可以在线测 —— 拿部署时的状态去查采集覆盖，算一个出界比例。")


if __name__ == "__main__":
    main()

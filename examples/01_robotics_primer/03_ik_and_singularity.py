#!/usr/bin/env python3
"""
实验 3：数值逆运动学与冗余
============================

真实的 6/7 自由度机械臂没有解析 IK，全靠数值迭代。这个脚本用 3 连杆平面臂
（控制 2 维末端位置，所以是冗余的）演示四件在数据里看得见的事：

  1. 数值 IK 怎么迭代收敛（阻尼最小二乘）
  2. 目标靠近奇异位形时，收敛变慢甚至卡死
  3. 冗余臂有无穷多组解 -> 同一个末端轨迹，关节轨迹可以完全不同
  4. 关节限位会让 IK 直接卡在墙上
  5. 雅可比的实际用途：速度映射、力映射、零空间
  6. 冗余的自运动：末端钉死，肘部自己沿一条弧滑

第 3 点最重要：它是"为什么数据集里同类任务的关节轨迹差异巨大"的答案。
第 5、6 点是配套文档 kinematics.md 第 3、5 节要引用的数字。

运行：
    python examples/01_robotics_primer/03_ik_and_singularity.py

只依赖 Python 标准库。
"""

import math
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

L = [0.6, 0.5, 0.4]          # 三个连杆长度
LIMITS = [(-math.pi, math.pi)] * 3


# ---------------------------------------------------------------- 正运动学

def fk(q):
    """3 连杆平面臂的正运动学，返回末端 (x, y)。"""
    x = y = 0.0
    acc = 0.0
    for i, (l, qi) in enumerate(zip(L, q)):
        acc += qi
        x += l * math.cos(acc)
        y += l * math.sin(acc)
    return x, y


def jacobian(q):
    """2x3 雅可比矩阵。J[i][j] = d(末端第i维) / d(关节j)"""
    s = [0.0] * 3
    c = [0.0] * 3
    acc = 0.0
    for i, qi in enumerate(q):
        acc += qi
        s[i], c[i] = math.sin(acc), math.cos(acc)

    # 第 j 列只受 j 及之前的连杆影响
    jx = [0.0] * 3
    jy = [0.0] * 3
    for j in range(3):
        for i in range(j, 3):
            jx[j] -= L[i] * s[i]
            jy[j] += L[i] * c[i]
    return [jx, jy]


# ---------------------------------------------------------------- 数值 IK

def ik_dls(target, q0, max_iter=200, tol=1e-6, lam=0.05, limits=None,
           step_scale=0.6):
    """
    阻尼最小二乘（Damped Least Squares / Levenberg-Marquardt）求 IK。

        dq = J^T (J J^T + lam^2 I)^-1 * e

    比纯伪逆好在：接近奇异时 lam 项防止解爆炸。
    lam 越大越稳但收敛越慢，是一个必须调的取舍。

    limits 默认为 ±180°（真实关节都有行程限制，不限制会解出 453° 这种
    物理上不存在的角度）。

    返回 (q, 迭代历史, 是否收敛)
    """
    if limits is None:
        limits = [(-math.pi, math.pi)] * 3

    q = list(q0)
    hist = []

    for it in range(max_iter):
        p = fk(q)
        e = [target[0] - p[0], target[1] - p[1]]
        err = math.hypot(e[0], e[1])
        hist.append((it, err))
        if err < tol:
            return q, hist, True

        J = jacobian(q)

        # A = J J^T + lam^2 I  (2x2)
        a00 = sum(J[0][k] * J[0][k] for k in range(3)) + lam * lam
        a01 = sum(J[0][k] * J[1][k] for k in range(3))
        a11 = sum(J[1][k] * J[1][k] for k in range(3)) + lam * lam

        det = a00 * a11 - a01 * a01
        if abs(det) < 1e-15:
            return q, hist, False

        # 解 A * y = e
        y0 = (a11 * e[0] - a01 * e[1]) / det
        y1 = (-a01 * e[0] + a00 * e[1]) / det

        # dq = J^T * y
        dq = [J[0][k] * y0 + J[1][k] * y1 for k in range(3)]

        for k in range(3):
            q[k] += step_scale * dq[k]
            lo, hi = limits[k]
            q[k] = max(lo, min(hi, q[k]))

    return q, hist, False


# ---------------------------------------------------------------- 工具

def cond_number(q):
    """2x3 雅可比的条件数（用 J J^T 的特征值算，2x2 有闭式解）。"""
    J = jacobian(q)
    a = sum(J[0][k] ** 2 for k in range(3))
    b = sum(J[1][k] ** 2 for k in range(3))
    c = sum(J[0][k] * J[1][k] for k in range(3))
    tr, det = a + b, a * b - c * c
    disc = max(0.0, tr * tr / 4 - det)
    l1 = tr / 2 + math.sqrt(disc)
    l2 = tr / 2 - math.sqrt(disc)
    if l2 <= 1e-15:
        return float("inf")
    return math.sqrt(l1 / l2)


def pinv_apply(J, v):
    """J^+ v ：2x3 雅可比的伪逆右乘一个 2 维向量，得到 3 维关节速度。"""
    a = sum(J[0][k] ** 2 for k in range(3))
    b = sum(J[1][k] ** 2 for k in range(3))
    c = sum(J[0][k] * J[1][k] for k in range(3))
    det = a * b - c * c
    y0 = (b * v[0] - c * v[1]) / det
    y1 = (-c * v[0] + a * v[1]) / det
    return [J[0][k] * y0 + J[1][k] * y1 for k in range(3)]


def null_direction(q):
    """返回 J·n = 0 的单位方向 n（零空间方向）。

    做法：任取一个 z，做投影 (I - J^+J)z。它不是零向量，就说明存在一个
    关节变化方向让末端速度恰好为 0 —— 这正是"冗余"的数学定义。
    """
    J = jacobian(q)
    z = [1.0, 0.0, 0.0]
    jz = [sum(J[i][k] * z[k] for k in range(3)) for i in range(2)]
    p = pinv_apply(J, jz)
    n = [z[k] - p[k] for k in range(3)]
    ln = math.hypot(*n)
    return [v / ln for v in n]


def fmt_deg(q):
    return "[" + ", ".join(f"{math.degrees(v):>7.1f}°" for v in q) + "]"


def main():
    print("=" * 74)
    print("实验 3：数值 IK 与冗余")
    print(f"3 连杆平面臂，L = {L}，控制 2 维末端位置 -> 冗余 1 个自由度")
    print("=" * 74)

    # -------- 1. 正常收敛
    print("\n[1] 正常情况：目标可达，迭代收敛")
    target = (1.0, 0.6)
    q0 = [0.2, 0.4, -0.2]        # 从非奇异构型出发
    q_sol, hist, ok = ik_dls(target, q0)
    print(f"  目标：{target}   初始 q：{fmt_deg(q0)}")
    print(f"  {'迭代':>5} {'末端误差(m)':>14}")
    for it, err in hist[:6]:
        print(f"  {it:>5} {err:>14.6f}")
    if len(hist) > 6:
        print(f"  ...  （共 {len(hist)} 次迭代）")
        print(f"  {len(hist) - 1:>5} {hist[-1][1]:>14.6f}")
    px, py = fk(q_sol)
    print(f"\n  收敛：{ok}   解 q = {fmt_deg(q_sol)}")
    print(f"  验证 FK：({px:.6f}, {py:.6f})，误差 {math.hypot(px - target[0], py - target[1]):.2e} m")

    # -------- 2. 冗余：同一个末端，无数组关节
    print("\n[2] 冗余：同一个末端位置，无穷多组关节角")
    print(f"  目标固定为 {target}，只改变初始 q，看会收敛到哪：")
    print(f"  {'初始 q':>34} {'收敛解 q':>30} {'迭代':>5} {'条件数':>9}")
    solutions = []
    for q_init in [(0.2, 0.4, -0.2), (0.5, -0.5, 0.5), (-0.5, 0.8, -0.4), (1.0, 1.0, -1.0)]:
        qs, h, okk = ik_dls(target, list(q_init))
        verified = fk(qs)
        if math.hypot(verified[0] - target[0], verified[1] - target[1]) > 1e-4:
            continue
        solutions.append(qs)
        print(f"  {fmt_deg(q_init):>34} {fmt_deg(qs):>30} {len(h) - 1:>5} {cond_number(qs):>9.2f}")

    if len(solutions) >= 2:
        spread = max(abs(a - b) for a, b in zip(solutions[0], solutions[-1]))
        print(f"\n  两组解的关节角最大差异：{math.degrees(spread):.1f}°")
        print("  末端轨迹可以一模一样，关节轨迹完全不同。")
        print("  含义：拿'末端位姿'当 action 会丢信息；拿'关节角'当 action")
        print("        又会因为冗余而高度依赖初始状态 —— 这是动作表示的两难。")

    # -------- 3. 接近奇异
    print("\n[3] 接近奇异：目标在最大伸展距离附近")
    max_r = sum(L)
    print(f"  最大伸展 = {max_r:.2f} m。逐步把目标推向边界：")
    print(f"  {'目标半径':>10} {'迭代次数':>9} {'是否收敛':>9} {'最终误差(m)':>13} {'条件数':>10}")
    for frac in [0.5, 0.8, 0.95, 0.99, 0.999, 1.0]:
        r = max_r * frac
        tgt = (r, 0.0)
        qs, h, okk = ik_dls(tgt, [0.1, 0.1, -0.1], max_iter=300)
        px, py = fk(qs)
        err = math.hypot(px - tgt[0], py - tgt[1])
        print(f"  {r:>10.4f} {len(h) - 1:>9} {'是' if okk else '否':>9} "
              f"{err:>13.2e} {cond_number(qs):>10.1f}")

    print("\n  目标越靠边界，条件数越大，同样步长下收敛越慢。")
    print("  完全伸直时（frac=1.0）雅可比降秩，末端沿径向彻底失去运动能力。")

    # -------- 4. 关节限位
    print("\n[4] 关节限位会让 IK 卡在墙上")
    tight = [(-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3)]     # 人为限死每个关节
    qs, h, okk = ik_dls(target, [0.0, 0.0, 0.0], limits=tight, max_iter=300)
    px, py = fk(qs)
    print(f"  把三个关节都限制在 ±0.3 rad（±17°）内，再求 {target}")
    print(f"  结果 q = {fmt_deg(qs)}  末端 = ({px:.4f}, {py:.4f})  "
          f"误差 {math.hypot(px - target[0], py - target[1]):.4f} m")
    print(f"  收敛：{okk}  -> 卡在关节限位上，误差无法消除")
    print("\n  真机上的表现：机械臂顶到限位，策略还在输出指令，")
    print("  数据里会出现'动作在变但状态不变'的片段 —— 这类样本必须剔除，")
    print("  否则模型会学到'无效动作'。")

    # -------- 5. 雅可比的三个真实用途
    print("\n[5] 雅可比能用来干什么（取一个普通构型，避开奇异）")
    q = [math.radians(v) for v in (70.0, -40.0, -80.0)]
    pj = fk(q)
    print(f"  q = (70°, -40°, -80°)，末端在 ({pj[0]:.3f}, {pj[1]:.3f})")
    J = jacobian(q)
    print("  先看三列 —— 每列 = 只有一个关节转 1 rad/s 时，末端的速度：")
    for j in range(3):
        vx, vy = J[0][j], J[1][j]
        print(f"    关节 {j+1}：({vx:+.3f}, {vy:+.3f}) m/s   大小 {math.hypot(vx, vy):.3f} m/s")

    print("\n  ① 速度映射：想让末端以 0.10 m/s 沿 +x 走，各关节该转多快？")
    qd = pinv_apply(J, (0.10, 0.0))
    print(f"     qdot = [{qd[0]:+.4f}, {qd[1]:+.4f}, {qd[2]:+.4f}] rad/s"
          f"  = [{math.degrees(qd[0]):+.2f}, {math.degrees(qd[1]):+.2f}, "
          f"{math.degrees(qd[2]):+.2f}] deg/s")
    back = [sum(J[i][k] * qd[k] for k in range(3)) for i in range(2)]
    print(f"     回代验算 J*qdot = ({back[0]:+.6f}, {back[1]:+.6f}) m/s")
    signs = "".join("正" if v > 0 else "反" for v in qd)
    print(f"     方向是（{signs[0]}、{signs[1]}、{signs[2]}）—— 关节要互相配合，"
          f"不是各转各的。伪逆给的是“关节转速平方和最小”的那组。")

    print("\n  ② 力映射：末端被 10 N 的力往下压，各关节要出多大力矩扛住？")
    F = (0.0, -10.0)
    tau = [J[0][k] * F[0] + J[1][k] * F[1] for k in range(3)]
    print(f"     tau = J^T F = [{tau[0]:+.2f}, {tau[1]:+.2f}, {tau[2]:+.2f}] N·m")
    print("     关节 1 扛得最多：它离末端最远，同样的末端力在它身上杠杆臂最长。")

    print("\n  ③ 零空间：存在一个关节变化方向，走它末端完全不动")
    n = null_direction(q)
    print(f"     n = [{n[0]:+.3f}, {n[1]:+.3f}, {n[2]:+.3f}]"
          f"  （三项都不小，但方向不同：一个关节反向转，另外两个补回来）")
    p0 = fk(q)
    print(f"     {'走多少':>8} {'关节各变化多少':>28} {'末端漂移':>12}")
    for d in (5, 10, 20):
        step = math.radians(d)
        q2 = [q[k] + step * n[k] for k in range(3)]
        dq = [math.degrees(step * n[k]) for k in range(3)]
        p2 = fk(q2)
        print(f"     {d:>6}° [{dq[0]:>+7.2f}°, {dq[1]:>+7.2f}°, {dq[2]:>+7.2f}°]"
              f" {math.hypot(p2[0] - p0[0], p2[1] - p0[1]) * 1000:>9.2f} mm")
    print("     速度级上严格为 0；走大步时只剩二阶残差，所以'零空间'是局部性质。")

    # -------- 6. 冗余的自运动
    print("\n[6] 冗余的自运动：末端钉死，肘部沿弧线自己滑")
    q = list(q_sol)
    print(f"  目标锁死在 {target}，沿零空间一步 1° 地走 240°，中途每 80° 取一次快照：")
    print(f"  {'参数':>5} {'关节角 q':>32} {'肘 (x, y)':>20} {'末端':>18}")
    for s_ in range(0, 241):
        if s_ % 80 == 0:
            ex = L[0] * math.cos(q[0]); ey = L[0] * math.sin(q[0])
            px, py = fk(q)
            print(f"  {s_:>5} {fmt_deg(q):>32} ({ex:>+6.3f}, {ey:>+6.3f})"
                  f"   ({px:.4f}, {py:.4f})")
        n = null_direction(q)
        for k in range(3):
            q[k] += math.radians(1.0) * n[k]
        if any(abs(v) > math.pi - 0.05 for v in q):
            print("   （继续走会撞上 ±180° 关节限位，到此为止）")
            break
        q = list(ik_dls(target, q, max_iter=80)[0])
    print("\n  末端读数一直不变，肘部位置却从右下绕到了左上 ——")
    print("  这就是为什么'同一个任务的关节轨迹'可以差几十度。")

    print("\n" + "=" * 74)
    print("结论（数据平台视角）：")
    print("  1. 同一末端轨迹对应多组关节轨迹，动作表示的选择决定数据能不能复用")
    print("  2. 接近奇异的样本信噪比极差，入库时应标记或剔除")
    print("  3. 关节限位导致的'指令-状态脱节'是常见脏数据，需要自动检测")
    print("  4. 数值 IK 的结果依赖初始值 -> 采集系统必须记录初始构型")
    print("  5. 零空间方向让末端不动而关节在动，是'动作空间选哪个'绕不开的根")
    print("=" * 74)


if __name__ == "__main__":
    main()

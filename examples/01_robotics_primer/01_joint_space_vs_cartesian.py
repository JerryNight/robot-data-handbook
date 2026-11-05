#!/usr/bin/env python3
"""
实验 1：关节空间 vs 笛卡尔空间
================================

一个 2 连杆平面臂（link1 = 1.0 m，link2 = 0.7 m），回答三个问题：

  1. 关节角变了，末端位置怎么变？-> 映射是非线性的
  2. 同一个末端位置，有几组关节角能做到？-> IK 多解
  3. 为什么靠近"完全伸直"时，关节稍微动一点末端就跳很远？-> 奇异性

运行：
    python examples/01_robotics_primer/01_joint_space_vs_cartesian.py

只依赖 Python 标准库。
"""

import math
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

L1 = 1.0   # 大臂长度 m
L2 = 0.7   # 小臂长度 m


# ---------------------------------------------------------------- 正运动学

def fk(theta1, theta2):
    """正运动学：给定关节角（弧度），算出末端位置。有唯一解，便宜。"""
    x = L1 * math.cos(theta1) + L2 * math.cos(theta1 + theta2)
    y = L1 * math.sin(theta1) + L2 * math.sin(theta1 + theta2)
    return x, y


# ---------------------------------------------------------------- 逆运动学

def ik(x, y):
    """
    逆运动学：给定末端位置，反推关节角。

    平面 2 连杆有解析解，且有两组（"肘上" / "肘下"）。
    这是几何法，靠余弦定理：

        cos(theta2) = (r^2 - L1^2 - L2^2) / (2 * L1 * L2)

    r 是目标点到原点的距离。|cos| > 1 表示够不着。
    """
    r_sq = x * x + y * y
    r = math.sqrt(r_sq)

    if r > L1 + L2:
        return []                      # 超出最大伸展距离
    if r < abs(L1 - L2):
        return []                      # 落在内部死区

    cos_t2 = (r_sq - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    cos_t2 = max(-1.0, min(1.0, cos_t2))   # 夹紧，防浮点误差

    t2a = math.acos(cos_t2)
    solutions = []

    for t2 in (t2a, -t2a):
        # 由末端方向角反推第一关节
        phi = math.atan2(y, x)
        t1 = phi - math.atan2(L2 * math.sin(t2), L1 + L2 * math.cos(t2))
        solutions.append((t1, t2))
    return solutions


def is_elbow_up(t1, t2, x, y):
    """
    判断这组解是"肘上"还是"肘下"。

    判据：肘点（第一关节的转轴）落在基座 -> 末端这条连线的哪一侧。
    用叉积比"看 theta2 正负"可靠：

        cross = 末端向量 x 肘点向量
        cross > 0 -> 肘点在连线的逆时针一侧，记作"肘上"
        cross < 0 -> 顺时针一侧，记作"肘下"

    为什么不能只看 theta2 的正负？因为 theta2 是相对于大臂量的角度，
    它的符号只反映"小臂往哪个方向折"，不代表肘在空间里的高低。
    只有当目标点恰好落在 +x 轴上时，两者才碰巧一致；目标一转到别的
    方向，结论就反过来了。不同教材对 theta2 零位的定义也不一样。
    """
    elbow_x = L1 * math.cos(t1)
    elbow_y = L1 * math.sin(t1)
    cross = x * elbow_y - y * elbow_x
    return cross > 0


def condition_number(t1, t2):
    """
    雅可比矩阵的条件数。越大越接近奇异，IK 越不可靠。

    2 连杆平面臂的雅可比：
        J = [ -L1*s1 - L2*s12,  -L2*s12 ]
            [  L1*c1 + L2*c12,   L2*c12 ]

    条件数 = 最大奇异值 / 最小奇异值。奇异时最小奇异值 -> 0，条件数 -> ∞。
    这里用行列式倒数做简化代理指标（等价于最小奇异值的下界）。
    """
    det = L1 * L2 * math.sin(t2)
    if abs(det) < 1e-12:
        return float("inf")
    return abs((L1 * L1 + L2 * L2 + 2 * L1 * L2 * math.cos(t2)) / det)


# ---------------------------------------------------------------- 文本可视化

def plot_workspace(width=61, height=21):
    """把关节角空间采样映射成末端点，用字符画出来。"""
    pts = []
    for i in range(721):
        for j in range(361):
            t1 = -math.pi + i * (2 * math.pi / 720)
            t2 = -math.pi + j * (2 * math.pi / 360)
            pts.append(fk(t1, t2))

    grid = [[" "] * width for _ in range(height)]
    x_min, x_max = -(L1 + L2), (L1 + L2)
    y_min, y_max = -(L1 + L2), (L1 + L2)

    for x, y in pts:
        col = int((x - x_min) / (x_max - x_min) * (width - 1))
        row = int((y_max - y) / (y_max - y_min) * (height - 1))
        if 0 <= row < height and 0 <= col < width:
            grid[row][col] = "."

    # 标记原点
    oc = int((0 - x_min) / (x_max - x_min) * (width - 1))
    orow = int((y_max - 0) / (y_max - y_min) * (height - 1))
    grid[orow][oc] = "O"

    print(f"  可达工作空间（半径 {L1 + L2:.1f} m 的圆盘，'.' 为可达点）")
    print("  +" + "-" * width + "+")
    for row in grid:
        print("  |" + "".join(row) + "|")
    print("  +" + "-" * width + "+")
    print("  O = 基座原点")


def main():
    print("=" * 66)
    print("实验 1：关节空间 vs 笛卡尔空间")
    print(f"2 连杆平面臂，L1 = {L1} m，L2 = {L2} m")
    print("=" * 66)

    # -------- 1. 正运动学：关节角 -> 末端
    print("\n[1] 正运动学：改关节角，看末端怎么动")
    print(f"  {'theta1':>8} {'theta2':>8} {'末端 x':>9} {'末端 y':>9}   {'|末端|':>8}")
    for d1, d2 in [(0, 0), (0, 90), (30, 45), (60, 90), (90, -90), (45, 0)]:
        x, y = fk(math.radians(d1), math.radians(d2))
        print(f"  {d1:>8}° {d2:>8}° {x:>9.4f} {y:>9.4f}   {math.hypot(x, y):>8.4f}")

    print("\n  注意 (30,45) 和 (45,0) 这两行：关节角各变化 15° 和 45°，")
    print("  末端位移完全不同。关节空间是均匀的，笛卡尔空间不是。")

    # -------- 2. 逆运动学：一个点，两组解
    print("\n[2] 逆运动学：同一个末端位置，有两组关节角")
    tx, ty = 1.2, 0.6
    print(f"  目标点：({tx}, {ty})，距离 {math.hypot(tx, ty):.4f} m")
    sols = ik(tx, ty)
    for k, (t1, t2) in enumerate(sols, 1):
        name = "肘上 (elbow up)" if is_elbow_up(t1, t2, tx, ty) else "肘下 (elbow down)"
        veri_x, veri_y = fk(t1, t2)
        print(f"  解 {k}: theta1={math.degrees(t1):>8.2f}°  theta2={math.degrees(t2):>8.2f}°"
              f"  -> 验证 FK = ({veri_x:.4f}, {veri_y:.4f})   {name}")

    print("\n  两组解末端位置相同，但姿态完全不同。")
    print("  数据平台的含义：只记末端位置是不够的，关节构型也必须记。")

    # -------- 3. 奇异性
    print("\n[3] 奇异性：theta2 -> 0 时，雅可比降秩")
    print(f"  {'theta2':>8} {'|det J|':>10} {'条件数':>10}   沿径向再走 1mm？")
    for d2 in [90, 45, 10, 5, 2, 1, 0.1]:
        t2 = math.radians(d2)
        det = L1 * L2 * math.sin(t2)
        cond = condition_number(0.0, t2)

        # 当前末端位置，沿径向（远离基座方向）挪 1 mm，看还能不能解出 IK
        x0, y0 = fk(0.0, t2)
        r = math.hypot(x0, y0)
        x1, y1 = x0 * (r + 0.001) / r, y0 * (r + 0.001) / r
        reach = "可达" if ik(x1, y1) else "不可达（已顶到边界）"
        print(f"  {d2:>7}° {abs(det):>10.5f} {cond:>10.1f}   {reach}")

    print("\n  theta2 = 0 是完全伸直，两臂共线 -> 雅可比两列线性相关 -> 降秩。")
    print("  条件数从 2 涨到 2000+，意味着逆问题极度敏感：")
    print("  末端只要有一点点误差，反推出来的关节角就能差几十度。")
    print("  真机表现：IK 在奇异点附近剧烈震荡，关节发出'咔咔'声。")
    print("\n  注意一个巧合：2 连杆的工作空间边界恰好就是奇异位形")
    print("  （伸直 = 够得最远 = 奇异）。冗余臂（7-DoF）的奇异藏在工作空间")
    print("  内部，不会顶到边界就报错，反而更难发现。")

    # -------- 4. 工作空间
    print("\n[4] 工作空间长什么样")
    plot_workspace()

    # -------- 5. 边界情况
    print("\n[5] 够不着的点会怎样")
    for tx, ty in [(1.8, 0.0), (2.0, 0.0), (0.2, 0.1)]:
        r = math.hypot(tx, ty)
        sols = ik(tx, ty)
        if sols:
            status = "可达，2 组解"
        elif r > L1 + L2:
            status = "不可达（超出最大伸展，太远）"
        else:
            status = "不可达（落在内部死区，太近）"
        print(f"  ({tx}, {ty})  距离 {r:.3f} m  ->  {status}")
    print(f"\n  最大伸展 = L1 + L2 = {L1 + L2:.1f} m，内部死区 = |L1 - L2| = {abs(L1 - L2):.1f} m")

    print("\n" + "=" * 66)
    print("结论（数据平台视角）：")
    print("  1. 关节角 -> 末端位置 是多对一映射，信息会丢，必须存原始关节角")
    print("  2. 同一末端位置有多种构型，'末端轨迹'不足以还原动作")
    print("  3. 奇异位形附近的数据噪声会被放大，采集时应避开")
    print("=" * 66)


if __name__ == "__main__":
    main()

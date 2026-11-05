#!/usr/bin/env python3
"""
实验 2：齐次变换与 mini TF 树
==============================

回答一个数据平台每天都要处理的问题：

    "相机在 t=0.5s 拍到的那个杯子，在机器人基座坐标系下的位置是？"

要算出这个答案，需要把一条坐标链上的变换按时间插值后连乘。
这个脚本用 120 行手写实现 TF2 的核心语义：

  - 4x4 齐次变换矩阵（旋转 + 平移）
  - 变换组合、求逆、点变换
  - 带时间戳的变换树（两帧关键帧 + 线性插值）
  - 外参偏差如何污染下游结果

运行：
    python examples/01_robotics_primer/02_frames_and_transforms.py

只依赖 Python 标准库。
"""

import math
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ================================================================ 4x4 变换

def identity():
    return [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]


def translation(x, y, z):
    m = identity()
    m[0][3], m[1][3], m[2][3] = x, y, z
    return m


def rpy(roll, pitch, yaw):
    """
    欧拉角 -> 旋转矩阵，顺序 Rz * Ry * Rx（即先绕 x 转，最后绕 z 转）。

    这是 ROS 的默认约定（REP-103，rpy 固定轴 roll/pitch/yaw）。
    """
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = [[1, 0, 0], [0, cr, -sr], [0, sr, cr]]
    ry = [[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]
    rz = [[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]

    def mul3(a, b):
        return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
                for i in range(3)]

    r = mul3(rz, mul3(ry, rx))
    m = identity()
    for i in range(3):
        for j in range(3):
            m[i][j] = r[i][j]
    return m


def matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)]
            for i in range(4)]


def inverse(m):
    """
    刚体变换的逆：R^T 和 -R^T * t。
    不要用通用矩阵求逆——刚体变换有闭式解，更快更准。
    """
    r = [[m[j][i] for j in range(3)] for i in range(3)]      # 旋转部分转置
    t = [m[0][3], m[1][3], m[2][3]]

    out = identity()
    for i in range(3):
        for j in range(3):
            out[i][j] = r[i][j]
    for i in range(3):
        out[i][3] = -sum(r[i][k] * t[k] for k in range(3))
    return out


def apply(m, point):
    """把一个 3D 点变换到目标坐标系。"""
    x, y, z = point
    return (
        m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3],
        m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3],
        m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3],
    )


def lerp(a, b, t):
    """两个变换之间线性插值（简化版 TF 插值：位置线性、旋转也未做球面插值）。"""
    return [[a[i][j] + (b[i][j] - a[i][j]) * t for j in range(4)] for i in range(4)]


# ================================================================ mini TF 树

class MiniTF:
    """
    极简 TF 树。

    结构：edges[(parent, child)] = [(t0, M0), (t1, M1), ...]
    M 表示 child 在 parent 坐标系下的位姿。

    查询 query(target, source, t) 返回把 source 坐标系的点
    变换到 target 坐标系所需的矩阵。
    """

    def __init__(self):
        self.edges = {}

    def add(self, parent, child, timestamp, matrix):
        self.edges.setdefault((parent, child), []).append((timestamp, matrix))

    def _lookup_edge(self, parent, child, t):
        """取单条边的变换，按时间在关键帧之间插值。"""
        frames = sorted(self.edges.get((parent, child), []), key=lambda f: f[0])
        if not frames:
            raise KeyError(f"没有边 {parent} -> {child}")

        if len(frames) == 1 or t <= frames[0][0]:
            return frames[0][1]
        if t >= frames[-1][0]:
            return frames[-1][1]

        for k in range(len(frames) - 1):
            ta, ma = frames[k]
            tb, mb = frames[k + 1]
            if ta <= t <= tb:
                alpha = (t - ta) / (tb - ta)
                return lerp(ma, mb, alpha)
        return frames[-1][1]

    def _path(self, a, b):
        """在树上找 a 到 b 的路径，返回 [(frame_name, direction)]。"""
        # 建无向图
        graph = {}
        for (p, c) in self.edges:
            graph.setdefault(p, []).append(c)
            graph.setdefault(c, []).append(p)

        # BFS
        prev = {a: None}
        queue = [a]
        while queue:
            cur = queue.pop(0)
            if cur == b:
                break
            for nxt in graph.get(cur, []):
                if nxt not in prev:
                    prev[nxt] = cur
                    queue.append(nxt)

        if b not in prev:
            raise KeyError(f"{a} 和 {b} 不连通")

        path = []
        node = b
        while prev[node] is not None:
            path.append((prev[node], node))
            node = prev[node]
        path.reverse()
        return path

    def query(self, target, source, t):
        """返回 source -> target 的变换矩阵。"""
        path = self._path(source, target)
        m = identity()
        for (a, b) in path:
            # 路径沿 source -> target 走，但边是按 (父, 子) 存的，
            # 所以 (a, b) 这一跳通常对应存下来的边 (b, a)，直接用即可。
            if (b, a) in self.edges:
                step = self._lookup_edge(b, a, t)
            else:
                step = inverse(self._lookup_edge(a, b, t))
            m = matmul(step, m)
        return m


# ================================================================ 演示

def main():
    print("=" * 70)
    print("实验 2：齐次变换与 mini TF 树")
    print("=" * 70)

    # -------- 建一条真实的链：base -> shoulder -> upper -> wrist -> camera
    tf = MiniTF()

    # 静态部分（永远不会变的连杆偏置）
    tf.add("base", "shoulder", 0.0, translation(0.0, 0.0, 0.35))

    # 动态部分：两个关键帧，t=0 和 t=1
    tf.add("shoulder", "upper", 0.0, rpy(0.0, 0.0, 0.0))
    tf.add("shoulder", "upper", 1.0, rpy(0.0, 0.0, math.radians(30)))

    tf.add("upper", "wrist", 0.0, matmul(translation(0.4, 0.0, 0.0),
                                        rpy(0.0, math.radians(-20), 0.0)))
    tf.add("upper", "wrist", 1.0, matmul(translation(0.4, 0.0, 0.0),
                                        rpy(0.0, math.radians(-20), math.radians(10))))

    # 手眼标定给的外参（相机相对腕部），这里假设是静态的
    hand_eye = matmul(translation(0.08, 0.0, 0.05), rpy(0.0, 0.0, math.radians(-90)))
    tf.add("wrist", "camera", 0.0, hand_eye)

    print("\n[1] TF 树结构")
    for (p, c), frames in sorted(tf.edges.items()):
        print(f"  {p:<10} -> {c:<10}  {len(frames)} 个关键帧  "
              f"(t = {[f[0] for f in frames]})")

    # -------- 核心场景
    print("\n[2] 核心场景：相机看到杯子，求它在基座下的位置")
    cup_in_camera = (0.15, -0.05, 0.60)      # 相机坐标系里的一个点

    print(f"  相机坐标系下的杯子位置：{cup_in_camera}")
    print(f"\n  {'时刻':>6} {'base 下坐标':>32} {'相对 t=0 的位移':>16}")

    base_at_0 = None
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        m = tf.query("base", "camera", t)
        p = apply(m, cup_in_camera)
        if base_at_0 is None:
            base_at_0 = p
        drift = math.dist(p, base_at_0)
        print(f"  {t:>6.2f} ({p[0]:>8.4f}, {p[1]:>8.4f}, {p[2]:>8.4f})   {drift:>12.4f} m")

    print("\n  杯子在相机坐标里的三元组没变，但相机随手臂在动")
    print("  -> 同一个三元组落到 base 下就是不同的点。")
    print("  这就是为什么 TF 查询必须带时间戳：不带时间戳的变换没有意义。")

    # -------- 外参误差的影响
    print("\n[3] 外参错 2 度，会怎样？")
    bad_eye = matmul(translation(0.08, 0.0, 0.05),
                     rpy(math.radians(2), 0.0, math.radians(-90)))   # 俯仰偏 2°

    tf_bad = MiniTF()
    for (p, c), frames in tf.edges.items():
        for (ts, mat) in frames:
            tf_bad.add(p, c, ts, mat)
    tf_bad.edges[("wrist", "camera")] = [(0.0, bad_eye)]

    errs = []
    for t in [0.0, 0.5, 1.0]:
        good = apply(tf.query("base", "camera", t), cup_in_camera)
        bad = apply(tf_bad.query("base", "camera", t), cup_in_camera)
        err = math.dist(good, bad)
        errs.append(err)
        print(f"  t={t:.1f}  正确 ({good[0]:.4f}, {good[1]:.4f}, {good[2]:.4f})  "
              f"偏 2° 后 ({bad[0]:.4f}, {bad[1]:.4f}, {bad[2]:.4f})  "
              f"误差 {err * 1000:.1f} mm")

    print(f"\n  2° 的旋转误差，在 0.6 m 的距离上产生约 {max(errs) * 1000:.0f} mm 的位置误差。")
    print("  夹爪开合精度通常在 1 mm 量级 —— 这个误差已经足以让抓取失败。")
    print("  而标定误差不会报错，只会让数据'看起来正常但用不了'。")

    # -------- 逆变换
    print("\n[4] 变换求逆：验证 M * M^-1 = I（用 t=0 的关键帧，矩阵是合法刚体变换）")
    m = tf.query("base", "camera", 0.0)
    mi = inverse(m)
    prod = matmul(m, mi)
    max_err = max(abs(prod[i][j] - (1.0 if i == j else 0.0))
                  for i in range(4) for j in range(4))
    print(f"  最大偏差：{max_err:.2e}（浮点误差量级即为正确）")

    # 方向自检：往返一次必须回到原点，这是查"query 方向写反了"最省事的办法
    rt = matmul(tf.query("base", "camera", 0.0), tf.query("camera", "base", 0.0))
    rt_err = max(abs(rt[i][j] - (1.0 if i == j else 0.0))
                 for i in range(4) for j in range(4))
    print(f"  方向自检 query(base,camera)·query(camera,base) 与 I 的最大偏差：{rt_err:.2e}")
    print("  注意这条路走不通：两个方向同时写反时，乘积照样是 I。")
    print("  要抓方向 bug 只能看物理量：")
    print(f"  杯子在 base 下 z = {apply(m, cup_in_camera)[2]:.3f} m，"
          f"应当高于肩关节 0.35 m 而不是落在下方。")

    # -------- 旋转插值的坑
    print("\n[5] 为什么旋转不能用矩阵线性插值")
    print("  上面的 _lookup_edge 为了简单，对 4x4 矩阵做了线性插值。")
    print("  这正是很多数据平台踩的坑。检查一下插值结果的旋转部分是否还正交：")

    m_half = tf.query("base", "camera", 0.5)
    r = [[m_half[i][j] for j in range(3)] for i in range(3)]
    rtr = [[sum(r[k][i] * r[k][j] for k in range(3)) for j in range(3)]
           for i in range(3)]
    orth_err = max(abs(rtr[i][j] - (1.0 if i == j else 0.0))
                   for i in range(3) for j in range(3))
    print(f"  R^T·R 与单位阵的最大偏差：{orth_err:.2e}")

    if orth_err > 1e-6:
        print("\n  非零 -> 插值出来的矩阵不是合法旋转（行列式不再为 1）。")
        print("  用它求逆、反投影、算雅可比，结果会系统性偏移。")
        print("  正确做法：旋转在四元数或李群上插值（slerp / SE(3) 指数映射），")
        print("  位置才可以线性插值。这是 TF2 内部做的事，自己实现很容易漏掉。")
    else:
        print("\n  恰好为正交 —— 本次关键帧旋转量小，误差落在容差内。")
        print("  但这不是普遍规律，旋转量大时必然失效。")

    print("\n  [提示] 本脚本的 lerp 是故意留的教学简化版。")
    print("  生产环境的 TF 查询应直接用 tf2_ros / scipy.spatial.transform，")
    print("  不要自己写插值。")

    print("\n" + "=" * 70)
    print("结论（数据平台视角）：")
    print("  1. 变换必须带时间戳，'某时刻的位置'是唯一有意义的表述")
    print("  2. 外参误差按 距离 x 角度 放大，标定缺失的数据集价值直接归零")
    print("  3. 变换链要做完整性校验：任何一环缺失，整条链算不出来")
    print("  4. 插值旋转必须用 slerp / 李群，矩阵线性插值会破坏正交性")
    print("=" * 70)


if __name__ == "__main__":
    main()

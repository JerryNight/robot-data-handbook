#!/usr/bin/env python3
"""
实验 3：标定——内参能估准，外参会悄悄漂，而且漂了之后数据看不出来。

运行：python examples/03_data_pipeline/03_calibration.py

真正的难点不是「怎么标定」，而是这三件事：
    [1] 一组内参要多少张图才稳？答案比你想的多
    [2] 外参误差分「平移型」和「旋转型」，两者在残差上的指纹完全不同
    [3] 标定漂了之后，能不能从数据本身发现？能，但要借一个外部参考系
"""

import math
import random


def width(s: str) -> int:
    return sum(2 if ord(c) > 0x2E80 else 1 for c in s)


def pad(s: str, n: int) -> str:
    return s + " " * max(0, n - width(s))


# ---------------------------------------------------------------- 小工具

def solve2(a, b, c, d, e, f):
    """解 2x2 线性方程组  [a b; c d] [x y]^T = [e f]^T"""
    det = a * d - b * c
    if abs(det) < 1e-18:
        return None
    return (e * d - b * f) / det, (a * f - e * c) / det


def fit_plane(pts):
    """
    最小二乘拟合平面 z = ax + by + c（对水平桌面够用）。
    返回 (a, b, c)。用于从点云反推相机姿态。
    """
    n = len(pts)
    sx = sum(p[0] for p in pts)
    sy = sum(p[1] for p in pts)
    sz = sum(p[2] for p in pts)
    sxx = sum(p[0] * p[0] for p in pts)
    sxy = sum(p[0] * p[1] for p in pts)
    syy = sum(p[1] * p[1] for p in pts)
    sxz = sum(p[0] * p[2] for p in pts)
    syz = sum(p[1] * p[2] for p in pts)

    # 3x3 正规方程，用克拉默法则解（比写高斯消元短）
    m = [[sxx, sxy, sx], [sxy, syy, sy], [sx, sy, float(n)]]
    rhs = [sxz, syz, sz]

    def det3(mat):
        return (mat[0][0] * (mat[1][1] * mat[2][2] - mat[1][2] * mat[2][1])
                - mat[0][1] * (mat[1][0] * mat[2][2] - mat[1][2] * mat[2][0])
                + mat[0][2] * (mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0]))

    d0 = det3(m)
    if abs(d0) < 1e-18:
        return None
    out = []
    for col in range(3):
        mm = [row[:] for row in m]
        for r in range(3):
            mm[r][col] = rhs[r]
        out.append(det3(mm) / d0)
    return tuple(out)


# ---------------------------------------------------------------- [1] 内参

def board_points(rng, n, spread_x, spread_y):
    """标定板上的点：板面近似平坦，所以 z 的散布远小于 x/y。"""
    return [(rng.uniform(-spread_x, spread_x),
             rng.uniform(-spread_y, spread_y),
             rng.uniform(-0.002, 0.002)) for _ in range(n)]


def rms_residual(pts, f_cand, d):
    """
    在给定 (f, d) 下算重投影残差。

    u 和 v 都用「减均值」处理，等价于把 cx、cy 一并优化掉——
    因为它们只是平移量。所以残差只反映**尺度**是否对。
    """
    us = [f_cand * p[0] / (d + p[2]) for p in pts]
    vs = [f_cand * p[1] / (d + p[2]) for p in pts]
    mu = sum(us) / len(us)
    mv = sum(vs) / len(vs)
    ou = sum(p[3] for p in pts) / len(pts)
    ov = sum(p[4] for p in pts) / len(pts)
    e2 = 0.0
    for i, p in enumerate(pts):
        e2 += (us[i] - mu - (p[3] - ou)) ** 2 + (vs[i] - mv - (p[4] - ov)) ** 2
    return math.sqrt(e2 / (2 * len(pts)))


def best_distance(pts, f_cand, lo=0.4, hi=2.4, steps=400):
    """单视图：给定候选焦距，搜一个最合适的距离。返回 (残差, d)。"""
    best = (float("inf"), lo)
    for k in range(steps + 1):
        d = lo + (hi - lo) * k / steps
        r = rms_residual(pts, f_cand, d)
        if r < best[0]:
            best = (r, d)
    return best


def part1_intrinsics():
    print("\n[1] 内参的真正难点：焦距和距离是同一个自由度")
    print("-" * 76)
    F_TRUE, CX_TRUE, CY_TRUE = 600.0, 319.4, 241.2
    D_TRUE = 1.000          # 相机到标定板中心的真实距离

    print("  模型：u = f·x/(d+z) + cx。标定板是平的，z ≈ 0，")
    print("  于是 f/(d+z) ≈ f/d —— **只有这个比值进得了图像**。")
    print("  所以单张标定板图，永远解不出 f 和 d 各是多少，只能解出它们的比。")
    print()

    rng = random.Random(3)
    n = 60
    pts = []
    for x, y, z in board_points(rng, n, 0.15, 0.11):
        u = F_TRUE * x / (D_TRUE + z) + CX_TRUE + rng.gauss(0, 0.2)
        v = F_TRUE * y / (D_TRUE + z) + CY_TRUE + rng.gauss(0, 0.2)
        pts.append((x, y, z, u, v))

    print("  现在假装我们不知道 f，试一堆候选值，各自配一个最优距离：")
    print("  %s%s%s%s" % (pad("候选焦距", 12), pad("最优距离", 14),
                          pad("f/d 比值", 14), "残差 RMS"))
    print("  " + "-" * 60)
    for f_cand in (480.0, 540.0, 600.0, 660.0, 750.0):
        rms, d = best_distance(pts, f_cand)
        print("  %s%s%s%s" % (pad("%.0f" % f_cand, 12), pad("%.3f m" % d, 14),
                              pad("%.1f" % (f_cand / d), 14), "%.3f px" % rms))

    print()
    print("  480 和 750 的残差与真值 600 几乎一样小。这不是巧合，")
    print("  是**结构性退化**：整条 f/d = 常数 的曲线上，图像一模一样。")
    print("  单视图标定的所有「最优解」都不唯一，优化器只是随手停在了某一点。")

    print("\n  怎么打破？给一个**已知的长度**。机器人正好适合干这个：")
    print("  控制末端精确移动 300mm，拍第二张。这个 300mm 是机械量，")
    print("  不依赖相机，于是 (f, d) 被两个方程锁死。\n")

    pts2 = []
    for x, y, z in board_points(rng, n, 0.15, 0.11):
        d2 = D_TRUE + 0.300
        u = F_TRUE * x / (d2 + z) + CX_TRUE + rng.gauss(0, 0.2)
        v = F_TRUE * y / (d2 + z) + CY_TRUE + rng.gauss(0, 0.2)
        pts2.append((x, y, z, u, v))

    print("  %s%s%s" % (pad("候选焦距", 12), pad("拟合出的 d0", 16), "两视图联合残差"))
    print("  " + "-" * 56)
    for f_cand in (480.0, 540.0, 600.0, 660.0, 750.0):
        best = (float("inf"), 0.0)
        for k in range(401):
            d0 = 0.4 + 2.0 * k / 400
            # 第二视图的距离由「已知位移 300mm」决定，不是自由参数
            r1 = rms_residual(pts, f_cand, d0)
            r2 = rms_residual(pts2, f_cand, d0 + 0.300)
            r = math.sqrt((r1 ** 2 + r2 ** 2) / 2)
            if r < best[0]:
                best = (r, d0)
        mark = "  <-- 真值" if abs(f_cand - F_TRUE) < 1e-9 else ""
        print("  %s%s%s%s" % (pad("%.0f" % f_cand, 12), pad("%.3f m" % best[1], 16),
                              "%.3f px" % best[0], mark))

    print()
    print("  加了 300mm 这个机械约束之后，残差在 600 处才有明显的谷。")
    print("  这就是为什么标定流程里要「平移一段已知距离」或者「多姿态」——")
    print("  每多一个**独立的已知量**，就消掉一个自由度。")
    print()
    print("  顺带解释了手眼标定的经典要求：至少 3 个**旋转轴不共线**的姿态。")
    print("  共线的姿态提供的约束是重复的，等价于只有一张图。")


# ---------------------------------------------------------------- [2] 外参

def part2_extrinsics_signature():
    print("\n[2] 外参误差的两种形态，在残差上有不同指纹")
    print("-" * 76)
    print("  场景：相机装在腕部，标定得到一个相机→末端的外参 X。")
    print("  之后用 6 个不同姿态去摸同一个固定点 P（真值已知），看残差。")
    print()

    P_WORLD = (0.55, 0.10, 0.02)
    poses = [(0.4, 0.0, 0.3, 0), (0.45, 0.1, 0.35, 20),
             (0.5, -0.1, 0.4, -20), (0.55, 0.05, 0.45, 35),
             (0.6, -0.05, 0.35, -35), (0.5, 0.15, 0.3, 10)]

    def residual(t_err, r_err_deg):
        """
        误差传播，只保留一次项：

            真值   p_cam = R · p_end + t
            估计   p̂_end = R̂⁻¹ (p_cam − t̂),   R̂ = R·ΔR,  t̂ = t + Δt
            代入   p̂_end = ΔR⁻¹ · p_end − ΔR⁻¹ · Δt

        所以残差 = |p̂_end − p_end|：
            · 纯平移：与 p_end 无关，恒为 |Δt|
            · 纯旋转：等于 2·|p_end|·sin(Δθ/2)，随力臂增长
        """
        out = []
        dr = math.radians(r_err_deg)
        c, s = math.cos(dr), math.sin(dr)
        for tx, ty, tz, rz in poses:
            # 世界系 -> 末端系（真实的几何关系，不含误差）
            cw, sw = math.cos(math.radians(rz)), math.sin(math.radians(rz))
            dx, dy = P_WORLD[0] - tx, P_WORLD[1] - ty
            px = cw * dx + sw * dy
            py = -sw * dx + cw * dy
            # ΔR⁻¹ 作用在 (p_end − Δt) 上；2D 旋转取逆 = 转 -dr
            ax, ay = px - t_err[0], py - t_err[1]
            ex = c * ax - s * ay          # ΔR⁻¹ = R(-dr)
            ey = s * ax + c * ay
            ex, ey = ex - px, ey - py
            out.append(math.hypot(ex, ey))
        return out

    print("  %s%s%s" % (pad("误差类型", 22), pad("残差均值", 14), "残差范围"))
    print("  " + "-" * 58)
    cases = [
        ("无误差", (0, 0, 0), 0.0),
        ("纯平移 10mm", (0.010, 0.0, 0.0), 0.0),
        ("纯旋转 1°", (0, 0, 0), 1.0),
        ("纯旋转 3°", (0, 0, 0), 3.0),
        ("平移 10mm + 旋转 3°", (0.010, 0.0, 0.0), 3.0),
    ]
    for label, t_err, r_err in cases:
        rs = residual(t_err, r_err)
        m = sum(rs) / len(rs)
        print("  %s%s%s" % (pad(label, 22), pad("%.1f mm" % (m * 1000), 14),
                            "%.1f ~ %.1f mm" % (min(rs) * 1000, max(rs) * 1000)))

    print()
    print("  指纹很清楚：")
    print("    · **平移误差** → 每个姿态的残差几乎一样大（因为它与位姿无关）")
    print("    · **旋转误差** → 残差随「点到旋转中心的距离」变化，范围被拉开")
    print("  所以看到「残差随位置线性增长」，先查旋转；")
    print("  看到「所有位置都偏同一截」，先查平移。这条判据比重新标一次更快。")

    print()
    print("  为什么旋转误差更危险：它会被力臂放大 ——")
    print("  %s%s" % (pad("工作距离", 16), "1° 旋转误差造成的偏移"))
    print("  " + "-" * 44)
    for d in (0.1, 0.3, 0.5, 1.0, 2.0):
        print("  %s%.1f mm" % (pad("%.1f m" % d, 16), d * math.tan(math.radians(1)) * 1000))


# ---------------------------------------------------------------- [3] 融合

def part3_fusion_misalignment():
    print("\n[3] 外参错了，多视角融合会错多少")
    print("-" * 76)
    print("  两个相机看同一块区域，各自出点云后拼到一起。")
    print("  如果外参（相对位姿）有误差，同一块表面会出现**双层**。")
    print()
    print("  %s%s%s" % (pad("外参角度误差", 16), pad("1m 处错位", 14), "视觉表现"))
    print("  " + "-" * 62)
    for deg in (0.1, 0.5, 1.0, 2.0, 5.0):
        e = 1.0 * math.tan(math.radians(deg)) * 1000
        note = ("看不出来" if e < 2 else
                "边缘发虚，像失焦" if e < 8 else
                "明显重影" if e < 25 else "双层结构，不可用")
        print("  %s%s%s" % (pad("%.1f°" % deg, 16), pad("%.1f mm" % e, 14), note))

    print()
    print("  注意 0.5° 就只有 8.7mm —— 而机器人领域「标定得还行」的标准")
    print("  经常是 0.5° 量级。对抓取够用，对**精细操作不够**。")
    print("  更麻烦的是：这种误差在单帧里看不出来，只有融合之后才显形，")
    print("  所以必须在数据集里存好每个传感器的外参和标定时间。")


# ---------------------------------------------------------------- [4] 检测

def part4_detect_drift():
    print("\n[4] 标定漂了，能不能从数据本身发现？能——借重力")
    print("-" * 76)
    print("  场景：相机装在腕部，某次碰撞把支架撞歪了 1.5°，没人发现。")
    print("  之后采的所有数据里，桌面在相机系里都歪了 1.5°。")
    print()
    print("  检测办法：拟合桌面平面，比较它与**重力方向**的夹角。")
    print("  重力来自 IMU，是独立于相机的参考 —— 两个传感器不会一起错。")
    print()

    rng = random.Random(5)
    THRESH = 0.3      # 判定阈值：桌面应该是水平的，倾角超过 0.3° 就报警
    print("  %s%s%s" % (pad("真实倾斜", 14), pad("拟合出的法向角", 20),
                          "判定（阈值 %.1f°）" % THRESH))
    print("  " + "-" * 62)
    for true_deg in (0.0, 1.5, 5.0):
        th = math.radians(true_deg)
        pts = []
        for _ in range(2000):
            x = rng.uniform(-0.4, 0.4)
            y = rng.uniform(-0.3, 0.3)
            z = x * math.sin(th) + rng.gauss(0, 0.002)     # 2mm 深度噪声
            pts.append((x, y, z))
        a, b, _ = fit_plane(pts)
        got = math.degrees(math.atan(math.hypot(a, b)))
        note = "正常" if got <= THRESH else "报警：相机可能被碰过"
        print("  %s%s%s" % (pad("%.1f°" % true_deg, 14),
                            pad("%.2f°" % got, 20), note))

    print()
    print("  这个方法的价值在于：它**不需要重新标定**，只需要一段静止数据 + 重力。")
    print("  可以做成采集时的自动质检项——每天开工前拍 10 秒桌面，")
    print("  拟合出的倾角超过 0.3° 就提示「相机可能被碰过」。")
    print()
    print("  这也解释了为什么数据集必须存 IMU：它的作用不只是观测，")
    print("  还是所有其他传感器的**参照系**。")

    print("\n" + "-" * 76)
    print("  四句话总结标定这件事：")
    print("  1. 内参可以标得很准，前提是标定板铺满画面、姿态足够多")
    print("  2. 外参一定会漂（碰撞、温度、装配松动），所以要能检测不能只靠重标")
    print("  3. 残差的形状比残差的大小更有诊断价值")
    print("  4. 标定参数是**数据**，必须进数据集、带时间戳、能追溯")


if __name__ == "__main__":
    print("=" * 78)
    print("实验 3：标定的精度、误差形态与漂移检测")
    print("=" * 78)
    part1_intrinsics()
    part2_extrinsics_signature()
    part3_fusion_misalignment()
    part4_detect_drift()
    print("=" * 78)

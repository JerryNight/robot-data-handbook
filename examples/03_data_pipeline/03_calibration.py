#!/usr/bin/env python3
"""
实验 3：标定——重投影残差 0.2 px 的时候，焦距可以错 25%。

运行：python examples/03_data_pipeline/03_calibration.py

标定报告上最漂亮的那个数字是「重投影残差 0.2 px」。这一篇要说明它为什么
几乎不能证明任何事，以及真正能证明事情的三个东西：

    [1] 内参退化：f 和 d 只有比值进得了图像，残差在整条曲线上都是 0.2 px
    [2] 已知长度：一条机械尺子把退化打破，尺子本身多准就直接变成焦距多准
    [3] 标定板覆盖率：σ_f ∝ 1/(板宽 · √点数)，这是「铺满画面」的量化理由
    [4] 外参指纹：平移误差与力臂无关，旋转误差随力臂线性增长
    [5] 力臂：同一个 1°，在 10 cm 处是 1.7 mm，在 2 m 处是 34.9 mm
    [6] 融合：外参误差在单帧里看不见，拼起来才是双层
    [7] 可观测性：借重力只能测 6 个自由度里的 2 个
    [8] 灵敏度：32 个点就够检出 0.3°，瓶颈在参考物不在算法
    [9] 热胀：所有常见材料都要 50 K 以上才胀得出 0.3°
    [10] 检测节奏：所以 0.3° 都是机械事件，每天查一次就够，但必须每天
    [11] 决策表
"""

import math
import random
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def width(s: str) -> int:
    return sum(2 if ord(c) > 0x2E80 else 1 for c in s)


def pad(s: str, n: int) -> str:
    return s + " " * max(0, n - width(s))


def head(*cells) -> str:
    return "  " + "".join(pad(c, w) for c, w in cells)


# ---------------------------------------------------------------- 几何小工具

def rot_xyz(x, y, z, ax_deg, ay_deg, az_deg):
    """Rz·Ry·Rx，依次绕 x、y、z 转。角度单位是度。"""
    ax, ay, az = math.radians(ax_deg), math.radians(ay_deg), math.radians(az_deg)
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)

    y1, z1 = cx * y - sx * z, sx * y + cx * z
    x2, z2 = cy * x + sy * z1, -sy * x + cy * z1
    return cz * x2 - sz * y1, sz * x2 + cz * y1, z2


def fit_plane(pts):
    """
    最小二乘拟合 z = ax + by + c，返回 (a, b, c)。
    法向与 z 轴的夹角就是 sqrt(a²+b²) 的反正切（小角度下）。
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
    return out[0], out[1], out[2]


def plane_angle_deg(pts):
    """拟合平面法向与 z 轴的夹角。"""
    a, b, _ = fit_plane(pts)
    return math.degrees(math.atan(math.hypot(a, b)))


# ---------------------------------------------------------------- 针孔模型

F_TRUE = 600.0          # 真值焦距，像素
D_TRUE = 1.000          # 相机到标定板中心的真实距离，米
CX_TRUE, CY_TRUE = 319.4, 241.2


def board_points(rng, n, half_w, half_h, thickness=0.002):
    """标定板上的点。板是平的，所以 z 的散布（2mm）远小于 x/y。"""
    return [(rng.uniform(-half_w, half_w), rng.uniform(-half_h, half_h),
             rng.uniform(-thickness, thickness)) for _ in range(n)]


def project(pts, f, d, sigma_px, rng):
    """把板上的点投到像素平面，加像素噪声。返回 (x, y, z, u, v)。"""
    out = []
    for x, y, z in pts:
        u = f * x / (d + z) + CX_TRUE + rng.gauss(0, sigma_px)
        v = f * y / (d + z) + CY_TRUE + rng.gauss(0, sigma_px)
        out.append((x, y, z, u, v))
    return out


def slope_u(pts):
    """u 对 x 的斜率，也就是 f/d（像素/米）。ce 项被减均值消掉。"""
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    mu = sum(p[3] for p in pts) / n
    den = sum((p[0] - mx) ** 2 for p in pts)
    return sum((p[0] - mx) * (p[3] - mu) for p in pts) / den if den else 0.0


def rms_shift(pts, f, d):
    """给定 f 和 d，只允许一个整体平移时，u 方向的重投影残差 RMS。"""
    n = len(pts)
    r = [p[3] - f * p[0] / (d + p[2]) for p in pts]
    m = sum(r) / n
    return math.sqrt(sum((v - m) ** 2 for v in r) / n)


def joint_rms(pts1, pts2, f, d0, L):
    """两视图联合残差：第二视图的距离被「已知位移 L」锁成 d0 + L。"""
    r1 = rms_shift(pts1, f, d0)
    r2 = rms_shift(pts2, f, d0 + L)
    return math.sqrt((r1 ** 2 + r2 ** 2) / 2)


def best_joint(pts1, pts2, f, L, lo=0.4, hi=2.4, steps=200):
    """固定 f，搜最优 d0。返回 (残差, d0)。"""
    best = (float("inf"), lo)
    for k in range(steps + 1):
        d = lo + (hi - lo) * k / steps
        r = joint_rms(pts1, pts2, f, d, L)
        if r < best[0]:
            best = (r, d)
    return best


# ---------------------------------------------------------------- [1] 退化

def part1_intrinsics(sigma_px=0.2, n=60):
    print("\n[1] 内参：残差 0.2 px 的时候，焦距可以差 25%")
    print("-" * 76)
    print("  针孔模型：u = f·x/(d+z) + cx。标定板是平的，z ≈ 0，于是 u ≈ (f/d)·x + cx。")
    print("  **只有 f/d 这个比值进得了图像。**单张标定板图永远解不出 f 和 d 各是多少。")
    print()

    rng = random.Random(3)
    pts = project(board_points(rng, n, 0.15, 0.11), F_TRUE, D_TRUE, sigma_px, rng)
    m = slope_u(pts)
    pts2 = project(board_points(rng, n, 0.15, 0.11), F_TRUE, D_TRUE + 0.300, sigma_px, rng)
    curve = []
    for i in range(41):
        f_cand = 400.0 + 10.0 * i
        curve.append((f_cand, rms_shift(pts, f_cand, f_cand / m),
                      best_joint(pts, pts2, f_cand, 0.300)[0]))

    print("  真值 f=%.0f px、d=%.3f m。现在假装不知道 f，试一堆候选值，" % (F_TRUE, D_TRUE))
    print("  「最优距离」由 f/d = 实测斜率 %.1f px/m 直接定出来：" % m)
    print("  %s%s%s%s" % (pad("候选焦距", 12), pad("最优距离", 14),
                          pad("f/d 比值", 14), "残差 RMS"))
    print("  " + "-" * 60)
    for f_cand in (480.0, 540.0, 600.0, 660.0, 750.0):
        d = f_cand / m
        res = rms_shift(pts, f_cand, d)
        print("  %s%s%s%s" % (pad("%.0f" % f_cand, 12), pad("%.3f m" % d, 14),
                              pad("%.1f" % (f_cand / d), 14), "%.3f px" % res))

    lo, hi = 480.0 / F_TRUE - 1, 750.0 / F_TRUE - 1
    rs = [rms_shift(pts, f, f / m) for f in (480.0, 540.0, 600.0, 660.0, 750.0)]
    print()
    print("  焦距从 480 到 750（%+.0f%% ~ %+.0f%%），残差只在 %.3f ~ %.3f px 之间动，"
          % (lo * 100, hi * 100, min(rs), max(rs)))
    print("  变化 %.1f%%。这不是「优化没收敛」，是**结构性退化**：整条 f/d = 常数 的曲线上，"
          % ((max(rs) / min(rs) - 1) * 100))
    print("  图像一模一样。优化器会随手停在某一点，然后报告一个很漂亮的残差。")
    print()
    print("  所以「重投影残差 0.2 px」这句结论，等价于「我找到了一条残差为 0.2 px 的曲线」，")
    print("  它没有说这条曲线上的哪一个点是对的。")
    print()
    print("  把残差画成「候选焦距 → 最小可达残差」的曲线，看得更清楚：")
    print("  %s%s%s" % (pad("候选焦距", 12), pad("单视图", 14), "加 300mm 位移"))
    print("  " + "-" * 52)
    for f_cand in (420.0, 480.0, 540.0, 600.0, 660.0, 720.0, 780.0):
        r1 = rms_shift(pts, f_cand, f_cand / m)
        r2 = best_joint(pts, pts2, f_cand, 0.300)[0]
        print("  %s%s%s" % (pad("%.0f" % f_cand, 12), pad("%.3f px" % r1, 14),
                            "%.3f px" % r2))
    print()
    print("  单视图那一列基本是常数；双视图那一列在 600 处有个谷。")
    print("  谷不是「优化变好了」，是**多了一个真约束**。")

    return {"m": m, "points": pts, "points2": pts2, "curve": curve}


# ---------------------------------------------------------------- [2] 已知长度

def part2_known_length(sigma_px=0.2, n=60, lengths=(0.05, 0.10, 0.30, 1.00)):
    print("\n[2] 打破退化的是一条**已知长度**——以及那条尺子本身有多准")
    print("-" * 76)
    print("  机器人正好擅长给长度：控制末端走 300 mm 是机械量，不依赖相机。")
    print("  拍第二张，第二视图的距离就被锁成 d0 + L，(f, d) 的两个自由度被两个方程卡住。")
    print()
    print("  但 L 要多大才够？两件事要分开看：**能检出**（20% 的错能被残差抓到）")
    print("  和**能估准**（把你的估计值信到几位）。")
    print()

    print("  %s%s%s%s" % (pad("已知位移 L", 14), pad("谷深", 12),
                          pad("σ_f", 12), "σ_f / f"))
    print("  " + "-" * 54)
    rows = []
    for L in lengths:
        rng2 = random.Random(17)
        p1 = project(board_points(rng2, n, 0.15, 0.11), F_TRUE, D_TRUE, sigma_px, rng2)
        p2 = project(board_points(rng2, n, 0.15, 0.11), F_TRUE, D_TRUE + L, sigma_px, rng2)
        r_true, _ = best_joint(p1, p2, F_TRUE, L)
        r_wrong, _ = best_joint(p1, p2, 480.0, L)
        depth = r_wrong - r_true
        fs = []
        for t in range(400):
            rng3 = random.Random(4000 + t * 31 + int(L * 1000))
            q1 = project(board_points(rng3, n, 0.15, 0.11), F_TRUE, D_TRUE, sigma_px, rng3)
            q2 = project(board_points(rng3, n, 0.15, 0.11), F_TRUE, D_TRUE + L, sigma_px, rng3)
            ma, mb = slope_u(q1), slope_u(q2)
            if abs(ma - mb) > 1e-9:
                fs.append(ma * mb * L / (ma - mb))
        mu = sum(fs) / len(fs)
        sd = math.sqrt(sum((x - mu) ** 2 for x in fs) / len(fs))
        rows.append((L, depth, sd, r_true))
        print("  %s%s%s%s" % (pad("%.0f mm" % (L * 1000), 14), pad("%.3f px" % depth, 12),
                              pad("%.1f px" % sd, 12), "%.2f%%" % (sd / F_TRUE * 100)))

    print()
    print("  两列讲的是两件事。最左边 50 mm 的谷深也有 %.2f px，是噪声本底 %.2f px 的 %.1f 倍——"
          % (rows[0][1], rows[0][3], rows[0][1] / rows[0][3]))
    print("  也就是说 **20%% 的焦距错误确实能被抓出来**。但同一行的 σ_f 是 %.0f px（%.1f%%），"
          % (rows[0][2], rows[0][2] / F_TRUE * 100))
    print("  你的估计值只能信到这个精度，仍然做不了精细抓取。")
    print("  抬到 300 mm，σ_f 降到 %.1f px（%.2f%%）；抬到 1 m，才到 %.1f px。"
          % (rows[2][2], rows[2][2] / F_TRUE * 100, rows[3][2]))
    print("  **能检出和能估准是两件事**，工程上要的是后者。")
    print()
    print("  为什么是「位移占工作距离的比例」说了算：谷深 ∝ 两视图之间的像差，而像差 ∝ L/d；")
    print("  同理 σ_f/f ≈ (d+L)/L 的放大倍数乘上斜率噪声。L/d 从 0.05 提到 0.3，")
    print("  放大倍数从 21 降到 4.3 —— 这才是 300 mm 这条经验值的来源。")
    print()
    print("  更麻烦的是尺子本身：估计量是 f_hat = m1·m2·L / (m1 − m2)，L 直接乘在外面，")
    print("  所以 **L 的相对误差 1:1 变成 f 的相对误差**：")
    print()

    rng4 = random.Random(19)
    p1 = project(board_points(rng4, n, 0.15, 0.11), F_TRUE, D_TRUE, sigma_px, rng4)
    p2 = project(board_points(rng4, n, 0.15, 0.11), F_TRUE, D_TRUE + 0.300, sigma_px, rng4)
    m1, m2 = slope_u(p1), slope_u(p2)
    f_clean = m1 * m2 * 0.300 / (m1 - m2)
    print("  %s%s%s" % (pad("300mm 尺子的误差", 20), pad("估出的焦距", 14), "偏差"))
    print("  " + "-" * 50)
    f_last = f_clean
    for err_mm in (0.0, 1.0, 3.0, 10.0):
        L_used = 0.300 + err_mm / 1000.0
        f_hat = m1 * m2 * L_used / (m1 - m2)
        f_last = f_hat
        print("  %s%s%s" % (pad("%+.0f mm" % err_mm, 20), pad("%.1f px" % f_hat, 14),
                            "%+.2f%%  (%+.1f px)" % ((f_hat / F_TRUE - 1) * 100,
                                                     f_hat - F_TRUE)))
    delta = 10.0 / 300.0
    print()
    print("  干净数据下估出 %.1f px（真值 600，误差 %.2f%%）—— 说明这个估计量本身是对的。" %
          (f_clean, (f_clean / F_TRUE - 1) * 100))
    print("  但尺子差 10 mm（%.1f%%），焦距就偏 %.1f%%（%+.1f px），比例几乎一模一样。"
          % (delta * 100, (f_last / F_TRUE - 1) * 100, f_last - F_TRUE))
    print("  **标定精度的天花板，往往不在相机，在你用来给尺度的那把尺子上。**")
    return {"rows": rows}


# ---------------------------------------------------------------- [3] 覆盖率

def part3_board_coverage(L=0.300, sigma_px=0.2, trials=400):
    print("\n[3] 「标定板要铺满画面」——这是 σ_f ∝ 1/(板宽·√点数) 的白话版")
    print("-" * 76)
    print("  估计量 f_hat = m1·m2·L/(m1−m2) 里只有两个斜率，而斜率的标准差是")
    print("      σ_m = σ_px / √(Σ(x−x̄)²) ≈ σ_px·√12 / (板宽·√点数)")
    print("  所以：板宽翻倍 → 误差减半；点数翻四倍 → 误差减半。")
    print()
    print("  实测（300mm 已知位移、0.2 px 像素噪声、每组 400 次）：")
    print()
    fov_w = 640.0 / F_TRUE          # 1m 处画面的物理宽度
    print("  %s%s%s%s%s" % (pad("板宽", 10), pad("占画面", 10), pad("点数", 8),
                            pad("σ_f", 12), "σ_f / f"))
    print("  " + "-" * 58)
    rows = []
    for bw in (0.10, 0.20, 0.40):
        for n in (40, 80, 160):
            fs = []
            for t in range(trials):
                rng = random.Random(1000 + t * 7 + int(bw * 1000) + n)
                p1 = project(board_points(rng, n, bw / 2, bw / 2 * 0.73), F_TRUE, D_TRUE, sigma_px, rng)
                p2 = project(board_points(rng, n, bw / 2, bw / 2 * 0.73), F_TRUE, D_TRUE + L, sigma_px, rng)
                m1, m2 = slope_u(p1), slope_u(p2)
                if abs(m1 - m2) > 1e-9:
                    fs.append(m1 * m2 * L / (m1 - m2))
            mu = sum(fs) / len(fs)
            sd = math.sqrt(sum((x - mu) ** 2 for x in fs) / len(fs))
            rows.append((bw, n, mu, sd))
            print("  %s%s%s%s%s" % (pad("%.0f mm" % (bw * 1000), 10),
                                    pad("%.0f%%" % (bw / fov_w * 100), 10),
                                    pad(str(n), 8), pad("%.1f px" % sd, 12),
                                    "%.2f%%" % (sd / F_TRUE * 100)))

    print()
    small = [r for r in rows if r[0] == 0.10 and r[1] == 40][0]
    big = [r for r in rows if r[0] == 0.40 and r[1] == 40][0]
    print("  板宽 100 → 400 mm（4 倍），σ_f 从 %.1f px 降到 %.1f px（%.1f 倍）。"
          % (small[3], big[3], small[3] / big[3]))
    dense = [r for r in rows if r[0] == 0.10 and r[1] == 160][0]
    print("  点数 40 → 160（4 倍），σ_f 从 %.1f px 降到 %.1f px（%.1f 倍）。"
          % (small[3], dense[3], small[3] / dense[3]))
    print("  一个是**线性**收益，一个是**平方根**收益。想让点数的收益赶上板宽，")
    print("  点要密 12 倍。所以「让标定板占满画面」不是习惯，是 1/板宽 这个因子；")
    print("  而「多拍几张」之所以有用，是因为它同时在放大有效板宽（不同姿态拼起来）。")
    return {"rows": rows, "fov_w": fov_w}


# ---------------------------------------------------------------- [4] 外参指纹

POSES = [(0.40, 0.00, 0.30, 0), (0.45, 0.10, 0.35, 20),
         (0.50, -0.10, 0.40, -20), (0.55, 0.05, 0.45, 35),
         (0.60, -0.05, 0.35, -35), (0.50, 0.15, 0.30, 10)]
P_WORLD = (0.55, 0.10, 0.02)


def extrinsic_residuals(t_err, r_err_deg):
    """
    误差传播，只保留一次项：

        真值   p_cam = R·p_end + t
        估计   p̂_end = R̂⁻¹(p_cam − t̂),   R̂ = R·ΔR,  t̂ = t + Δt
        代入   p̂_end = ΔR⁻¹·p_end − ΔR⁻¹·Δt

    所以残差 = |p̂_end − p_end|：
        · 纯平移：与 p_end 无关，恒为 |Δt|
        · 纯旋转：等于 2·|p_end|·sin(Δθ/2)，随力臂增长

    返回 [(力臂 |p_end|, 残差)]。
    """
    out = []
    dr = math.radians(r_err_deg)
    c, s = math.cos(dr), math.sin(dr)
    for tx, ty, tz, rz in POSES:
        cw, sw = math.cos(math.radians(rz)), math.sin(math.radians(rz))
        dx, dy = P_WORLD[0] - tx, P_WORLD[1] - ty
        px, py = cw * dx + sw * dy, -sw * dx + cw * dy
        ax, ay = px - t_err[0], py - t_err[1]
        ex = c * ax - s * ay - px
        ey = s * ax + c * ay - py
        out.append((math.hypot(px, py), math.hypot(ex, ey)))
    return out


def part4_extrinsics_signature():
    print("\n[4] 外参错了，残差会变大——但**残差的形状比大小更有诊断价值**")
    print("-" * 76)
    print("  场景：相机装在腕部，标定得到相机→末端的外参。之后用 6 个不同姿态")
    print("  去摸同一个固定点 P，看残差跟「力臂」（末端原点到 P 的距离）的关系。")
    print()
    print("  %s%s%s%s" % (pad("误差类型", 22), pad("残差均值", 12),
                          pad("残差范围", 18), "与力臂的关系"))
    print("  " + "-" * 76)
    cases = [("无误差", (0.0, 0.0, 0.0), 0.0),
             ("纯平移 10 mm", (0.010, 0.0, 0.0), 0.0),
             ("纯旋转 1°", (0.0, 0.0, 0.0), 1.0),
             ("纯旋转 3°", (0.0, 0.0, 0.0), 3.0),
             ("平移 10mm + 旋转 3°", (0.010, 0.0, 0.0), 3.0)]
    series = []
    for label, t_err, r_err in cases:
        rs = extrinsic_residuals(t_err, r_err)
        vals = [r for _, r in rs]
        mean = sum(vals) / len(vals)
        # 力臂与残差的相关系数：+1 说明残差完全由力臂解释
        arms = [a for a, _ in rs]
        ma = sum(arms) / len(arms)
        mr = mean
        num = sum((a - ma) * (r - mr) for a, r in rs)
        den = math.sqrt(sum((a - ma) ** 2 for a in arms) * sum((r - mr) ** 2 for r in vals))
        corr = num / den if den > 1e-12 else 0.0
        trend = "无关（恒定）" if abs(corr) < 0.2 else "线性增长 r=%.2f" % corr
        series.append((label, rs))
        print("  %s%s%s%s" % (pad(label, 22), pad("%.1f mm" % (mean * 1000), 12),
                              pad("%.1f ~ %.1f mm" % (min(vals) * 1000, max(vals) * 1000), 18),
                              trend))

    print()
    print("  指纹：")
    print("    · **平移误差** → 每个姿态残差几乎一样大，与力臂无关")
    print("    · **旋转误差** → 残差与力臂几乎完全线性相关（r≈1），范围被拉开")
    print("  所以看到「残差随位置变大」，先查旋转（装配角度、支架变形）；")
    print("  看到「所有位置偏一样多」，先查平移（安装位置）。")
    print("  这条判据比重新标一次更快，而且不用停机。")
    return {"series": series}


def part5_lever_arm():
    print("\n[5] 同一个 1°，在 10cm 处无害，在 2m 处是 3.5cm")
    print("-" * 76)
    print("  旋转误差被力臂放大：偏移 = d·tan(Δθ)。所以「标定精度 1°」这句话")
    print("  没有意义，**必须带上工作距离**。")
    print()
    print("  %s%s%s" % (pad("工作距离", 14), pad("1° 造成的偏移", 18), "2°"))
    print("  " + "-" * 54)
    for d in (0.1, 0.3, 0.5, 1.0, 2.0):
        print("  %s%s%s" % (pad("%.1f m" % d, 14),
                            pad("%.1f mm" % (d * math.tan(math.radians(1)) * 1000), 18),
                            "%.1f mm" % (d * math.tan(math.radians(2)) * 1000)))
    print()
    print("  表里最该记住的是 1.0 m 那一行：17.5 mm。抓一个 40 mm 的物体，")
    print("  17.5 mm 的偏置已经比物体半径大——**抓空或者撞上去，取决于符号。**")


# ---------------------------------------------------------------- [6] 融合

def part6_fusion():
    print("\n[6] 外参误差在单帧里看不见，只有融合之后才显形")
    print("-" * 76)
    print("  两个相机看同一块区域，各自出点云再拼起来。外参（相对位姿）有误差时，")
    print("  同一块表面会出现**双层**——错位量就是两个相机各自偏移之差。")
    print()
    print("  %s%s%s" % (pad("外参角度误差", 16), pad("1m 处错位", 14), "视觉表现"))
    print("  " + "-" * 66)
    for deg in (0.1, 0.5, 1.0, 2.0, 5.0):
        e = 1.0 * math.tan(math.radians(deg)) * 1000
        note = ("看不出来" if e < 2 else
                "边缘发虚，像失焦" if e < 8 else
                "明显重影" if e < 25 else "双层结构，不可用")
        print("  %s%s%s" % (pad("%.1f°" % deg, 16), pad("%.1f mm" % e, 14), note))
    print()
    print("  注意 0.5° 就是 8.7 mm，而「标定得还行」的行业标准经常就是 0.5° 量级。")
    print("  对抓取够用，对精细操作不够。而这一档误差在单个相机的图里完全看不出来：")
    print("  每张图都清晰、都对焦、都没有畸变——错的是两张图之间的**关系**。")
    print("  所以「看数据觉得没问题」不能作为标定合格的判据。")


# ---------------------------------------------------------------- [7] 可观测性

def part7_detectability(noise_m=0.002, n=2000):
    print("\n[7] 借重力检测漂移：能测 6 个自由度里的 2 个，不是 6 个")
    print("-" * 76)
    print("  办法：拟合桌面平面，把它和**重力**比较。重力来自 IMU，是独立于相机的")
    print("  参考系——两个传感器不会一起错。这是检测成立的基础。")
    print()
    print("  但要点在于：比较的一方是**标定时刻冻结下来的那个法向**，不是现算的。")
    print("  如果两边都从当次数据里现推，它们永远一致，什么也测不出来。")
    print()
    print("  把外参按照 6 个自由度分别改一下，看「拟合出的倾角」变成多少：")
    print()

    rng = random.Random(31)
    cases = [("绕 x 转 1.5°（pitch）", (1.5, 0.0, 0.0), (0.0, 0.0, 0.0)),
             ("绕 y 转 1.5°（roll）", (0.0, 1.5, 0.0), (0.0, 0.0, 0.0)),
             ("绕 z 转 5.0°（yaw）", (0.0, 0.0, 5.0), (0.0, 0.0, 0.0)),
             ("沿 x 平移 50 mm", (0.0, 0.0, 0.0), (0.050, 0.0, 0.0)),
             ("沿 y 平移 50 mm", (0.0, 0.0, 0.0), (0.0, 0.050, 0.0)),
             ("沿 z 平移 50 mm", (0.0, 0.0, 0.0), (0.0, 0.0, 0.050))]
    print("  %s%s%s" % (pad("引入的误差", 24), pad("拟合出的倾角", 18), "能不能发现"))
    print("  " + "-" * 66)
    rows = []
    for label, (ax, ay, az), (tx, ty, tz) in cases:
        pts = []
        for _ in range(n):
            x = rng.uniform(-0.4, 0.4)
            y = rng.uniform(-0.3, 0.3)
            px, py, pz = rot_xyz(x, y, 0.0, ax, ay, az)
            pts.append((px + tx, py + ty, pz + tz + rng.gauss(0, noise_m)))
        got = plane_angle_deg(pts)
        ok = got > 0.3
        rows.append((label, got, ok))
        print("  %s%s%s" % (pad(label, 24), pad("%.2f°" % got, 18),
                            "能" if ok else "**不能**（法向没动）"))

    print()
    print("  平板只能告诉你「法向歪了没有」。绕法向转（yaw）和平移都不改变法向，")
    print("  所以这 4 个自由度是**盲区**。要补上它们，参考物必须提供另一种信息：")
    print()
    print("    yaw      → 一个带方向的图案（标定板、AprilTag），或机器人的直线运动方向")
    print("  平移      → 一个已知尺寸的物体（多看点云的尺规），或机械臂碰到桌面的位置")
    print()
    print("  一句话：**参考物必须在你想测的那个自由度上「有变化」，否则它就是个常数。**")
    blind = [r for r in rows if not r[2]]
    print("  这一条比灵敏度重要得多——灵敏度可以靠加点数买，可观测性买不到。")
    return {"rows": rows, "blind": len(blind)}


# ---------------------------------------------------------------- [8] 灵敏度

def part8_sensitivity(noise_list=(0.0005, 0.002, 0.005), n_list=(16, 32, 64, 256, 1024),
                      trials=60):
    print("\n[8] 灵敏度不是瓶颈：几十个点就够检出 0.3°")
    print("-" * 76)
    print("  桌面真的水平（倾角 0），只有深度噪声。看「测出来的倾角」散布多大：")
    print("  散布就是这个方案的**分辨率下限**。")
    print()
    print("  %s%s" % (pad("点数", 10), "".join(pad("σ_d=%.1fmm" % (s * 1000), 14)
                                              for s in noise_list)))
    print("  " + "-" * (10 + 14 * len(noise_list)))
    rows = []
    for n in n_list:
        line = pad(str(n), 10)
        for s in noise_list:
            vals = []
            for t in range(trials):
                rng = random.Random(500 + t * 13 + n + int(s * 1e6))
                pts = [(rng.uniform(-0.4, 0.4), rng.uniform(-0.3, 0.3),
                        rng.gauss(0, s)) for _ in range(n)]
                vals.append(plane_angle_deg(pts))
            mu = sum(vals) / len(vals)
            sd = math.sqrt(sum((x - mu) ** 2 for x in vals) / len(vals))
            rows.append((n, s, sd))
            line += pad("%.3f°" % sd, 14)
        print("  " + line)

    first = [r for r in rows if r[1] == noise_list[1]]
    print()
    print("  两个规律：点数翻 4 倍，散布减半（√N）；噪声翻 4 倍，散布翻 4 倍（线性）。")
    print("  拿中间那一列（σ_d=2 mm，典型的结构光 / 双目深度噪声）按 3σ 判据反推：")
    for n, s, sd in first:
        if 3 * sd < 0.3:
            print("    %d 个点 → 3σ = %.3f°，第一次低于 0.3° 的阈值" % (n, 3 * sd))
            break
    last = first[-1]
    print("    %d 个点 → 3σ = %.3f°，比阈值小一个量级" % (last[0], 3 * last[2]))
    print()
    print("  所以分辨率从来不是这个方案的短板——「每天拍 10 秒」随便就是几万个点。")
    print("  真正的短板是：")
    print("    1. 第 7 节那 4 个测不出来的自由度")
    print("    2. 桌面本身是不是真的水平（地板、桌腿、垫片，都是 0.1° 量级的偏差）")
    print("  第二条决定了精度的下限：**你没法用一个 0.1° 不准的参考去测 0.01°。**")
    print("  这也是为什么这个方法的实用阈值是 0.3° 而不是 0.03°——不是测不出来，")
    print("  是「桌面到底有多水平」这件事本身就只到 0.1° 量级。")
    return {"rows": rows}


# ---------------------------------------------------------------- [9] 热胀

MATERIALS = [("铝", 23e-6), ("钢", 12e-6), ("ABS / PLA 打印件", 90e-6), ("碳纤维（沿纤维）", 2e-6)]

def part9_thermal(threshold_deg=0.3, delta_t=10.0):
    print("\n[9] 热胀到不了 0.3°，所以 0.3° 的漂移都是机械事件")
    print("-" * 76)
    print("  一个常见猜测是「温度变化让支架慢慢变形，标定就漂了」。算一下量级。")
    print()
    print("  支架两端有温差时，角变形 ≈ α·ΔT（长度在比值里消掉了，所以与支架多长无关）。")
    print("  两列：%.0f K 温差下胀出多少度，以及要胀出 %.1f° 需要多少温差。"
          % (delta_t, threshold_deg))
    print()
    print("  %s%s%s%s" % (pad("材料", 22), pad("α", 14),
                          pad("%.0f K 温差的角漂移" % delta_t, 22), "要 0.3° 需要"))
    print("  " + "-" * 72)
    need = []
    for name, alpha in MATERIALS:
        drift = math.degrees(alpha * delta_t)
        dt = math.radians(threshold_deg) / alpha
        need.append(dt)
        print("  %s%s%s%s" % (pad(name, 22), pad("%.0f ppm/K" % (alpha * 1e6), 14),
                              pad("%.3f°" % drift, 22), "%.0f K" % dt))
    print()
    print("  最「能胀」的 3D 打印件也要 %.0f K 的温差才胀得出 0.3°，铝要 %.0f K。" %
          (need[2], need[0]))
    print("  而 %.0f K 的温差下，最坏的那个材料也只漂 %.3f°（铝 %.3f°）——"
          % (delta_t, math.degrees(MATERIALS[2][1] * delta_t),
             math.degrees(MATERIALS[0][1] * delta_t)))
    print("  比 0.3° 的阈值小一个量级。")
    print()
    print("  换句话说：**这个量级的漂移不可能是热胀，只能是机械事件**——碰撞、")
    print("  螺丝松动、卡扣没卡到底。机械事件是阶跃，不是斜坡，这决定了检测节奏（下一节）。")
    print()
    print("  顺带算一下 %.0f K 温差下的平移量（支架 100 mm）：" % delta_t)
    for name, alpha in MATERIALS[:2]:
        disp = 0.100 * alpha * delta_t * 1e6
        print("    %s：%.0f µm —— 在 1 m 处相当于 %.3f mm 的视差"
              % (pad(name, 22), disp, disp / 1000.0))
    print("    这个量级对抓取完全无所谓——热胀在平移上也不是威胁。")


# ---------------------------------------------------------------- [10] 节奏

def days_to_detect(rate, noise_deg=0.02, k_max=60):
    """从第 1 天起做累积平均，按 3σ 判据，返回第一次确认漂移所需的天数。

    累积平均把噪声压到 σ/√k，同时把漂移聚成 rate·k 的信号。
    两者相向而行，交点就是「能确认」的那一天——它远早于漂移长到阈值。
    """
    for k in range(1, k_max + 1):
        signal = rate * (k + 1) / 2.0          # 前 k 天漂移的均值
        if signal > 3.0 * noise_deg / math.sqrt(k):
            return k
    return None


def part10_schedule(noise_deg=0.02, threshold=0.3, rates=(0.005, 0.010, 0.020)):
    print("\n[10] 所以检测节奏是「每天 10 秒」，而不是「每个月重标一次」")
    print("-" * 76)
    print("  两类事件，两种检法：")
    print()
    print("  **阶跃（碰撞）**：倾角从 0.01° 一步跳到 1.5°，信噪比 %.0f。一次测量就报。"
          % (1.5 / noise_deg))
    print("  代价只是「那天有没有做这件事」。所以频率的意义不是灵敏度，是**延迟**：")
    print("  每天查一次，最坏情况是 1 天的数据废掉；每月查一次，最坏是 1 个月。")
    print()
    print("  **斜坡（松动蠕变）**：每天 %.3f° 这种量级，单次测量完全看不出来" % rates[1])
    print("  （信号 0.01° 小于噪声 0.02°）。但累积均值可以把噪声压下去：")
    print()

    print("  %s%s%s%s" % (pad("每天漂移", 14), pad("检出天数", 12),
                          pad("检出时累积", 16), "同期单次测量能不能看到"))
    print("  " + "-" * 74)
    for rate in rates:
        k = days_to_detect(rate, noise_deg)
        cum = rate * k
        single = ("看不到（%.3f° < 3σ=%.2f°）" % (rate, 3 * noise_deg)
                  if rate < 3 * noise_deg else "能看到")
        print("  %s%s%s%s" % (pad("%.3f°/天" % rate, 14), pad("%d 天" % k, 12),
                              pad("%.3f°" % cum, 16), single))

    print()
    k = days_to_detect(rates[1], noise_deg)
    print("  每天 %.3f° 的蠕变，%.0f 天就能确认（累计 %.2f°），而阈值是 %.1f°——"
          % (rates[1], k, rates[1] * k, threshold))
    print("  也就是说**它在漂到阈值的 %.0f%% 时就被发现了**。"
          % (rates[1] * k / threshold * 100))
    print()
    print("  这就是「每天 10 秒」的全部理由：不是为了测得准（第 8 节说了，")
    print("  灵敏度早就够了），是为了**把发现时刻提前**。数据平台的成本在这里是")
    print("  不对称的：一次 30 秒的检查，挡住的是从漂移发生到被发现之间的全部数据。")
    print()
    print("  也顺便说明「重标一次」为什么不是解法：重标修的是当前状态，")
    print("  修不了「你不知道它已经漂了三个月」这件事。**可检测性比精度重要。**")


# ---------------------------------------------------------------- [11] 结论

def part11_conclusion():
    print("\n[11] 一张决策表")
    print("-" * 76)
    rows = [("只有内参、无已知尺度", "不可解", "f/d 退化，残差再小也不唯一"),
            ("内参 + 50 mm 位移", "σ_f ≈ 1.6%", "能检出 20% 的错，但定不准"),
            ("内参 + 300 mm 位移", "σ_f ≈ 0.3%", "尺子本身的误差 1:1 进来"),
            ("板宽 100 mm（画面 9%）", "σ_f ≈ 1.0%", "小板 = 短基线，斜率测不准"),
            ("板宽 400 mm（画面 38%）", "σ_f ≈ 0.3%", "做大板是线性收益，比加点便宜"),
            ("外参精度 0.5°", "1 m 处 8.7 mm", "对抓取够，对精细操作不够"),
            ("外参精度 1°", "1 m 处 17.5 mm", "已大于 40 mm 物体的半径"),
            ("每天 10 秒平面检查", "检出 pitch/roll 漂移", "不覆盖 yaw 和平移"),
            ("yaw / 平移漂移", "平面检查测不出", "要靶标、已知尺寸或机械触点")]
    print(head(("做法", 24), ("能给出什么", 22), ("代价 / 局限", 0)))
    print("  " + "-" * 76)
    for a, b, c in rows:
        print("  %s%s%s" % (pad(a, 24), pad(b, 22), c))
    print("\n  一句话：标定的精度写在你报告的残差里，标定的可信度写在你能不能")
    print("  发现它错了。前者是数学问题，后者是工程问题，而工程问题才是掉数据的地方。")


if __name__ == "__main__":
    print("=" * 78)
    print("实验 3：标定的退化、指纹与可检测性")
    print("=" * 78)
    part1_intrinsics()
    part2_known_length()
    part3_board_coverage()
    part4_extrinsics_signature()
    part5_lever_arm()
    part6_fusion()
    part7_detectability()
    part8_sensitivity()
    part9_thermal()
    part10_schedule()
    part11_conclusion()
    print("\n" + "=" * 78)

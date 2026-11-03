#!/usr/bin/env python3
"""
实验 2：针孔相机模型，以及四个真实的数据坑
============================================

相机输出的是"像素 + 深度"，不是"米"。要把像素变成机器人能用的三维点，
必须过一次内外参。这个脚本量化四件事：

  1. 一个像素在真实世界里有多大 —— 随距离线性变大
  2. 外参装歪 1 度，误差有多大 —— 也随距离线性变大
  3. 深度图的量化台阶在什么条件下才看得见
  4. 卷帘快门怎么把静止的墙拍歪

最后算一遍数据量，回答"为什么机器人数据集都用视频编码存图像"。

运行：
    python examples/02_sensor_data/02_camera_model.py

只依赖 Python 标准库。
"""

import math
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# RealSense D435 一类深度相机的典型内参
W, H = 640, 480
FX = FY = 525.0          # 焦距（像素）
CX, CY = 320.0, 240.0    # 主点（通常接近图像中心）


# ---------------------------------------------------------------- 相机模型

def project(p):
    """世界(相机系)三维点 -> 像素坐标。"""
    x, y, z = p
    u = FX * x / z + CX
    v = FY * y / z + CY
    return u, v


def backproject(u, v, z):
    """像素 + 深度 -> 相机系三维点。"""
    x = (u - CX) * z / FX
    y = (v - CY) * z / FY
    return x, y, z


def gsd(z):
    """ground sampling distance：depth=z 处，一个像素对应多少米。"""
    return z / FX


def fov_deg():
    """水平/垂直视场角。"""
    return (2 * math.degrees(math.atan(W / 2 / FX)),
            2 * math.degrees(math.atan(H / 2 / FY)))


# ---------------------------------------------------------------- 各部分

def part1_pixel_size():
    print("\n[1] 一个像素有多大")
    print("-" * 70)
    hfov, vfov = fov_deg()
    print(f"  内参 fx=fy={FX:.0f}px  主点=({CX:.0f}, {CY:.0f})")
    print(f"  视场角：水平 {hfov:.1f}°  垂直 {vfov:.1f}°")
    print()
    print(f"  {'距离 z':>8}{'1 像素 =':>12}{'整幅画面宽':>14}{'相对 1m 处':>12}")
    print("  " + "-" * 46)
    base = gsd(1.0)
    for z in (0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0):
        g = gsd(z)
        print(f"  {z:>6.1f}m{g * 1000:>10.2f}mm{W * g:>12.3f}m{g / base:>11.1f}x")

    print("\n  关键：像素是角度单位，不是长度单位。")
    print(f"  同一个 5 像素的定位误差，在 0.5m 处是 {gsd(0.5) * 5 * 1000:.1f}mm，"
          f"在 3m 处是 {gsd(3.0) * 5 * 1000:.1f}mm。")
    print("  抓取任务都在 0.3–1m，所以 1 像素 = 1–2mm；导航任务在 3–10m，1 像素 = 6–19mm。")
    print("  同一个模型在两个距离上表现差 10 倍，不是模型的问题。")


def part2_roundtrip():
    print("\n[2] 投影 -> 反投影，能不能还原")
    print("-" * 70)
    pts = [(0.0, 0.0, 1.0), (0.1, -0.05, 0.5), (-0.2, 0.15, 2.0)]
    print(f"  {'原始点':>22}{'像素':>16}{'还原点':>26}{'误差':>10}")
    worst = 0.0
    for p in pts:
        u, v = project(p)
        q = backproject(u, v, p[2])
        err = math.dist(p, q)
        worst = max(worst, err)
        print(f"  ({p[0]:>6.3f},{p[1]:>6.3f},{p[2]:>5.2f})"
              f"  ({u:>6.1f},{v:>6.1f})"
              f"   ({q[0]:>7.4f},{q[1]:>7.4f},{q[2]:>5.2f}){err:>10.1e}")

    print(f"\n  最大误差 {worst:.1e} —— 数学上无损。（这里恰好整除得到 0，一般点位在 1e-16 量级）")
    print("  但注意：反投影需要 z。而 z 是传感器测的，不是算出来的。")
    print("  所以像素 -> 三维点的精度，完全由深度精度决定。这一句是后面所有坑的根源。")


def part3_extrinsics():
    print("\n[3] 相机装歪 1 度会怎样")
    print("-" * 70)
    print("  两类外参误差，后果完全不同：")
    print()

    # 绕光轴滚转：只是整幅图旋转，可以用图像处理纠正
    print("  (a) 绕光轴滚转（roll）：整幅图旋转，深度不变")
    print("      -> 可以用标定矩阵事后纠正，危害小")

    # 绕 x/y 俯仰：点的横向位置随深度偏移
    print("\n  (b) 绕 x 或 y 轴俯仰（pitch/yaw）：\"看到的位置\"整体平移，且随深度放大")
    print(f"  {'俯仰角':>8}{'0.5m 处':>12}{'1m 处':>12}{'3m 处':>12}{'5m 处':>12}")
    print("  " + "-" * 56)
    for deg in (0.5, 1.0, 2.0, 5.0):
        rad = math.radians(deg)
        row = "".join(f"{z * math.tan(rad) * 1000:>10.1f}mm" for z in (0.5, 1.0, 3.0, 5.0))
        print(f"  {deg:>7.1f}°{row}")

    rad1 = math.radians(1.0)
    z = 1.0
    shift_m = z * math.tan(rad1)
    print(f"\n  1° 的俯仰，在 1m 处让所有点整体偏移 {shift_m * 1000:.1f}mm，"
          f"折合 {shift_m / gsd(z):.1f} 像素。")
    print(f"  同一误差在 3m 处是 {3.0 * math.tan(rad1) * 1000:.1f}mm —— "
          f"放大 {3.0 / 1.0:.0f} 倍。")
    print("  在抓取任务里 17mm 意味着：以为手在杯子上方，实际偏了半个杯口。")

    print("\n  而且这个误差**无法从数据本身发现**：")
    print("  - 图像看起来完全正常（整幅图平移 17mm，没有畸变）")
    print("  - 深度值完全正常")
    print("  - 只有把点云和机器人基座对齐时，才会发现系统性错位")
    print("  这就是为什么标定文件必须作为数据集的元数据，而不是相机的一份本地配置。")


def part4_depth_quantization():
    print("\n[4] 深度量化：台阶什么时候看得见")
    print("-" * 70)
    print("  深度图通常存成 16-bit 整数，单位 mm（量程 0–65m 够用）。")
    print("  量化台阶 = 1mm。但台阶在图像上是否可见，取决于一个像素跨了多少深度：")
    print("      dz/dpx = tan(倾角) * GSD(z)")
    print()
    print(f"  {'平面倾角':>8}{'1m 处 dz/px':>14}{'3m 处 dz/px':>14}{'1m处几像素一台阶':>20}")
    print("  " + "-" * 58)
    for deg in (2, 5, 10, 30, 60, 85):
        t = math.tan(math.radians(deg))
        d1 = t * gsd(1.0)
        d3 = t * gsd(3.0)
        step_px = 0.001 / d1 if d1 > 0 else float("inf")
        note = "无台阶" if step_px < 1 else f"{step_px:.1f}"
        print(f"  {deg:>7.0f}°{d1 * 1000:>11.2f}mm{d3 * 1000:>11.2f}mm{note:>20}")

    print("\n  反直觉的地方：**越平的表面，台阶越明显**。")
    print("  对着墙（倾角接近 0°）拍，墙上会出现一圈圈等高线状的台阶；")
    print("  对着 30° 以上的斜面拍，反而很平滑。因为斜面一个像素跨越的深度")
    print("  本来就超过 1mm，量化误差被信号本身的梯度盖住了。")
    print()
    print("  工程后果：深度图不能直接求导算法向量。")
    print("  平滑墙面上相邻像素的深度差被量化成 0 或 1mm，梯度是纯噪声。")
    print("  这是为什么点云处理第一步总是要做离群点滤波。")


def part5_rolling_shutter():
    print("\n[5] 卷帘快门：静止的墙为什么会歪")
    print("-" * 70)
    fps = 30.0
    t_frame = 1.0 / fps
    t_line = t_frame / H
    print(f"  {fps:.0f} fps，一帧 {t_frame * 1000:.1f}ms，逐行读出，每行 {t_line * 1e6:.1f}us")
    print(f"  第 1 行和第 {H} 行相差 {t_frame * 1000:.1f}ms")
    print()
    print(f"  {'横向运动速度':>14}{'首末行时间差内的位移':>22}{'1m 处折合像素':>16}")
    print("  " + "-" * 54)
    for v in (0.1, 0.5, 1.0, 2.0, 5.0):
        d = v * t_frame
        px = d / gsd(1.0)
        print(f"  {v:>12.1f}m/s{d * 1000:>19.1f}mm{px:>16.1f}px")

    print("\n  1 m/s 是机械臂典型的末端速度。这时一根竖直的杆子会被拍成")
    print(f"  倾斜约 {1.0 * t_frame / gsd(1.0):.0f} 像素 —— 在 640 宽的图里肉眼看得出。")
    print()
    print("  为什么这对数据平台重要：")
    print("  - 卷帘畸变让'同一时刻'这个概念失效。一行一行读出来的，每行时刻都不同。")
    print("  - 快速运动的数据（抓取瞬间、抛接）畸变最严重，而这些恰是最有价值的样本。")
    print("  - 缓解手段：全局快门相机（贵）、提高帧率（减小读出时间）、")
    print("    或用 IMU 数据做逐行去畸变（主流做法，但要求 IMU 和相机硬件同步）。")


def part6_data_volume():
    print("\n[6] 数据量：为什么机器人数据集都用视频编码")
    print("-" * 70)
    fps = 30.0
    n = int(30 * fps)        # 30 秒 episode 的帧数
    raw_rgb = W * H * 3
    raw_d = W * H * 2

    print(f"  一段 30 秒 episode，{n} 帧，{W}x{H}")
    print()
    print(f"  {'格式':>26}{'每帧':>12}{'每秒':>14}{'30 秒总计':>14}")
    print("  " + "-" * 66)
    rows = [
        ("RGB 原始 (RGB888)", raw_rgb, fps * raw_rgb),
        ("RGB 逐帧 JPEG (q=90)", raw_rgb / 15, fps * raw_rgb / 15),
        ("RGB H.264 (1.5 Mbps)", 1.5e6 / 8 / fps, 1.5e6 / 8),
        ("深度 原始 (16-bit)", raw_d, fps * raw_d),
        ("深度 PNG 无损", raw_d / 2.2, fps * raw_d / 2.2),
    ]
    for name, per_frame, per_sec in rows:
        total = per_sec * 30
        print(f"  {name:>24}{per_frame / 1024:>9.0f}KB{per_sec / 1e6:>11.2f}MB/s"
              f"{total / 1e6:>11.1f}MB")

    print(f"\n  原始 RGB 与 H.264 相差 {raw_rgb * fps / (1.5e6 / 8):.0f} 倍。")
    print("  这不是优化技巧，是能不能落地的分界线：")
    print(f"  100 小时纯 RGB 原始数据 = {100 * 3600 * raw_rgb * fps / 1e12:.1f} TB，")
    print(f"  用 H.264 是 {100 * 3600 * 1.5e6 / 8 / 1e12:.2f} TB。")
    print()
    print("  代价：视频编码是有损的，而且帧间编码让'随机访问某一帧'变慢。")
    print("  所以 LeRobot / Open X-Embodiment 的选择是：图像走 H.264 分块编码，")
    print("  低维数据（状态、动作）走 Parquet。两种格式各管一段。")


def main():
    print("=" * 70)
    print("实验 2：针孔相机模型与四个数据坑")
    print("=" * 70)
    part1_pixel_size()
    part2_roundtrip()
    part3_extrinsics()
    part4_depth_quantization()
    part5_rolling_shutter()
    part6_data_volume()

    print("\n" + "=" * 70)
    print("结论（数据平台视角）：")
    print("  1. 内参必须和图像一起存。没有 K 的图，等于没有坐标系的点。")
    print("  2. 外参误差随距离线性放大，且无法从数据本身自查 —— 必须靠标定流程保证。")
    print("  3. 深度量化台阶让'平滑表面'变成噪声源，点云入库前必须滤波。")
    print("  4. 卷帘快门让'每帧一个时间戳'这个假设失效，高速数据要单独处理。")
    print("  5. 图像必须走视频编码，否则存储成本高一个数量级。")
    print("=" * 70)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
实验 2：一台 RGB-D 相机输出的到底是什么，以及五个真实的数据坑
================================================================

相机输出的是"像素颜色 + 像素距离"，不是"米"，也不是"三维点"。
从这两张二维数组到一个机器人能用的点云，中间隔着内参、畸变、深度测量原理、
外参、时间五道换算。每一步错了，数据都不会报错，只会静默地偏。

这个脚本量化十件事：

  1. 一帧数据到底是几个字节，位深意味着什么
  2. 一个像素在真实世界里有多大 —— 随距离线性变大
  3. 投影 -> 反投影能不能还原
  4. 分辨率改了，内参怎么跟着变
  5. 镜头畸变有多大 —— 为什么直线拍出来是弯的
  6. 深度是怎么从视差算出来的，精度为什么随距离平方变差
  7. 深度量化台阶什么时候看得见，它怎么毁掉法向量
  8. 相机装歪 1 度，误差有多大
  9. 卷帘快门怎么把静止的墙拍歪
 10. 数据量 —— 为什么机器人数据集都用视频编码存图像

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
BASELINE = 0.05          # 双目基线 50mm（左右两个红外相机之间的间距）


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


def fov_for(fx, width):
    """给定焦距和画幅宽度，算水平视场角。"""
    return 2 * math.degrees(math.atan(width / 2 / fx))


def distort(u, v, k1, k2=0.0):
    """针孔坐标 -> 带畸变的实际像素坐标（Brown-Conrady 径向项）。"""
    xn = (u - CX) / FX
    yn = (v - CY) / FY
    r2 = xn * xn + yn * yn
    factor = 1 + k1 * r2 + k2 * r2 * r2
    return xn * factor * FX + CX, yn * factor * FY + CY


def undistort(u, v, k1, k2=0.0, iters=30):
    """畸变像素 -> 理想针孔像素。畸变公式没有解析逆，用不动点迭代。"""
    xn = (u - CX) / FX
    yn = (v - CY) / FY
    for _ in range(iters):
        r2 = xn * xn + yn * yn
        factor = 1 + k1 * r2 + k2 * r2 * r2
        xn = (u - CX) / FX / factor
        yn = (v - CY) / FY / factor
    return xn * FX + CX, yn * FY + CY


def stereo_depth_error(z, sigma_d=0.1, fx=FX, baseline=BASELINE):
    """双目测深度的标准差：sigma_z = z^2 / (fx * B) * sigma_d。"""
    return z * z / (fx * baseline) * sigma_d


# ---------------------------------------------------------------- 各部分

def part1_frame_bytes():
    print("\n[1] 一帧数据到底是几个字节")
    print("-" * 70)
    rgb = W * H * 3
    depth = W * H * 2
    print(f"  彩色图：{W} x {H} x 3 通道，uint8（每通道 0-255）")
    print(f"          = {rgb} 字节 = {rgb / 1024:.0f} KB")
    print(f"  深度图：{W} x {H}，uint16（每个像素 0-65535）")
    print(f"          = {depth} 字节 = {depth / 1024:.0f} KB")
    print(f"  一帧合计 {(rgb + depth) / 1024:.0f} KB，30 fps 就是 "
          f"{(rgb + depth) * 30 / 1e6:.1f} MB/s")
    print()
    print("  uint8 只有 256 级。这不是'颜色不准'，是动态范围只有 48 dB 左右。")
    print("  同一张图里既有阴影里的黑猫又有窗户，8 bit 装不下 —— 要么猫糊成一片，")
    print("  要么窗户外过曝。HDR 传感器就是为了这个，代价是每通道 2 字节起。")
    print()
    print("  深度图的位深不是'精度'：uint16 存的单位是毫米，所以量程 0-65.5m，")
    print("  而量化台阶固定 1mm —— 在 0.3m 处是 0.33% 的相对误差，5m 处是 0.02%。")
    print("  所以量化从来不是深度的瓶颈，真正的误差来自测量过程本身（见 [6]）。")


def part2_pixel_size():
    print("\n[2] 一个像素有多大")
    print("-" * 70)
    hfov, vfov = fov_deg()
    print(f"  内参 fx=fy={FX:.0f}px  主点=({CX:.0f}, {CY:.0f})")
    print(f"  视场角：水平 {hfov:.1f}°  垂直 {vfov:.1f}°")
    print()
    print(f"  {'距离 z':>8}{'1 像素 =':>12}{'整幅画面宽':>14}{'每平米像素':>14}")
    print("  " + "-" * 48)
    base = gsd(1.0)
    for z in (0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0):
        g = gsd(z)
        print(f"  {z:>6.1f}m{g * 1000:>10.2f}mm{W * g:>12.3f}m{1 / (g * g):>13.0f}")

    print("\n  关键：像素是角度单位，不是长度单位。")
    print(f"  同一个 5 像素的定位误差，在 0.5m 处是 {gsd(0.5) * 5 * 1000:.1f}mm，"
          f"在 3m 处是 {gsd(3.0) * 5 * 1000:.1f}mm。")
    print(f"  距离差 10 倍（0.3m vs 3m），同一个像素代表的面积差 "
          f"{(gsd(3.0) / gsd(0.3)) ** 2:.0f} 倍。")
    print()
    print("  换成'东西在图上占多少像素'更直观。一个 8 厘米高的杯子：")
    for z in (0.5, 1.0, 2.0):
        print(f"      {z:.1f}m 处 = {FX * 0.08 / z:.0f} 像素高")
    print("  同样是 640x480 的图，近处的杯子占满半个画面，远处只是一个小点。")
    print("  所以'分辨率'不是均匀的：画面中心的一平方米用掉几十万像素，")
    print("  画面边缘同样大的一块只有几万。抓取任务都在 0.3–1m，")
    print("  1 像素 = 1–2mm；导航任务在 3–10m，1 像素 = 6–19mm。")
    print("  同一个模型在两个距离上表现差 10 倍，不是模型的问题。")


def part3_roundtrip():
    print("\n[3] 投影 -> 反投影，能不能还原")
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


def part4_resolution_scale():
    print("\n[4] 分辨率改了，内参怎么变")
    print("-" * 70)
    print("  很多人以为'同一台相机换个分辨率，内参不变'。不对 —— 内参是按像素算的。")
    print()
    print(f"  {'分辨率':>12}{'fx':>10}{'cx':>10}{'水平视场':>10}{'相对 640 宽':>12}")
    print("  " + "-" * 54)
    for scale, label in ((1.0, "640x480"), (0.5, "320x240"), (0.25, "160x120")):
        fx = FX * scale
        cx = CX * scale
        print(f"  {label:>12}{fx:>10.1f}{cx:>10.1f}{fov_for(fx, W * scale):>9.1f}°"
              f"{1 / scale:>11.0f}x")
    print()
    print("  规律：fx、cx、cy 全部按同一个比例缩放，视场角不变。")
    print("  所以分辨率降一半，1 像素代表的物理尺寸翻倍，定位精度掉一半。")
    print("  这也是为什么'把图缩到 224x224 喂给模型'是有代价的：")
    print(f"  224 宽的图上 1 像素 = {640 / 224:.1f} 个原始像素 = "
          f"1m 处 {gsd(1.0) * 640 / 224 * 1000:.1f}mm。")


def part5_distortion():
    print("\n[5] 镜头畸变：为什么直线拍出来是弯的")
    print("-" * 70)
    print("  真实的镜头不是针孔。光线穿过镜片边缘时折得更多，于是：")
    print("      r_distorted = r * (1 + k1*r^2 + k2*r^4)      r 是归一化像距")
    print("  k1 < 0 是桶形（广角常见），k1 > 0 是枕形。r 越大偏得越多，")
    print("  所以画面中心几乎不变，四个角偏得最厉害。")
    print()
    k1 = -0.3
    print(f"  取一个典型广角镜头的 k1 = {k1}（D435 的彩色流出厂已校正，这里用未校正的算）")
    print()
    print(f"  {'位置':>12}{'归一化半径':>12}{'畸变后偏移':>12}{'1m 处折合':>12}{'0.3m 处':>10}")
    print("  " + "-" * 60)
    for u, v, name in ((320, 240, "画面中心"), (320, 480, "下边中点"),
                       (640, 240, "右边中点"), (640, 480, "右下角"),
                       (0, 0, "左上角")):
        xn, yn = (u - CX) / FX, (v - CY) / FY
        r = math.hypot(xn, yn)
        du, dv = distort(u, v, k1)
        shift = math.hypot(du - u, dv - v)
        print(f"  {name:>12}{r:>12.2f}{shift:>10.1f}px"
              f"{shift * gsd(1.0) * 1000:>10.0f}mm{shift * gsd(0.3) * 1000:>8.0f}mm")

    print()
    print("  角落偏 70 像素听起来还行？换算到 1 米外就是 13 厘米 ——")
    print("  抓一个 8 厘米的杯子，你会抓在空气上。而且这张图看起来毫无异常，")
    print("  因为所有直线依然'看起来是直线'（整个画面被平滑地拉伸了）。")
    print()
    print("  反解：畸变公式没有解析逆，工程上用不动点迭代逼近。验证往返：")
    worst = 0.0
    for u, v in ((640, 480), (100, 60), (500, 400)):
        du, dv = distort(u, v, k1)
        uu, vv = undistort(du, dv, k1)
        worst = max(worst, math.hypot(uu - u, vv - v))
    print(f"      去畸变 -> 再畸变 -> 再回来，最大残差 {worst:.2e} 像素。可逆。")
    print("  但前提是 k1 对。k1 错了或者没存，这个逆就永远算不出来。")
    print("  结论：内参和畸变系数必须成对存，只存 fx/fy/cx/cy 是不够的。")


def part6_disparity():
    print("\n[6] 深度从哪来：视差 -> 距离，以及为什么精度随距离平方变差")
    print("-" * 70)
    print("  双目相机不是'发射什么'，而是'看同一个点，左右眼位置差多少'。")
    print("  左右两个相机相隔 B（基线），同一个点在两张图上的横坐标差 d（视差）：")
    print()
    print("      Z = fx * B / d        视差越大 -> 越近；视差趋于 0 -> 无穷远")
    print()
    print(f"  基线 B = {BASELINE * 1000:.0f}mm，fx = {FX:.0f}px，于是：")
    print()
    print(f"  {'距离':>8}{'视差':>10}{'视差错 1px -> 深度差':>22}")
    print("  " + "-" * 46)
    for z in (0.3, 0.5, 1.0, 2.0, 3.0, 5.0):
        d = FX * BASELINE / z
        dz = stereo_depth_error(z, sigma_d=1.0)
        print(f"  {z:>6.1f}m{d:>8.1f}px{dz * 1000:>18.0f}mm")

    print("\n  注意右边那一列长得多快：距离翻倍，深度误差变 4 倍（平方关系）。")
    print("  原因在公式里：sigma_z 正比于 z^2 / d，而 d 本身正比于 1/z，")
    print("  所以 sigma_z 正比于 z^2。这是几何决定的，不是这个型号不行。")
    print()
    print("  换成更实际的亚像素匹配精度（0.1 像素）：")
    print(f"  {'距离':>8}{'0.1px 匹配':>14}{'对比：16 线激光雷达':>22}")
    print("  " + "-" * 46)
    lidar = {1.0: 10, 3.0: 20, 5.0: 40, 8.0: 60}
    for z in (1.0, 3.0, 5.0, 8.0):
        print(f"  {z:>6.1f}m{stereo_depth_error(z, 0.1) * 1000:>12.0f}mm"
              f"   约 ±{lidar[z]}mm（型号相关，此处为量级参考）")
    print()
    # 求两条误差曲线的交点，别猜
    def lidar_at(z):
        pts = [(1.0, 10.0), (3.0, 20.0), (5.0, 40.0), (8.0, 60.0)]
        for i in range(len(pts) - 1):
            z0, e0 = pts[i]
            z1, e1 = pts[i + 1]
            if z0 <= z <= z1:
                return e0 + (e1 - e0) * (z - z0) / (z1 - z0)
        return pts[-1][1]

    cross = None
    z = 0.5
    while z < 8:
        if stereo_depth_error(z, 0.1) * 1000 >= lidar_at(z):
            cross = z
            break
        z += 0.01
    print(f"  两条误差曲线的交点大约在 {cross:.1f}m：{cross:.1f}m 以内深度相机更细，")
    print(f"  以外激光雷达更稳。这就是为什么抓取用深度相机、建图用激光雷达 ——")
    print("  不是谁更先进，是它们的误差随距离增长的方式不同（平方 vs 近似线性）。")
    print()
    print("  ToF（飞行时间）走的是另一条路：直接测光往返时间，")
    print("  sigma_z 大致正比于 z（不是 z^2），所以远处反而比双目好。")
    print("  但 ToF 有另一类问题：多路径反射、阳光干扰、近处饱和。")


def part7_depth_quantization():
    print("\n[7] 深度量化：台阶什么时候看得见，以及它怎么毁掉法向量")
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
    print("  工程后果：深度图不能直接求导算法向量。算一下 1mm 的量化台阶")
    print("  在两个相邻像素之间看起来是多大的倾角：")
    print()
    print(f"  {'距离':>8}{'相邻像素间距':>14}{'1mm 台阶 = 倾角':>18}")
    print("  " + "-" * 42)
    for z in (0.5, 1.0, 2.0, 3.0):
        g = gsd(z)
        deg = math.degrees(math.atan(0.001 / g))
        print(f"  {z:>6.1f}m{g * 1000:>12.2f}mm{deg:>16.1f}°")
    print()
    print("  1 米处，一堵完全平的墙上，相邻像素深度差只要量化成 1mm，")
    print(f"  求出来的法向量就歪了 {math.degrees(math.atan(0.001 / gsd(1.0))):.0f}°。")
    print("  这就是为什么点云处理第一步总是做平滑/离群点滤波，而不是直接算法向量。")


def part8_extrinsics():
    print("\n[8] 相机装歪 1 度会怎样")
    print("-" * 70)
    print("  两类外参误差，后果完全不同：")
    print()

    print("  (a) 绕光轴滚转（roll）：整幅图旋转，深度不变")
    print("      -> 整幅图只是转了个角度，可以用一个标定矩阵事后纠正，危害小")

    print("\n  (b) 绕 x 或 y 轴俯仰（pitch/yaw）：\"看到的位置\"整体平移，且随深度线性放大")
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


def part9_rolling_shutter():
    print("\n[9] 卷帘快门：静止的墙为什么会歪")
    print("-" * 70)
    fps = 30.0
    t_frame = 1.0 / fps
    t_line = t_frame / H
    print(f"  {fps:.0f} fps，一帧 {t_frame * 1000:.1f}ms，逐行读出，每行 {t_line * 1e6:.1f}us")
    print(f"  第 1 行曝光和第 {H} 行曝光相差整整 {t_frame * 1000:.1f}ms ——"
          f" 这不是'一帧内是同一时刻'。")
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
    print("  - 缓解手段：全局快门相机（贵 3-5 倍）、提高帧率（缩短读出时间）、")
    print("    或用 IMU 数据做逐行去畸变（主流做法，但要求 IMU 和相机硬件同步）。")


def part10_data_volume():
    print("\n[10] 数据量：为什么机器人数据集都用视频编码")
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
    print("实验 2：一台 RGB-D 相机输出的到底是什么")
    print("=" * 70)
    part1_frame_bytes()
    part2_pixel_size()
    part3_roundtrip()
    part4_resolution_scale()
    part5_distortion()
    part6_disparity()
    part7_depth_quantization()
    part8_extrinsics()
    part9_rolling_shutter()
    part10_data_volume()

    print("\n" + "=" * 70)
    print("结论（数据平台视角）：")
    print("  1. 内参 + 畸变系数必须和图像一起存。没有它们的图，只是没有坐标系的颜色。")
    print("  2. 深度不是'另一个通道'，是另一种物理测量：精度随距离平方变差。")
    print("  3. 外参误差随距离线性放大，且无法从数据本身自查 —— 必须靠标定流程保证。")
    print("  4. 空洞、飞点、量化台阶是深度图的固有产物，入库前必须标记而不是删除。")
    print("  5. 卷帘快门让'每帧一个时间戳'这个假设失效，高速数据要单独处理。")
    print("  6. 图像必须走视频编码，否则存储成本高一个数量级。")
    print("=" * 70)


if __name__ == "__main__":
    main()

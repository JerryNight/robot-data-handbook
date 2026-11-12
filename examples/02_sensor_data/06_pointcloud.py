#!/usr/bin/env python3
"""
实验 6：点云为什么难存、难用
==============================

激光雷达和深度相机都产出"三维点"，但两者的数据结构完全不同。
这个脚本量化：

  1. 角分辨率决定点间距 —— 越远越稀
  2. 线数对点数、带宽的影响
  3. 点云 vs 深度图：点数差几十倍，但谁的信息更多
  4. 投影到图像平面后的空洞率
  5. 存储格式与量化压缩的取舍
  6. 运动畸变：一帧不是同一个时刻
  7. 多回波、无效点与 intensity 的语义
  8. 体素化的内存：随范围三次方增长

运行：
    python examples/02_sensor_data/06_pointcloud.py

只依赖 Python 标准库。
"""

import math
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 一个典型 16 线机械式激光雷达
H_RES_DEG = 0.2        # 水平角分辨率
V_LINES = 16           # 线数
V_FOV_DEG = 30.0       # 垂直视场
FPS = 10.0

# 对比用的深度相机
W, H = 640, 480
CAM_FPS = 30.0
CAM_HFOV = 62.7        # 与实验 2 的 fx=525 对应
CAM_VFOV = 49.1


def part1_spacing():
    print("\n[1] 点间距：角分辨率 × 距离")
    print("-" * 72)
    v_res = V_FOV_DEG / (V_LINES - 1)
    print(f"  水平角分辨率 {H_RES_DEG}°   垂直 {V_LINES} 线 / {V_FOV_DEG}° "
          f"-> 线间 {v_res:.2f}°")
    print()
    print(f"  {'距离':>8}{'水平点间距':>14}{'垂直点间距':>14}{'各向异性比':>14}")
    print("  " + "-" * 56)
    for r in (2, 5, 10, 20, 50):
        h = r * math.tan(math.radians(H_RES_DEG))
        v = r * math.tan(math.radians(v_res))
        print(f"  {r:>6.0f}m{h * 1000:>12.1f}mm{v * 1000:>12.1f}mm"
              f"{v / h:>16.0f}x")

    print("\n  两个关键观察：")
    print("  1. 点间距随距离**线性**增长。10m 处水平 35mm、垂直 349mm。")
    print("  2. 水平和垂直差 10 倍 —— 因为水平 0.2°、垂直 2°，差一个数量级。")
    print()
    print("  第 2 点意味着：点云在各向异性上是严重极化的。")
    print("  远处的一辆车只有几行点，你能看到它的高度分布，但看不出细节。")
    print("  这不是分辨率不够，是**采样网格本身是长条形的**。")


def part2_point_count():
    print("\n[2] 点数与带宽")
    print("-" * 72)
    pts_per_ring = int(360.0 / H_RES_DEG)
    print(f"  水平一圈 {pts_per_ring} 点，{V_LINES} 线 -> "
          f"每帧 {pts_per_ring * V_LINES} 点")
    print()
    print(f"  {'线数':>6}{'每帧点数':>12}{'每秒点数':>14}{'原始带宽*':>14}")
    print("  " + "-" * 48)
    for lines in (16, 32, 64, 128):
        n = pts_per_ring * lines
        bps = n * FPS * 16        # xyz float32 + intensity = 16 bytes
        print(f"  {lines:>6}{n:>12,}{n * FPS:>14,.0f}{bps / 1e6:>11.2f}MB/s")

    print("\n  * 每点 4 个 float32（x, y, z, intensity）= 16 字节")
    print()
    print(f"  128 线雷达是 16 线的 8 倍点数、8 倍带宽，")
    print(f"  换来的是垂直分辨率从 {V_FOV_DEG / 15:.1f}° 提升到 "
          f"{V_FOV_DEG / 127:.2f}°。")
    print(f"  10m 处的垂直点间距从 {10 * math.tan(math.radians(2.0)) * 1000:.0f}mm "
          f"降到 {10 * math.tan(math.radians(V_FOV_DEG / 127)) * 1000:.0f}mm。")
    print()
    print("  这就是激光雷达按线数卖钱的原因 —— 但它同时也是数据平台的成本。")
    print(f"  8 小时采集，128 线：{pts_per_ring * 128 * FPS * 16 * 3600 * 8 / 1e12:.1f} TB（未压缩）")


def part3_vs_depth():
    print("\n[3] 点云 vs 深度图：谁的信息更多")
    print("-" * 72)
    lidar_pts = int(360.0 / H_RES_DEG) * V_LINES
    cam_pts = W * H
    print(f"  {'':<18}{'每帧点数':>12}{'帧率':>8}{'每秒点数':>14}")
    print("  " + "-" * 54)
    print(f"  {'16 线雷达':<16}{lidar_pts:>12,}{FPS:>7.0f}{lidar_pts * FPS:>14,.0f}")
    print(f"  {'640x480 深度图':<15}{cam_pts:>12,}{CAM_FPS:>7.0f}{cam_pts * CAM_FPS:>14,.0f}")

    ratio = cam_pts * CAM_FPS / (lidar_pts * FPS)
    print(f"\n  深度图每秒的点数是雷达的 {ratio:.0f} 倍。")
    print()
    print("  但这不代表深度图信息多 32 倍。区别在这里：")
    print()
    print("  - 雷达每一点都是**独立的飞行时间测量**，噪声独立、无内插")
    print("  - 深度图每一点是**像素级相位差或立体匹配**的结果：")
    print("    纹理缺失处直接输出 0（无效值），平滑处噪声在空间上强相关")
    print("  - 深度图的无效像素常常占 10–30%（玻璃、黑色物体、过远）")
    print()
    print("  所以两个数字不能直接比。正确的问法是：**有效测量点有多少？**")
    print(f"  16 线雷达：{lidar_pts * FPS:,.0f} 点/秒，全部有效")
    print(f"  深度图 20% 无效：{cam_pts * CAM_FPS * 0.8:,.0f} 点/秒")
    print(f"  折算后还是 {cam_pts * CAM_FPS * 0.8 / (lidar_pts * FPS):.0f} 倍，但这个折算本身仍然误导：")
    print("  深度图的空间噪声是强相关的（相邻像素共享同一个结构光斑/匹配窗口），")
    print("  307,200 个像素的实际独立自由度远小于这个数。")
    print("  真正该比的是『每平方米多少独立测量』，不是『每秒多少点』。")
    print()
    print("  而按**角覆盖**算，雷达只有 16 条线，深度图覆盖整个二维视场。")
    print("  这才是本质差异：雷达稀疏但每次测量相互独立，深度图稠密但误差成片相关。")


def part4_projection_holes():
    print("\n[4] 把点云投到图像平面上，有多少空洞")
    print("-" * 72)
    pts_per_ring = int(360.0 / H_RES_DEG)
    # 只取相机水平视场内的那一部分
    cols = int(pts_per_ring * CAM_HFOV / 360.0)
    rows = V_LINES
    v_res = V_FOV_DEG / (V_LINES - 1)

    print(f"  相机视场 {CAM_HFOV}° × {CAM_VFOV}°，图像 {W}x{H}")
    print(f"  雷达落在这个视场内的点：水平 {cols} 列 × 垂直 {V_LINES} 行 = {cols * rows} 点")
    print()
    print(f"  {'':<16}{'数值':>10}")
    print("  " + "-" * 28)
    items = [
        ("图像总像素", f"{W * H:,}"),
        ("有数据的像素", f"{cols * rows:,}"),
        ("空洞率", f"{(1 - cols * rows / (W * H)) * 100:.1f}%"),
        ("水平：每几列一个点", f"{W / cols:.1f}"),
        ("垂直：每几行一条线", f"{H / CAM_VFOV * v_res:.1f}"),
    ]
    for label, val in items:
        print(f"  {label}{' ' * max(1, 22 - len(label) * 2)}{val:>8}")

    print("\n  98% 的像素是空的。这就是为什么点云不能直接喂给图像模型：")
    print("  卷积核在 98% 的位置上看到的都是『没有测量』，不是『黑色』。")
    print()
    print("  两种主流处理方式：")
    print("  1. **投影成 range image**：按 (方位角, 仰角) 重排成 2D 图，")
    print("     每个像素存距离，空洞填 0。这样就能用 2D 卷积，代价是分辨率由角分辨率决定。")
    print("  2. **体素化**：把空间切成 5cm 的格子，每格标记占用/空。")
    print("     适合做规划，但丢掉点的精确位置，而且内存随范围三次方增长。")
    print()
    print("  选哪种，直接决定了下游能用什么模型。这是数据平台最该提前定的事 ——")
    print("  格式定错了，模型再好也补不回来。")


def part5_storage():
    print("\n[5] 存储格式与量化压缩")
    print("-" * 72)
    n = int(360.0 / H_RES_DEG) * V_LINES
    per_frame_pts = n

    print(f"  一帧 {per_frame_pts:,} 点，{FPS:.0f} fps")
    print()
    print(f"  {'格式':>24}{'每帧':>12}{'每秒':>14}{'备注':>10}")
    print("  " + "-" * 62)
    rows = [
        ("PCD ascii", per_frame_pts * 60, "文本"),
        ("PCD binary (16B/点)", per_frame_pts * 16, "原始"),
        ("量化 16-bit xyz + 8-bit i", per_frame_pts * 7, "量化"),
        ("Draco 压缩（典型 5:1）", per_frame_pts * 16 / 5, "压缩"),
        ("range image + PNG", W * H * 2 / 8, "重投影"),
    ]
    for name, per_frame, note in rows:
        print(f"  {name:>22}{per_frame / 1024:>10.0f}KB{per_frame * FPS / 1e6:>11.2f}MB/s"
              f"{note:>12}")

    print("\n  PCD ascii 是灾难：每点约 60 字节，是二进制格式的 4 倍。")
    print("  但它是默认选项之一（因为可以用文本编辑器打开），要主动关掉。")
    print()
    print("  量化到 16-bit 的代价：")
    rng = 200.0        # 假设坐标范围 ±100m
    res = rng / 65536
    print(f"    坐标范围 ±100m，16-bit -> 分辨率 {res * 1000:.2f}mm")
    print(f"    对 10m 处的点，{res * 1000:.2f}mm 的量化误差远小于雷达自身的测距噪声（±20mm）")
    print("    所以量化几乎免费 —— 这是点云压缩里性价比最高的一步。")
    print()
    print("  但要注意：量化后的坐标不能直接用于多帧配准。")
    print("    ICP 一类算法需要亚毫米精度，量化误差会让配准抖动。")
    print("    正确做法：原始精度用于配准，量化版本用于训练和存储。")


def part6_motion_distortion():
    print("\n[6] 运动畸变：一帧不是同一个时刻")
    print("-" * 72)
    pts_per_ring = int(360.0 / H_RES_DEG)
    scan_t = 1.0 / FPS
    per_pt_us = scan_t / pts_per_ring * 1e6
    print(f"  旋转式雷达 {FPS:.0f} Hz：转一圈 {scan_t * 1000:.0f} ms，")
    print(f"  一圈 {pts_per_ring} 个方位角 -> 相邻两列相差 {per_pt_us:.1f} µs。")
    print()
    print("  关键区别：相机是整帧同时曝光，雷达是**逐列扫描**。")
    print(f"  所以一帧点云的『第一个点』和『最后一个点』相差 {scan_t * 1000:.0f} ms，")
    print("  而它们被贴上的是同一个时间戳。")
    print()

    v = 1.0            # m/s，机器人直行速度
    w = 30.0           # °/s，机器人转向角速度
    print(f"  假设机器人以 {v:.1f} m/s 直行 + {w:.0f}°/s 转向：")
    print()
    print(f"  {'方位角':>8}{'被扫到的时刻':>14}{'已前进':>10}{'转向偏了':>10}{'5m 处横向错位':>16}")
    print("  " + "-" * 56)
    for deg in (0, 90, 180, 270, 359.8):
        t = deg / 360.0 * scan_t
        moved = v * t
        ang = w * t
        lat = 5.0 * math.tan(math.radians(ang))
        print(f"  {deg:>7.1f}°{t * 1000:>12.1f}ms{moved * 100:>9.1f}cm"
              f"{ang:>9.2f}°{lat * 100:>10.1f}cm")

    print()
    print(f"  看最后一列：同样是 5 m 远处一个静止的点，")
    print(f"  扫描靠后的那部分横向错位 "
          f"{5.0 * math.tan(math.radians(w * scan_t)) * 100:.0f} cm；")
    print(f"  再加上直行带来的 {v * scan_t * 100:.0f} cm 整体位移，两者叠加。")
    print("  一面平直的墙，在这帧点云里会变成一段斜的 / 扭的曲面。")
    print()
    print("  注意：这不等于『运动模糊』。")
    print("    图像的运动模糊是空间上糊掉，点还是那个点；")
    print("    点云的畸变是几何上错位 —— 点依然锐利，但它不在该在的位置。")
    print("    所以它不能靠去卷积修，只能靠运动补偿（de-skew）。")
    print()
    print("  补偿要求：每一点都有自己的时间戳。")
    print(f"    只存帧时间戳 -> 只能用帧首或帧中一个时刻，残余误差 = v·T/2 = "
          f"{v * scan_t / 2 * 100:.1f} cm")
    print("    存逐点时间偏移（PCD 的 t 字段 / PointCloud2 的 per-point time）")
    print("    -> 每点单独插值出当时的位姿，残余误差降到毫米级")
    print()
    print("  这也是为什么雷达数据必须和 IMU / 轮速计一起采：")
    print("  补偿要的是『扫描过程中每时每刻的位姿』，而这个位姿只能靠高频的惯性数据插值。")


def part7_returns_and_invalid():
    print("\n[7] 多回波、无效点与 intensity 的语义")
    print("-" * 72)
    pts_per_ring = int(360.0 / H_RES_DEG)
    n = pts_per_ring * V_LINES
    print("  (a) 一条光线可以有多个答案")
    print(f"      激光穿过树冠：一部分能量被树叶反射，一部分继续打到地面。")
    print(f"      双回波模式下，同一条光线返回两个距离。")
    print()
    second = 0.30
    print(f"      16 线 {n:,} 点/帧，若 {second:.0%} 的光线有第二回波：")
    print(f"      点数变成 {n * (1 + second):,.0f}（+{second:.0%}），")
    print("      而且每个点必须带『第几回波』这个字段 —— 否则两批测量混成一堆，")
    print("      下游会把地面点和树冠点当成同一个表面的两次抖动。")
    print()
    print("  (b) 无效点和假点，来源完全不同")
    rows = [
        ("玻璃 / 镜面", "15–40%", "穿透或反射到别处，整片缺失"),
        ("雨雾", "5–20%", "假点：悬浮水滴被当成障碍物"),
        ("黑色低反射面", "5–15%", "距离跳变或直接丢失"),
        ("超量程", "视场景", "无回波"),
    ]
    for a, b, c in rows:
        print(f"      {a:<8}{b:>8}   {c}")
    print()
    print("      注意『无效』和『假点』是两件事：")
    print("      无效点是没有测量，假点是测量错了。前者能靠 mask 掉，后者会污染地图。")
    print("      雨天点云的噪点不是噪声大，是**多了很多不存在的物体**。")
    print()
    print("  (c) intensity 不是反射率")
    near, far = 5.0, 20.0
    print(f"      回波强度按 1/d² 衰减：{near:.0f} m 与 {far:.0f} m 差 "
          f"{(far / near) ** 2:.0f} 倍。")
    print("      同一块白墙，近处 intensity 255、远处可能只有 16。")
    print("      而且不同厂家给的 intensity 物理含义不同：")
    print("        · 回波峰值功率（与距离强相关）")
    print("        · 标定后的反射率（已做距离补偿）")
    print("        · 0–255 归一化（厂家自己的映射）")
    print("      所以跨设备拼接点云时，intensity 通道基本不可比 —— 只能当边缘/纹理线索用。")


def part8_voxel_memory():
    print("\n[8] 体素化：内存随范围三次方增长")
    print("-" * 72)
    print("  体素把空间切成固定大小的格子。听起来很规整，代价在内存：")
    print()
    print(f"  {'范围(边长)':>12}{'高度':>8}{'分辨率':>8}{'格数':>14}{'稠密内存(1B/格)':>18}")
    print("  " + "-" * 64)
    for side, height, res in ((100, 10, 0.05), (100, 10, 0.02),
                              (200, 10, 0.05), (200, 10, 0.02)):
        cells = (side / res) * (side / res) * (height / res)
        mem = cells  # 1 byte / cell
        unit = f"{mem / 1e9:.1f} GB" if mem > 1e9 else f"{mem / 1e6:.0f} MB"
        print(f"  {side:>10.0f}m{height:>7.0f}m{res * 100:>7.0f}cm{cells:>14.2e}{unit:>18}")

    print()
    worst = (200 / 0.02) ** 2 * (10 / 0.02)
    print("  两件事同时发生：")
    print("    · 范围加倍 -> 平面格数 ×4（高度不变时；三维都变就是 ×8）")
    print("    · 分辨率从 5cm 到 2cm -> 格数 ×15.6")
    print(f"  所以『200m 范围 + 2cm 精度』是 {worst / 1e9:.0f} GB 的量级，")
    print("  这在单车单帧上是不可能的，必须用稀疏结构（哈希 / 八叉树）。")
    print()
    n = int(360.0 / H_RES_DEG) * V_LINES
    dense_100 = (100 / 0.05) ** 2 * (10 / 0.05)
    print(f"  但一帧只有 {n:,} 个点 —— 稀疏表示只需要存非空的 {n:,} 个格子：")
    print(f"    稠密 100m/5cm：{dense_100 * 1e-6:.0f} MB")
    print(f"    稀疏（每格 16B：坐标+占用）：{n * 16 / 1e6:.2f} MB")
    print(f"    差 {dense_100 / (n * 16):.0f} 倍。")
    print()
    print("  这就是为什么点云库（PCL / Open3D）里体素过滤器都默认用稀疏结构。")
    print("  代价是失去了『按索引随机访问』的能力 —— 卷积用不了，必须用稀疏卷积。")


def main():
    print("=" * 72)
    print("实验 6：点云的稀疏性与存储")
    print("=" * 72)
    part1_spacing()
    part2_point_count()
    part3_vs_depth()
    part4_projection_holes()
    part5_storage()
    part6_motion_distortion()
    part7_returns_and_invalid()
    part8_voxel_memory()

    print("\n" + "=" * 72)
    print("结论（数据平台视角）：")
    print("  1. 点云的网格是各向异性的，远距离的垂直稀疏是最主要的限制")
    print("  2. 点数不等于信息量，要按『有效测量』而不是『像素数』来评估")
    print("  3. 投影成 range image 还是体素化，要按下游模型提前定，不能事后补")
    print("  4. 量化到 16-bit 几乎无损，是最该做的第一项压缩")
    print("  5. 逐点时间戳是运动补偿的前提，丢了它就只能整体近似")
    print("  6. 第几回波、是否无效、intensity 的物理含义，都要进 schema")
    print("=" * 72)


if __name__ == "__main__":
    main()

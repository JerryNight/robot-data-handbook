#!/usr/bin/env python3
"""
实验 5：六维力/力矩，以及触觉为什么至今没有统一格式
=====================================================

腕部力传感器测的是"传感器感受到的总力"，里面混着工具自身重力。
这个脚本量化：

  1. 六维 F/T 的读数怎么随姿态变化（重力在传感器坐标系里的投影）
  2. 不做重力补偿会错多少
  3. 零点漂移有多致命
  4. 触觉阵列的数据规模，以及它为什么碎片化

运行：
    python examples/02_sensor_data/05_force_torque.py

只依赖 Python 标准库。
"""

import math
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

G = 9.81
M_TOOL = 1.5                      # 末端工具 + 夹爪质量 kg
COM = (0.0, 0.0, 0.08)            # 工具重心在传感器坐标系中的位置 m


# ---------------------------------------------------------------- 模型

def gravity_reading(theta_deg):
    """
    传感器绕自身 y 轴旋转 theta（俯仰），世界重力在传感器系里的分量。

        世界重力 (0, 0, -mg)  --R^T-->  传感器系
        F_s = (-mg·sinθ, 0, -mg·cosθ)

    力矩 = r × F（r 是重心相对传感器的位置）。

    返回 (Fx, Fy, Fz, Tx, Ty, Tz)。
    """
    th = math.radians(theta_deg)
    fx = -M_TOOL * G * math.sin(th)
    fy = 0.0
    fz = -M_TOOL * G * math.cos(th)

    rx, ry, rz = COM
    tx = ry * fz - rz * fy
    ty = rz * fx - rx * fz
    tz = rx * fy - ry * fx
    return fx, fy, fz, tx, ty, tz


# ---------------------------------------------------------------- 各部分

def part1_meaning():
    print("\n[1] 六维力/力矩是什么")
    print("-" * 72)
    print("  三个力 + 三个力矩，全部在**传感器自己的坐标系**下：")
    print()
    print("    Fx, Fy, Fz   沿三个轴的推力/拉力 (N)")
    print("    Tx, Ty, Tz   绕三个轴的扭矩 (N·m)")
    print()
    print("  为什么必须强调『传感器坐标系』：同一个动作，传感器装的方向不同，")
    print("  六个数字完全不同。数据里如果不记传感器安装方向，这六个数字没法用。")
    print()
    print("  六维传感器的典型规格（ATI / 宇立 / Bota 一类）：")
    print(f"  {'指标':<22}{'典型值':<24}{'含义'}")
    print("  " + "-" * 66)
    rows = [
        ("量程 (力)", "100–500 N", "超过会永久损坏"),
        ("量程 (力矩)", "5–30 N·m", "拧螺丝时容易超"),
        ("分辨率", "0.01–0.1 N", "远好于精度，别被误导"),
        ("耦合误差", "1–2% FSS", "施加 Fx 会在 Fz 上读数"),
        ("零点漂移", "0.05–0.5 N/°C", "温度变了零点就变"),
        ("采样率", "100 Hz – 7 kHz", "高频才能抓到碰撞瞬间"),
    ]
    for a, b, c in rows:
        print(f"  {a:<22}{b:<24}{c}")


def part2_gravity_compensation():
    print("\n[2] 不做重力补偿，读数里有什么")
    print("-" * 72)
    print(f"  末端工具质量 {M_TOOL} kg，重心在传感器系 {COM} m")
    print("  传感器绕自身 y 轴俯仰，看六个读数怎么变：")
    print()
    print(f"  {'俯仰角':>6}{'Fx(N)':>9}{'Fz(N)':>9}{'Ty(N·m)':>10}"
          f"{'|F|(N)':>10}   说明")
    print("  " + "-" * 60)
    for deg in (0, 15, 30, 45, 60, 90, 120, 150, 180):
        fx, fy, fz, tx, ty, tz = gravity_reading(deg)
        mag = math.sqrt(fx * fx + fy * fy + fz * fz)
        if abs(deg) < 1:
            note = "纯竖直，力全在 z"
        elif abs(deg - 90) < 1:
            note = "纯水平，力全在 x"
        elif abs(deg - 180) < 1:
            note = "倒过来，z 反号"
        else:
            note = ""
        print(f"  {deg:>5}°{fx:>9.2f}{fz:>9.2f}{ty:>10.3f}{mag:>10.2f}   {note}")

    print(f"\n  力的大小恒为 {M_TOOL * G:.2f} N —— 它一直是同一个重力，")
    print("  只是在传感器坐标系里换了方向。")
    print()
    print("  这就是为什么原始 F/T 数据不能直接拿来学：")
    print("  机器人只是转了个手腕，传感器读数就从 (0, 0, -14.7) 变成 (-14.7, 0, 0)，")
    print("  而这一时刻它什么都没碰。")

    print("\n  重力补偿要做三步：")
    print("    1. 知道传感器在世界系下的姿态 R（来自正运动学 + TF）")
    print("    2. F_world_gravity = (0, 0, -mg)，转到传感器系：F_g = R^T · (0,0,-mg)")
    print("    3. 补偿后 F_ext = F_measured - F_g，力矩同样减去 r × F_g")
    print()
    print(f"  第 2 步依赖姿态，所以**每一条 F/T 数据都必须配一条同时刻的姿态**。")
    print("  这也是为什么很多公开数据集里的 F/T 是不可用的 —— 只存了六个数，没存姿态。")


def part3_with_contact():
    print("\n[3] 有接触力时：三种读法差多少")
    print("-" * 72)
    print("  场景：工具沿自身 z 轴向下按压，真实接触力恒为 5 N（传感器系 (0,0,-5)）。")
    print("  机器人手腕一边压一边改变俯仰角。三种读法对比：")
    print()
    print("    读法① 直接取 |Fz|                 <- 最常见的错")
    print("    读法② 取 |F| 再减重力模长          <- 看起来聪明，仍然错")
    print("    读法③ 先用姿态把重力转过去再减      <- 正确做法")
    print()
    print(f"  {'姿态':>5}{'Fx':>9}{'Fz':>9}{'读法①':>10}{'读法②':>10}{'读法③':>10}")
    print("  " + "-" * 56)
    for deg in (0, 45, 90, 135, 180):
        gx, _, gz, _, _, _ = gravity_reading(deg)
        cx, cz = 0.0, -5.0                 # 接触力在传感器系里固定，沿 -z
        fx, fz = gx + cx, gz + cz
        read1 = abs(fz)
        read2 = math.hypot(fx, fz) - M_TOOL * G
        read3 = 5.0                        # 补偿后还原真值
        print(f"  {deg:>4}°{fx:>9.2f}{fz:>9.2f}{read1:>10.2f}{read2:>10.2f}{read3:>10.2f}")

    print("\n  真值那一行本该全是 5.00。实际读法①②的跨度：")
    print("    读法① 从 5.00 到 19.71 N，误差最高 +14.71 N（= 整个重力）")
    print("    读法② 从 -5.00 到 5.00 N，过了 135° 直接变成负的接触力")
    print()
    print("  为什么读法②也不行：它假设重力和接触力同向，模长可以直接相减。")
    print("  一旦手腕转到某个角度让两者成直角，这个假设就彻底崩了。")
    print()
    print("  这就是力控数据里最隐蔽的一类污染：**数值合理、量级合理、但物理上错**。")
    print("  它不会让 loss 变差（模型照样能拟合），只会让策略在换姿态时失效。")


def part4_zero_drift():
    print("\n[4] 零点漂移：比重力更难缠")
    print("-" * 72)
    print("  力传感器每次上电，零点都不一样（几十 mN 到几百 mN）。")
    print("  常见做法是开机时空载归零。但温度一变，零点又会漂。")
    print()
    print(f"  {'零点误差':>10}{'5N 接触时的相对误差':>22}{'1N 精细操作的相对误差':>24}")
    print("  " + "-" * 58)
    for z in (0.02, 0.05, 0.1, 0.2, 0.5, 1.0):
        print(f"  {z:>8.2f}N{(z / 5.0) * 100:>20.0f}%{(z / 1.0) * 100:>22.0f}%")

    print("\n  看点：对 5N 的粗操作，0.1N 的零点误差只有 2%，无所谓；")
    print("  但对 1N 的精细操作（插孔、拧螺丝），同样误差是 10%。")
    print()
    print("  所以精细力控数据集必须记录：")
    print("  - 每次零点的标定值和标定时刻")
    print("  - 传感器温度（如果驱动能读到）")
    print("  - 静止段（用来事后重新估计零点）")
    print()
    print("  这和 IMU 的结论完全一致：**静止段是校准的黄金数据，别当废数据裁掉。**")


def part5_tactile():
    print("\n[5] 触觉：数据量最小，格式最乱")
    print("-" * 72)
    print(f"  {'方案':>24}{'分辨率':>14}{'帧率':>10}{'原始带宽':>14}")
    print("  " + "-" * 64)
    rows = [
        ("单点力传感器", "1 点", "1 kHz", 4e3),
        ("16x16 阵列 (8-bit)", "256 点", "100 Hz", 256 * 100),
        ("32x32 阵列 (8-bit)", "1024 点", "100 Hz", 1024 * 100),
        ("64x64 阵列 (16-bit)", "4096 点", "1 kHz", 4096 * 2 * 1000),
        ("视觉触觉 (GelSight)", "320x240x3", "30 Hz", 320 * 240 * 3 * 30),
        ("对比：RGB 相机", "640x480x3", "30 Hz", 640 * 480 * 3 * 30),
    ]
    for name, res, fps, bps in rows:
        print(f"  {name:>22}{res:>14}{fps:>10}{bps / 1024:>11.0f}KB/s")

    print("\n  触觉的数据量比相机小两三个数量级 —— 但它是所有模态里最碎片化的。")
    print("  原因不在数据量，在这三件事：")
    print()
    print("  1. **物理原理完全不同**：压阻、电容、压电、光学、磁性。")
    print("     压阻测的是静态压力，压电测的是压力变化率，两者的数据不能放进同一张表。")
    print("  2. **没有统一的空间语义**：16x16 的阵列贴在哪里、多大面积、")
    print("     每个单元对应皮肤上哪一块 —— 不同产品完全不可比。")
    print("  3. **标定不可迁移**：每个单元的一致性差异能到 20%，")
    print("     厂商的标定文件是私有格式，换个传感器就得重新标。")
    print()
    print("  结果：触觉数据几乎无法跨数据集复用。这也是为什么 Open X-Embodiment")
    print("  这类大规模聚合数据集里，触觉基本是缺席的。")
    print()
    print("  对比一下：图像的『640x480 RGB、像素在图像平面上的位置』这个语义")
    print("  三十年没变过，所以图像能跨本体、跨任务聚合。触觉还没有这个东西。")
    print("  这是 07 章里我认为最值得做的开放问题之一。")


def main():
    print("=" * 72)
    print("实验 5：六维力/力矩与触觉")
    print("=" * 72)
    part1_meaning()
    part2_gravity_compensation()
    part3_with_contact()
    part4_zero_drift()
    part5_tactile()

    print("\n" + "=" * 72)
    print("结论（数据平台视角）：")
    print("  1. F/T 数据必须和姿态成对存储，否则重力补偿无从下手")
    print("  2. 补偿应该在入库前做好并同时保留原始值，两者都要")
    print("  3. 零点标定值和温度是元数据，不是调试信息")
    print("  4. 触觉的瓶颈是语义不统一，不是带宽 —— 这是聚合数据集缺触觉的根因")
    print("=" * 72)


if __name__ == "__main__":
    main()

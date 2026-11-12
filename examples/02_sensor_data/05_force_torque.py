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
  5. 力矩的参考点：同样是 1 N 的力，换个参考点力矩就减半
  6. 耦合误差：压 x 方向，z 通道也会动
  7. 接触检测阈值：3σ 在 1 kHz 下每秒误报几次

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
    contact_n = 5.0                    # <- 改这里可以换任务尺度
    print(f"  {'姿态':>5}{'Fx':>9}{'Fz':>9}{'读法①':>10}{'读法②':>10}{'读法③':>10}")
    print("  " + "-" * 56)
    rows = []
    for deg in (0, 45, 90, 135, 180):
        gx, _, gz, _, _, _ = gravity_reading(deg)
        cx, cz = 0.0, -contact_n           # 接触力在传感器系里固定，沿 -z
        fx, fz = gx + cx, gz + cz
        read1 = abs(fz)
        read2 = math.hypot(fx, fz) - M_TOOL * G
        read3 = contact_n                  # 补偿后还原真值
        rows.append((deg, read1, read2))
        print(f"  {deg:>4}°{fx:>9.2f}{fz:>9.2f}{read1:>10.2f}{read2:>10.2f}{read3:>10.2f}")

    r1 = [r[1] for r in rows]
    r2 = [r[2] for r in rows]
    print(f"\n  真值那一行本该全是 {contact_n:.2f}。实际读法①②的跨度：")
    print(f"    读法① 从 {min(r1):.2f} 到 {max(r1):.2f} N，"
          f"误差最高 {max(r1) - contact_n:+.2f} N（= 整个重力）")
    neg = [f"{r[0]}° 的 {r[2]:.2f} N" for r in rows if r[2] < 0]
    if neg:
        print(f"    读法② 从 {min(r2):.2f} 到 {max(r2):.2f} N，出现负接触力：{'、'.join(neg)}")
    else:
        print(f"    读法② 从 {min(r2):.2f} 到 {max(r2):.2f} N")
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


def part6_torque_reference():
    print("\n[6] 力矩的参考点：同一股力，换个原点力矩就变了")
    print("-" * 72)
    print("  力矩的定义里有位置：τ = r × F。r 是力作用点相对于**你选的原点**。")
    print("  所以『力矩是多少』这句话不完整 —— 必须说『相对哪个点』。")
    print()
    print(f"  工具重心在传感器系 z = {COM[2]} m，重力 {M_TOOL * G:.2f} N。")
    print("  传感器出厂标定时选用的是传感器原点。如果有人把它填成了法兰面")
    print("  （沿 z 偏 0.04 m），r 就变成 0.04，力矩读数整体缩一半：")
    print()
    print(f"  {'角度':>6}{'Fx(N)':>10}{'Ty 原点0.08':>14}{'Ty 原点0.04':>14}{'比值':>8}")
    print("  " + "-" * 56)
    for deg in (0, 30, 45, 60, 90):
        fx, _, fz, _, ty, _ = gravity_reading(deg)
        ty2 = ty * 0.04 / COM[2]
        ratio_txt = f"{ty2 / ty:.2f}x" if abs(ty) > 1e-12 else "—"
        print(f"  {deg:>5}°{fx:>10.2f}{ty:>14.3f}{ty2:>14.3f}{ratio_txt:>8}")

    print()
    print("  看 45° 那一行：-0.832 变成 -0.416。**力和力矩的量级都在正常范围内**，")
    print("  没有任何一项越界、没有任何报错，只是力矩通道整体偏了一半。")
    print()
    print("  这就是力觉数据里最难查的一类错误：它不是噪声，是一个**比例系数**错了。")
    print("  用这样的数据做力控，机器人会以为自己的力矩只有实际的一半，")
    print("  于是继续加力 —— 目标 5 N·m，实际会压到 10 N·m 才读数 5。")
    print()
    print("  工程做法：")
    print("  - 力矩数据的元数据里写明参考点（不是“传感器系”，而是“相对传感器原点多远”）")
    print("  - 参考点变更必须换版本号，不能原地改")
    print("  - 入库校验：静止时用已知质量块反算力矩，和读数比对")


def part7_crosstalk():
    print("\n[7] 耦合误差：压 x 方向，z 通道也会动")
    print("-" * 72)
    print("  六维传感器的六个通道不是正交的。出厂时标定一个 6×6 解耦矩阵，")
    print("  典型残余耦合 1–2% FSS（FSS = full scale，满量程）。")
    print()
    FSS = 200.0        # 力的满量程 N
    print(f"  假设满量程 {FSS:.0f} N，残余耦合 1.5% → 串扰系数 {0.015:.4f}")
    print()
    print(f"  {'施加 Fx':>10}{'Fz 虚读':>12}{'相当于 1 N 判据的':>20}")
    print("  " + "-" * 46)
    for fx in (10, 50, 100, 200):
        ghost = fx * 0.015
        note = "{:.0f}%".format(ghost / 1.0 * 100)
        print(f"  {fx:>8.0f}N{ghost:>10.2f}N{note:>18}")

    print()
    print("  读法：插孔这类任务，接触力判据常在 1 N 量级。")
    print(f"  只要你同时沿 x 方向推了 100 N（很常见，比如按压的同时在侧向找孔），")
    print(f"  z 通道就凭空多出 {100 * 0.015:.1f} N —— 足以让控制器误判“已经插到底了”。")
    print()
    print("  而且这个系数会随温度漂、随使用老化。所以：")
    print("  - 数据集要记录解耦矩阵的版本和标定时间")
    print("  - 只做“减法去重力”是不够的，严格做法是先乘解耦矩阵的逆")
    print("  - 但注意：解耦矩阵的逆会放大噪声，很多驱动默认不启用，要查文档")


def part8_contact_threshold():
    print("\n[8] 接触检测阈值：3σ 在 1 kHz 下每秒误报一次")
    print("-" * 72)
    print("  最常见的接触判据：读数超过 均值 + k·σ 就算“碰到了”。")
    print("  问题在于，噪声本身就会越过这条线 —— 而且越采越快，越容易撞上。")
    print()
    sigma = 0.05       # N，典型噪声
    fs = 1000.0        # Hz，采样率  <- 改这里可以看采样率对误报率的影响
    n_confirm = 5      # 连续确认点数 <- 改这里可以看延迟与误报的权衡
    print(f"  假设静止时噪声 σ = {sigma:.2f} N，采样率 {fs:.0f} Hz。")
    print()
    # 单边高斯尾概率
    def tail(k):
        # 用 erfc 近似，标准正态上尾
        return 0.5 * math.erfc(k / math.sqrt(2))

    print(f"  {'阈值':>10}{'单点越界概率':>16}{'每秒误报次数':>16}{'判定':>12}")
    print("  " + "-" * 56)
    for k in (2.0, 3.0, 4.0, 5.0):
        p = tail(k)
        per_sec = p * fs
        verdict = "不可用" if per_sec > 0.5 else ("勉强" if per_sec > 0.01 else "可用")
        print(f"  {k:>8.1f}σ{p:>15.2e}{per_sec:>14.3g}{verdict:>12}")

    print()
    print("  3σ 看着很保守（99.7% 都在里面），但 1 kHz 下每秒有 1.3 次越界。")
    print("  如果按『任一通道越界就算接触』，6 个通道加起来还要翻几倍。")
    print()
    print("  实际做法是两条：")
    print("  1. **滞回**：进入阈值 0.15 N，退出阈值 0.10 N。避免在阈值附近反复抖动")
    print(f"  2. **连续 N 点确认**：要求连续 {n_confirm} 个采样点都越界。")
    p_conf = tail(3.0) ** n_confirm
    print(f"     {n_confirm} 点都越界的概率是 {p_conf:.2e}，"
          f"每秒误报降到 {p_conf * fs:.2e} 次，判定延迟 {n_confirm / fs * 1000:.0f} ms")
    print()
    print("  这两条都是**采集端**该做的事：事后从原始数据重算，你没法知道当时")
    print("  用的是哪个阈值、有没有滞回。所以接触状态应该作为一列独立存下来，")
    print("  并且把生成它的阈值参数写进数据集元数据。")


def main():
    print("=" * 72)
    print("实验 5：六维力/力矩与触觉")
    print("=" * 72)
    part1_meaning()
    part2_gravity_compensation()
    part3_with_contact()
    part4_zero_drift()
    part5_tactile()
    part6_torque_reference()
    part7_crosstalk()
    part8_contact_threshold()

    print("\n" + "=" * 72)
    print("结论（数据平台视角）：")
    print("  1. F/T 数据必须和姿态成对存储，否则重力补偿无从下手")
    print("  2. 补偿应该在入库前做好并同时保留原始值，两者都要")
    print("  3. 零点标定值、温度、参考点、解耦矩阵版本都是元数据，不是调试信息")
    print("  4. 触觉的瓶颈是语义不统一，不是带宽 —— 这是聚合数据集缺触觉的根因")
    print("  5. 接触状态要么作为独立列存下来，要么把判定参数写进元数据")
    print("=" * 72)


if __name__ == "__main__":
    main()

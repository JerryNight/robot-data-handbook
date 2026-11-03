#!/usr/bin/env python3
"""
实验 4：同一个数据集，五种布局，随机访问成本差 100 倍。

运行：python examples/03_data_pipeline/04_formats.py

格式选择看起来是「偏好问题」，其实是**物理问题**：
训练时 dataloader 要随机抽样本，而磁盘只能按块读。
布局决定了「抽一帧」实际要搬多少字节。

    [1] 五种布局的 I/O 放大倍数（全部可精确计算）
    [2] 视频编码的双刃剑：体积降 40 倍，代价是随机访问要解整个 GOP
    [3] LeRobotDataset 的真实目录结构，以及那三个统计量为什么必须存
    [4] 一次真实规模转换的时间与空间账
"""

import math


def width(s: str) -> int:
    return sum(2 if ord(c) > 0x2E80 else 1 for c in s)


def pad(s: str, n: int) -> str:
    return s + " " * max(0, n - width(s))


# 一帧 640x480 RGB 的尺寸；后面所有数字都从这里推
W, H, C = 640, 480, 3
RAW_FRAME = W * H * C          # 921,600 字节
EPISODE_FRAMES = 100
N_EPISODES = 1000


def part1_layouts():
    print("\n[1] 五种布局：抽一帧要读多少字节")
    print("-" * 78)
    print("  数据集：%d 个 episode × %d 帧，单帧原始 %d KB（%dx%d RGB）"
          % (N_EPISODES, EPISODE_FRAMES, RAW_FRAME // 1024, W, H))
    print("  训练时随机抽一帧 —— 这就是 dataloader 每步都在做的事。")
    print()

    # (名称, 磁盘读取字节, 需解码成的像素字节, 说明)
    layouts = [
        ("A 每 episode 一个大文件",
         EPISODE_FRAMES * RAW_FRAME, EPISODE_FRAMES * RAW_FRAME,
         "读整集才能定位，简单但浪费"),
        ("B 分块 chunk=100 帧",
         100 * RAW_FRAME, 100 * RAW_FRAME,
         "h5py 默认块大小的常见误设"),
        ("C 分块 chunk=1 帧",
         RAW_FRAME, RAW_FRAME,
         "每帧独立块，接近最优"),
        ("D 每帧一个文件",
         RAW_FRAME, RAW_FRAME,
         "字节相同，inode 和寻道另算"),
        ("E 视频 H.264 GOP=30",
         30 * 13000, 30 * RAW_FRAME,
         "磁盘只读 0.4MB，但要解 30 帧"),
    ]

    print("  %s%s%s%s" % (pad("布局", 30), pad("磁盘读取", 12),
                          pad("解码像素量", 14), "说明"))
    print("  " + "-" * 78)
    for name, disk, decoded, note in layouts:
        print("  %s%s%s%s" % (pad(name, 30),
                              pad("%.2f MB" % (disk / 1e6), 12),
                              pad("%.2f MB" % (decoded / 1e6), 14), note))

    print()
    print("  A 和 C 差的不是压缩率，是**寻址粒度**：一个 batch 64 个样本，")
    for name, disk, _d, _n in (layouts[0], layouts[2]):
        print("    %s%6.0f MB" % (pad(name, 30), 64 * disk / 1e6))
    print("  差 %.0f 倍。磁盘带宽不变的情况下，这就是 epoch 时间的差距。"
          % (layouts[0][1] / layouts[2][1]))
    print()
    print("  E 那一行要竖着看：磁盘只读 0.39 MB（比 C 还少），")
    print("  但解码器必须处理 27.6 MB 的像素——**成本从 IO 转成了 CPU/GPU**。")

    print()
    print("  而 D（每帧一个文件）虽然字节数最优，却有另一个问题：")
    total_files = N_EPISODES * EPISODE_FRAMES
    print("    %d 个文件。ext4 的 inode 查找、S3 的 LIST、NFS 的元数据请求，"
          % total_files)
    print("    都会让「按名字打开」比「按 offset 读」慢一个数量级以上。")
    print("    经验阈值：单目录文件数超过 1 万，就该考虑打包或分桶。")


def part2_video_tradeoff():
    print("\n[2] 视频编码：省了 40 倍空间，赔上随机访问")
    print("-" * 78)
    # bits per pixel 的典型值（自然图像，公开基准的量级）
    codecs = [
        ("原始 RGB", 24.0, "无"),
        ("PNG（无损）", 9.0, "无"),
        ("JPEG q90", 1.6, "无（帧内）"),
        ("H.264 CRF23", 0.35, "GOP 内依赖"),
        ("H.265 CRF23", 0.25, "GOP 内依赖"),
    ]
    total_frames = N_EPISODES * EPISODE_FRAMES
    print("  %s%s%s%s" % (pad("编码", 16), pad("bpp", 8),
                          pad("单帧", 12), "全量 %s 帧  相对原始" % format(total_frames, ",")))
    print("  " + "-" * 62)
    base = None
    for name, bpp, _dep in codecs:
        per = W * H * bpp / 8
        if base is None:
            base = per
        print("  %s%s%s%s" % (pad(name, 16), pad("%.2f" % bpp, 8),
                              pad("%.0f KB" % (per / 1024), 12),
                              "%.1f GB  (%.0fx)" % (per * total_frames / 1e9,
                                                    base / per)))

    print()
    print("  只看这张表，结论是「必须用 H.264」。但代价在另一列，不能忽略：")
    print()
    gop = 30
    rows = [
        ("解一帧的代价", "O(1)", "平均 O(GOP/2)"),
        ("硬件解码器", "不需要", "需要，且并发数有限"),
        ("字节级随机访问", "可以", "不行，只能按 GOP 边界切"),
        ("无损还原", "是（PNG）", "否，颜色/高频细节丢失"),
    ]
    print("  %s%s%s" % (pad("", 22), pad("帧内编码", 16), "帧间编码"))
    print("  " + "-" * 58)
    for a, b, c in rows:
        print("  %s%s%s" % (pad(a, 22), pad(b, 16), c))

    print()
    print("  所以工业界的实际做法是**分层**：")
    print("    · 训练用：H.264/H.265 视频，解码后送 GPU（LeRobot 就这样做）")
    print("    · 评测/回放用：保留关键帧的 PNG，或者按需重编码")
    print("    · 高精度需求（标定、深度、力觉）：绝不走视频，用原始或无损")
    print()
    print("  一个具体数字：GOP=30 时随机抽一帧，平均要解 15 帧。")
    print("  GPU 批量解码下这几乎是免费的，但**在 CPU dataloader 里**会成为瓶颈——")
    print("  这就是为什么这类数据集通常要求 worker 数 > 8 或者直接用 DALI/torchcodec。")


def part3_lerobot_layout():
    print("\n[3] LeRobotDataset 的目录结构，和那三个不起眼的文件")
    print("-" * 78)
    tree = [
        ("meta/info.json", "总帧数、fps、每个 feature 的 shape/dtype"),
        ("meta/episodes.jsonl", "每个 episode 的起止帧号、任务文本"),
        ("meta/stats.json", "每个 feature 的 mean/std/min/max（全局）"),
        ("meta/tasks.jsonl", "任务文本去重后的表"),
        ("data/chunk-000/episode_000000.parquet", "低维数据：state/action/timestamp"),
        ("videos/chunk-000/observation.images.top/episode_000000.mp4", "图像走视频"),
    ]
    print("  %s%s" % (pad("路径", 62), "内容"))
    print("  " + "-" * 96)
    for p, d in tree:
        print("  %s%s" % (pad(p, 62), d))

    print()
    print("  三个设计决策，每个都在解决一个具体问题：")
    print()
    print("  ① **图像不进 Parquet，走 mp4。**")
    print("     原因就是上面 [1] 的表：Parquet 是列式，但一列 BLOB 里塞 60 万字节，")
    print("     压缩率极差、解压也慢。视频编码器专门为「连续相似帧」优化，")
    print("     天然比通用压缩强一个数量级。")
    print()
    print("  ② **meta/stats.json 存全局 mean/std。**")
    print("     这个文件看起来最没用，其实是复现性的关键：")
    print("     归一化统计量必须**训练和推理完全一致**。如果每次训练现算，")
    print("     换一批数据、换个采样顺序，统计量就变了——模型在真机上直接崩。")
    print("     存下来，等于把「数据预处理」变成了数据集的一部分。")
    print()
    print("  ③ **chunk-000 / chunk-001 分桶。**")
    print("     单目录一万个文件的那个阈值。分桶后每个目录固定 1000 个 episode，")
    print("     文件系统、S3 LIST、并行读取都不会退化。")

    print()
    print("  一个容易忽略的细节：episodes.jsonl 用 **jsonl 而不是 json**。")
    print("  一行一条，可以流式读、可以 grep、可以被 map-reduce 切分。")
    print("  十万个 episode 的单个 json 数组要整体载入内存——这是不必要的。")

    print()
    print("  对比 RLDS（TFRecord + schema）：")
    rows = [
        ("存储单元", "episode 级 tfrecord 分片", "帧级 parquet + 视频"),
        ("随机访问", "受分片大小限制", "帧级，粒度更细"),
        ("生态", "TF / JAX 为主", "PyTorch 为主"),
        ("schema", "必须显式声明", "从 parquet 推断 + info.json"),
        ("灵活度", "低（改 schema 要重写）", "高（加一列即可）"),
    ]
    print("  %s%s%s" % (pad("维度", 18), pad("RLDS / TFRecord", 28), "LeRobotDataset"))
    print("  " + "-" * 70)
    for a, b, c in rows:
        print("  %s%s%s" % (pad(a, 18), pad(b, 28), c))


def part4_conversion_cost():
    print("\n[4] 一次真实规模的转换：时间、空间、以及该不该做")
    print("-" * 78)
    # 一周的采集量：5000 个 episode，每个 10 秒（300 帧 @30Hz），3 路相机
    ep, frames, cams = 5000, 300, 3
    total_frames = ep * frames * cams
    raw_gb = total_frames * RAW_FRAME / 1e9
    h264_gb = total_frames * W * H * 0.35 / 8 / 1e9
    hours_of_robot = ep * frames / 30 / 3600

    print("  规模：%s 个 episode × %d 帧 × %d 路相机"
          % (format(ep, ","), frames, cams))
    print("  相当于 %d 台机器人各跑一周（单台 %.0f 小时有效数据）"
          % (1, hours_of_robot))
    print()
    print("  %s%s" % (pad("", 26), "数值"))
    print("  " + "-" * 46)
    for label, val in (
        ("总帧数", "%s 帧" % format(total_frames, ",")),
        ("原始体积（RGB 未压缩）", "%.2f TB" % (raw_gb / 1000)),
        ("H.264 CRF23 之后", "%.0f GB" % h264_gb),
        ("压缩比", "%.0f 倍" % (raw_gb / h264_gb)),
    ):
        print("  %s%s" % (pad(label, 26), val))

    print()
    print("  转换耗时（读原始 + 编码 + 写盘）：")
    for mbps in (200, 800, 2000):
        hours = raw_gb * 1000 / mbps / 3600
        print("    %5d MB/s -> %5.1f 小时%s"
              % (mbps, hours, "   <- 单机单线程" if mbps == 200 else ""))
    print("  注意这里按**原始体积**算读取：编码器必须先看到原始帧。")
    print("  所以一天的采集，转换确实要跑好几小时——这是流水线的真实成本。")

    print()
    print("  由此得到三条架构约束（都由「转换很贵且会失败」推出）：")
    print("    ① **幂等**：跑到 90% 崩了，要能按 episode 粒度续跑，不能从头来")
    print("    ② **可跳过**：原始数据在对象存储里，转换产物缺失就补那一个，")
    print("       不要「一个巨型 job 处理全部」—— 失败一次全废")
    print("    ③ **记版本**：编码参数（CRF、GOP、分辨率）写进 info.json。")
    print("       改一次参数整个数据集就换了，不记版本，两次训练结果不可比")

    print()
    print("  最后一条最容易被忽视：很多人以为「视频是原始数据的无损压缩版」，")
    print("  **不是**。H.264 CRF23 丢掉的细节是永久丢失的。")
    print("  如果下游任务需要精细纹理（布料抓取、透明物体、细线缆），")
    print("  视频数据集就是错的起点，后面再怎么训练也补不回来。")


if __name__ == "__main__":
    print("=" * 78)
    print("实验 4：格式选择是物理问题，不是偏好问题")
    print("=" * 78)
    part1_layouts()
    part2_video_tradeoff()
    part3_lerobot_layout()
    part4_conversion_cost()
    print("\n" + "=" * 78)

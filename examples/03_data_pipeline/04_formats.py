#!/usr/bin/env python3
"""
实验 4：格式选择不是偏好问题，是「块大小」问题。

运行：python examples/03_data_pipeline/04_formats.py

训练时 dataloader 每一步都在做同一件事：随机抽一帧。抽一帧要搬多少字节，
由**读取粒度**决定，跟压缩率关系不大。

    [1] 五种布局抽一帧的代价：真凶是块大小，不是文件数、也不是压缩率
    [2] 视频：把 O(1) 的随机访问变成 O(GOP)
    [3] 体积估算器：分辨率 x 帧率 x 相机数
    [4] 为什么图像不能进 Parquet：一行 72 B，旁边躺着 900 KB
    [5] 三档保真：体积和随机访问代价一起算
    [6] 转换预算：多久，以及「巨型 job」为什么几乎必然失败
    [7] LeRobotDataset 的目录结构，和 RLDS 的对比
    [8] 分桶：文件数到了什么量级会出问题
    [9] 决策表

关于外部输入
------------
本仓库零依赖，不装视频编码器，所以 bpp（每像素比特数）是**外部输入**，
取自公开基准的量级，不是本脚本测出来的。但由它推出的一切——体积、倍数、
耗时、失败概率——都是算的，而且 [5] 会做敏感性检查：结论的量级不依赖
bpp 的具体取值。
"""

import math

W, H, C = 640, 480, 3
RAW_FRAME = W * H * C          # 921 600 字节
FRAME_KB = RAW_FRAME / 1024.0  # 900 KB

# 外部输入：自然图像的典型 bpp（bits per pixel）
#   原始 24.0 / PNG 无损 9.0 / JPEG q90 1.6 / H.264 CRF23 0.35 / H.265 CRF23 0.25
BPP = {"原始 RGB": 24.0, "PNG（无损）": 9.0, "JPEG q90": 1.6,
       "H.264 CRF23": 0.35, "H.265 CRF23": 0.25}
DEPEND = {"原始 RGB": "无", "PNG（无损）": "无", "JPEG q90": "无（帧内）",
          "H.264 CRF23": "GOP 内依赖", "H.265 CRF23": "GOP 内依赖"}


def width(s: str) -> int:
    return sum(2 if ord(c) > 0x2E80 else 1 for c in s)


def pad(s: str, n: int) -> str:
    return s + " " * max(0, n - width(s))


def head(*cells) -> str:
    return "  " + "".join(pad(c, w) for c, w in cells)


def mb(b):
    return b / 1e6


def frame_bytes(codec):
    return W * H * BPP[codec] / 8.0


# ---------------------------------------------------------------- [1] 布局

EPISODE_FRAMES = 100
N_EPISODES = 1000


def part1_layouts(chunk_frames=100):
    print("\n[1] 五种布局：抽一帧要读多少字节")
    print("-" * 78)
    print("  数据集：%d 个 episode × %d 帧，单帧原始 %.0f KB（%dx%d RGB），一集 %.1f MB。"
          % (N_EPISODES, EPISODE_FRAMES, FRAME_KB, W, H,
             EPISODE_FRAMES * RAW_FRAME / 1e6))
    print()

    h264 = frame_bytes("H.264 CRF23")
    layouts = [
        ("A 定长记录 + 可 seek（.npy 式）", RAW_FRAME, RAW_FRAME,
         "一个大文件完全没问题"),
        ("B HDF5 chunk=%d 帧" % chunk_frames, chunk_frames * RAW_FRAME,
         chunk_frames * RAW_FRAME,
         "读一个块 = 读整集" if chunk_frames >= EPISODE_FRAMES else "块大小 = 读放大的倍数"),
        ("C HDF5 chunk=1 帧", RAW_FRAME, RAW_FRAME,
         "最小读单位 = 一帧"),
        ("D 每帧一个文件", RAW_FRAME, RAW_FRAME,
         "字节相同，元数据另算"),
        ("E 视频 H.264 GOP=30", 30 * h264, 30 * RAW_FRAME,
         "盘读 0.40 MB，但要解 30 帧"),
    ]

    print("  %s%s%s%s" % (pad("布局", 34), pad("抽一帧读盘", 14),
                          pad("解码像素", 14), "说明"))
    print("  " + "-" * 82)
    for name, disk, decoded, note in layouts:
        print("  %s%s%s%s" % (pad(name, 34), pad("%.2f MB" % mb(disk), 14),
                              pad("%.2f MB" % mb(decoded), 14), note))

    print()
    print("  先说一个反直觉的：**A 和 C 一样快。**一段定长记录、能算出偏移，")
    print("  一个大文件没有任何问题。所以「文件太大」不是问题，「块太大」才是。")
    print()
    ratio = layouts[1][1] / layouts[2][1]
    print("  B 和 C 只差一个 chunk 参数，随机访问差 %.0f 倍：" % ratio)
    print("    B chunk=%d 帧：一帧 %.2f MB，一个 batch 64 个样本 %.2f GB"
          % (chunk_frames, mb(layouts[1][1]), 64 * layouts[1][1] / 1e9))
    print("    C chunk=1 帧　：一帧 %.2f MB，一个 batch 64 个样本 %.0f MB"
          % (mb(layouts[2][1]), mb(64 * layouts[2][1])))
    print("  而两者的**顺序读**完全一样：整个 episode 都是 %.1f MB。"
          % (EPISODE_FRAMES * RAW_FRAME / 1e6))
    print()
    print("  这就是这类问题难被发现的原因：`for f in dataset: ...` 跑得飞快，")
    print("  `random.sample(dataset, 64)` 慢 %.0f 倍，而两段代码看起来都正常。" % ratio)
    print("  dataloader 天然是随机的，所以在训练时才会暴露。")
    print()
    print("  教训：压缩率不改变这个数。B 和 C 都是无压缩原始字节，一个字节都没省。")
    print("  **I/O 放大的分母是「你真正需要的最小单位」，不是文件大小。**")
    return {"layouts": layouts}


# ---------------------------------------------------------------- [2] GOP

def part2_gop(gops=(1, 10, 30, 60, 120)):
    print("\n[2] 视频把 O(1) 的随机访问变成 O(GOP)")
    print("-" * 78)
    print("  bpp 是外部输入（本仓库不装编码器），但由它推出来的都算得出来：")
    print()
    total_frames = N_EPISODES * EPISODE_FRAMES
    print("  %s%s%s%s" % (pad("编码", 16), pad("bpp", 8), pad("单帧", 10),
                          "全量 %s 帧  相对原始" % format(total_frames, ",")))
    print("  " + "-" * 64)
    for name in ("原始 RGB", "PNG（无损）", "JPEG q90", "H.264 CRF23", "H.265 CRF23"):
        per = frame_bytes(name)
        print("  %s%s%s%s" % (pad(name, 16), pad("%.2f" % BPP[name], 8),
                              pad("%.0f KB" % (per / 1024), 10),
                              "%.2f GB  (%.0fx)" % (per * total_frames / 1e9,
                                                    RAW_FRAME / per)))

    print()
    print("  只看体积，结论是「必须用 H.264」。但随机访问的账在另一列：")
    print("  取一帧要解整个 GOP，因为帧间编码里第 k 帧依赖前面的关键帧。")
    print()
    h264 = frame_bytes("H.264 CRF23")
    print("  %s%s%s%s" % (pad("GOP", 8), pad("读盘", 12), pad("必须解码", 14),
                          "相对「只解一帧」"))
    print("  " + "-" * 60)
    for gop in gops:
        print("  %s%s%s%s" % (pad(str(gop), 8),
                              pad("%.2f MB" % mb(gop * h264), 12),
                              pad("%.1f MB" % mb(gop * RAW_FRAME), 14),
                              "%.0f 倍" % gop))
    ex = 30 if 30 in gops else gops[len(gops) // 2]
    print()
    print("  注意「读盘」和「解码」的比值永远是 %.0f 倍（= 压缩比），" % (RAW_FRAME / h264))
    if ex == 1:
        print("  GOP=1 就是帧内编码（相当于 MJPEG）：读一帧解一帧，没有浪费，")
        print("  但每一帧都是关键帧，体积基本回到原始（压缩比 ~1）。这就是「用空间换粒度」。")
    else:
        print("  随 GOP 增长的是**每次访问的绝对浪费**：GOP=%d 时为了取一帧，" % ex)
        print("  要搬 %.2f MB 的压缩字节、解 %.1f MB 的像素，而真正要的只有 0.9 MB。"
              % (mb(ex * h264), mb(ex * RAW_FRAME)))
    print()
    print("  所以视频的正确用法是**顺序读**：训练时按 episode 从头播一遍，")
    print("  每帧的解码成本摊到 GOP 上就接近 1。一旦你开始按帧随机采样，")
    print("  GOP 就变成了惩罚。这也是为什么这类数据集读视频总是「整段解码」。")

    print()
    print("  %s%s%s" % (pad("维度", 22), pad("帧内编码", 18), "帧间编码"))
    print("  " + "-" * 62)
    for a, b, c in (("解一帧的代价", "O(1)", "平均 O(GOP/2)"),
                    ("硬件解码器", "不需要", "需要，且并发数有限"),
                    ("字节级随机访问", "可以", "不行，只能按 GOP 边界切"),
                    ("无损还原", "是（PNG）", "否，高频细节永久丢失")):
        print("  %s%s%s" % (pad(a, 22), pad(b, 18), c))
    return {"bpp": dict(BPP)}


# ---------------------------------------------------------------- [3] 体积

def part3_size_estimator(hours=100.0):
    print("\n[3] 体积估算器：先算清楚要存多少，再谈格式")
    print("-" * 78)
    print("  %s 小时有效数据，原始 RGB 与 H.264 CRF23 两种存法的体积："
          % ("%.0f" % hours))
    print()
    res = [(640, 480, "480p"), (1280, 720, "720p"), (1920, 1080, "1080p")]
    print("  %s%s%s%s%s" % (pad("分辨率", 12), pad("帧率", 8), pad("相机数", 10),
                            pad("原始", 14), "H.264"))
    print("  " + "-" * 70)
    rows = []
    for w, h, tag in res:
        for fps in (30, 60):
            for cams in (1, 3):
                n = hours * 3600 * fps * cams
                raw = n * w * h * C
                vid = n * w * h * BPP["H.264 CRF23"] / 8
                rows.append((tag, fps, cams, raw, vid))
                print("  %s%s%s%s%s" % (pad(tag, 12), pad("%d" % fps, 8),
                                        pad(str(cams), 10),
                                        pad("%.2f TB" % (raw / 1e12), 14),
                                        "%.0f GB" % (vid / 1e9)))
    print()
    lo = min(r[3] for r in rows)
    hi = max(r[3] for r in rows)
    print("  同一句「%.0f 小时数据」，体积从 %.2f TB 到 %.2f TB，差 %.0f 倍——"
          % (hours, lo / 1e12, hi / 1e12, hi / lo))
    print("  只有分辨率和相机数是自变量，格式还没开始选。")
    print()
    print("  这张表应该在**采集之前**算，不是之后。因为它决定三件事：")
    print("    · 采一周要多少盘（以及上传要多久）")
    print("    · 训练时一个 epoch 要从存储读多少字节")
    print("    · 有没有资格谈「存多份」——见 [5]")
    return {"rows": rows}


# ---------------------------------------------------------------- [4] 列式

def part4_columnar():
    print("\n[4] 为什么图像不能进 Parquet：一行 72 B，旁边躺着 900 KB")
    print("-" * 78)
    print("  一帧低维数据（7 维 state + 7 维 action + 4 个索引字段，float32）：")
    print()
    fields = [("observation.state（7 维）", 7 * 4), ("action（7 维）", 7 * 4),
              ("4 个索引字段", 4 * 4)]
    total = sum(b for _, b in fields)
    print("  %s%s%s" % (pad("字段", 30), pad("字节", 10), "占行宽"))
    print("  " + "-" * 54)
    for name, b in fields:
        print("  %s%s%s" % (pad(name, 30), pad(str(b), 10), "%.0f%%" % (b / total * 100)))
    print("  %s%s" % (pad("合计", 30), pad(str(total), 10)))
    print()
    print("  索引字段是 timestamp / frame_index / episode_index / index，各 4 B。")
    print()
    print("  Parquet 是列式：读 action 那一列只碰 %.0f%% 的字节。行式（SQLite / CSV）"
          % (28 / total * 100))
    print("  读一列要把整行读出来，放大 %.1f 倍——不算夸张。" % (total / 28))
    print()
    print("  真正的灾难在图像上。假如把那一帧 640×480 RGB 也塞进同一张表：")
    print("  %s%s" % (pad("行宽", 26), "%d B" % (total + RAW_FRAME)))
    print("  %s%s" % (pad("只想读 state", 26), "%d B" % (7 * 4)))
    print("  %s%s" % (pad("读放大", 26), "%.0f 倍" % ((total + RAW_FRAME) / 28)))
    print()
    print("  这就是 LeRobotDataset 把图像拎出去走 mp4 的原因——不是因为视频压缩率高，")
    print("  而是因为**一张表里混两种量级的数据，会让所有查询按最大的那个付费**。")
    print("  分工的依据是数据形态：低维数值适合列式，高维图像适合视频。")
    return {"row_bytes": total, "with_image": total + RAW_FRAME}


# ---------------------------------------------------------------- [5] 保真

def part5_fidelity():
    print("\n[5] 三档保真：体积和随机访问代价必须一起看")
    print("-" * 78)
    print("  同一份 10 万帧的数据，四种放法各自的总账：")
    print()
    print("  %s%s%s%s" % (pad("存法", 20), pad("体积", 12),
                          pad("抽一帧的代价", 18), "能不能还原"))
    print("  " + "-" * 72)
    n = 100000
    tiers = [("原始 RGB（分帧）", RAW_FRAME, RAW_FRAME, "是"),
             ("PNG 无损", frame_bytes("PNG（无损）"), RAW_FRAME, "是"),
             ("JPEG q90", frame_bytes("JPEG q90"), RAW_FRAME, "否，有损"),
             ("H.264 GOP=30", frame_bytes("H.264 CRF23"),
              30 * RAW_FRAME, "否，有损")]
    for name, per_frame, decode, reversible in tiers:
        print("  %s%s%s%s" % (pad(name, 20), pad("%.1f GB" % (per_frame * n / 1e9), 12),
                              pad("%.1f MB" % mb(decode), 18), reversible))
    print()
    print("  两列要一起读。原始 → H.264：体积从 %.1f GB 降到 %.1f GB（省 %.0f 倍），"
          % (tiers[0][1] * n / 1e9, tiers[3][1] * n / 1e9, RAW_FRAME / tiers[3][1]))
    print("  但抽一帧从 %.1f MB 涨到 %.1f MB（贵 %.0f 倍）。**省的是空间，赔的是粒度。**"
          % (mb(tiers[0][2]), mb(tiers[3][2]), tiers[3][2] / tiers[0][2]))
    print()
    print("  注意 PNG 那一行：体积只有原始的 %.0f%%，抽一帧还是 0.9 MB（帧内编码，O(1)），"
          % (BPP["PNG（无损）"] / 24 * 100))
    print("  而且完全可逆。它是**唯一同时满足「无损 + 随机访问」的档位**。")
    print()
    print("  所以合理的做法不是选一个，是**分层**：")
    print("    · 训练：H.264 视频，顺序解码，便宜")
    print("    · 回放 / 复核 / 标注：PNG 或更高码率，随机访问不卡")
    print("    · 标定、深度、力觉：绝不走视频，用原始或无损")
    print()
    print("  分层的成本就是存储翻倍。值不值，取决于 [3] 那张表：")
    print("  480p@30×1 的 100 小时是 %.2f TB 原始，翻一倍也是 %.2f TB；"
          % (100 * 3600 * 30 * RAW_FRAME / 1e12, 2 * 100 * 3600 * 30 * RAW_FRAME / 1e12))
    print("  1080p@60×3 是 %.0f TB，翻一倍就不可能了。**分辨率决定你有没有资格分层。**"
          % (100 * 3600 * 60 * 3 * 1920 * 1080 * 3 / 1e12))

    print()
    print("  外部输入的敏感性检查（bpp 不是我们测的，所以要看结论稳不稳）：")
    print()
    print("  %s%s%s" % (pad("H.264 bpp", 14), pad("压缩比", 14), "100 小时 480p@30×1"))
    print("  " + "-" * 56)
    hours_bytes = 100 * 3600 * 30 * RAW_FRAME
    for bpp in (0.25, 0.35, 1.0, 1.6):
        per = W * H * bpp / 8
        print("  %s%s%s" % (pad("%.2f" % bpp, 14),
                            pad("%.0f 倍" % (RAW_FRAME / per), 14),
                            "%.1f GB" % (hours_bytes / RAW_FRAME * per / 1e9)))
    print()
    print("  bpp 在 0.25~1.6 之间摆动，压缩比从 96 倍到 15 倍，体积差 6 倍——")
    print("  但「视频比原始小一个数量级以上」这个结论不动。**量级可以用，倍数要标条件。**")


# ---------------------------------------------------------------- [6] 转换

def part6_conversion(episodes=5000, frames=300, cams=3, fail_per_ep=0.001):
    print("\n[6] 转换预算：多久，以及「巨型 job」为什么几乎必然失败")
    print("-" * 78)
    total_frames = episodes * frames * cams
    raw = total_frames * RAW_FRAME
    vid = total_frames * frame_bytes("H.264 CRF23")
    print("  规模：%s 个 episode × %d 帧 × %d 路相机 = %s 帧"
          % (format(episodes, ","), frames, cams, format(total_frames, ",")))
    print("  相当于单台机器人跑 %.0f 小时的有效数据。" % (episodes * frames / 30 / 3600))
    print()
    print("  %s%s" % (pad("", 28), "数值"))
    print("  " + "-" * 48)
    for label, val in (("总帧数", "%s 帧" % format(total_frames, ",")),
                       ("原始体积（RGB）", "%.2f TB" % (raw / 1e12)),
                       ("H.264 CRF23 之后", "%.0f GB" % (vid / 1e9)),
                       ("压缩比", "%.0f 倍" % (raw / vid))):
        print("  %s%s" % (pad(label, 28), val))

    print()
    print("  转换耗时（必须按**原始体积**读，编码器要先看到原始帧）：")
    print()
    print("  %s%s%s" % (pad("读+编码吞吐", 16), pad("总耗时", 12), "说明"))
    print("  " + "-" * 52)
    hours_by_speed = {}
    for mbps in (200, 800, 2000):
        hours = raw / 1e6 / mbps / 3600
        hours_by_speed[mbps] = hours
        print("  %s%s%s" % (pad("%d MB/s" % mbps, 16), pad("%.1f 小时" % hours, 12),
                            {200: "单机单线程", 800: "多线程预取", 2000: "NVMe + 并行编码"}[mbps]))
    print()
    print("  一天采集换几小时转换，这个比例本身可以接受。真正的问题是**它会失败**。")
    print()

    h = hours_by_speed[200]
    per_ep = h * 3600 / episodes
    print("  假设跑 %.1f 小时的时候挂了（磁盘满、OOM、一个坏文件）：" % (h * 0.9))
    print("    · 整体重跑：再花 %.1f 小时，之前 90%% 的算力全丢" % h)
    print("      均匀分布的失败点，期望浪费 %.1f 小时（= 总时长的一半）" % (h / 2))
    print("    · 按 episode 续跑：只重跑没完成的那一集，%.1f 秒" % per_ep)
    print("      期望浪费 %.1f 秒" % per_ep)
    print("  两者差 %.0f 倍。**「幂等」不是代码洁癖，是 %.0f 小时的差别。**"
          % (h / 2 * 3600 / per_ep, h / 2))

    print()
    print("  更狠的一条：假设单个 episode 转换失败的概率是 %.2f%%（很乐观了）。"
          % (fail_per_ep * 100))
    print("  一个处理全部 %s 集的巨型 job，成功的概率是：" % format(episodes, ","))
    print()
    print("  %s%s%s" % (pad("job 规模（集）", 16), pad("全部成功的概率", 20), ""))
    print("  " + "-" * 56)
    for n in (10, 100, 1000, 5000, 10000):
        p = 1 - fail_per_ep
        q = p ** n * 100
        print("  %s%s" % (pad(format(n, ","), 16),
                          pad("%.2f%%" % q if q >= 0.01 else "%.3f%%" % q, 20)))
    print()
    p_all = (1 - fail_per_ep) ** episodes * 100
    verdict = "也就是**几乎必然失败**" if p_all < 5 else "但规模再翻一倍就只剩一半"
    print("  %.2f%%——%s。而按 episode 处理、失败就重试，" % (p_all, verdict))
    print("  只要单集重试能成功，整体成功率就是 100%。")
    print("  这不是可靠性工程，是概率论：**把一个大 job 拆成 N 个小 job，")
    print("  失败从「必然」变成「可重试」。**")
    return {"hours": hours_by_speed, "episodes": episodes}


# ---------------------------------------------------------------- [7] LeRobot

def part7_lerobot():
    print("\n[7] LeRobotDataset 的目录结构，和那三个不起眼的文件")
    print("-" * 78)
    tree = [("meta/info.json", "总帧数、fps、每个 feature 的 shape/dtype"),
            ("meta/episodes.jsonl", "每个 episode 的起止帧号、任务文本"),
            ("meta/stats.json", "每个 feature 的 mean/std/min/max（全局）"),
            ("meta/tasks.jsonl", "任务文本去重后的表"),
            ("data/chunk-000/episode_000000.parquet", "低维：state/action/timestamp"),
            ("videos/chunk-000/observation.images.top/episode_000000.mp4", "图像走视频")]
    print("  %s%s" % (pad("路径", 62), "内容"))
    print("  " + "-" * 96)
    for p, d in tree:
        print("  %s%s" % (pad(p, 62), d))

    print()
    print("  三个设计决策，每个都对应前几节的一条结论：")
    print()
    print("  ① **图像不进 Parquet，走 mp4。** ← [4]：一行 72 B 旁边躺着 900 KB，")
    print("     所有查询都按最大的那个付费。")
    print()
    print("  ② **meta/stats.json 存全局 mean/std。**")
    print("     归一化统计量必须训练和推理完全一致。每次现算，换一批数据、")
    print("     换个采样顺序，统计量就变了——模型在真机上直接崩。")
    print("     存下来，等于把「数据预处理」变成了数据集的一部分。")
    print()
    print("  ③ **chunk-000 / chunk-001 分桶。** ← [8]：单目录文件数有阈值。")
    print()
    print("  还有一个容易忽略的细节：meta 用 **jsonl 而不是 json**。")
    print("  一行一条，可以流式读、可以 grep、可以被 map-reduce 切分。")
    print("  十万个 episode 的单个 json 数组要整体载入内存——没有必要。")

    print()
    print("  和 RLDS（TFRecord + schema）对比：")
    rows = [("存储单元", "episode 级 tfrecord 分片", "帧级 parquet + 视频"),
            ("随机访问", "受分片大小限制", "帧级，粒度更细"),
            ("生态", "TF / JAX 为主", "PyTorch 为主"),
            ("schema", "必须显式声明", "从 parquet 推断 + info.json"),
            ("灵活度", "低（改 schema 要重写）", "高（加一列即可）")]
    print("  %s%s%s" % (pad("维度", 18), pad("RLDS / TFRecord", 28), "LeRobotDataset"))
    print("  " + "-" * 70)
    for a, b, c in rows:
        print("  %s%s%s" % (pad(a, 18), pad(b, 28), c))
    print()
    print("  选择依据不是「哪个更好」，是「你的训练栈在哪边」。")
    print("  **跨生态转换是有成本的，而且往往在转换中丢元数据。**")


# ---------------------------------------------------------------- [8] 分桶

def part8_bucketing(episodes=100000, per_dir=1000):
    print("\n[8] 分桶：文件数到了什么量级会出问题")
    print("-" * 78)
    print("  %s 个 episode，每个落若干个文件（parquet + 每路相机的 mp4）。"
          % format(episodes, ","))
    print()
    print("  %s%s%s%s" % (pad("分桶策略", 18), pad("每集文件数", 12),
                          pad("单目录文件数", 16), "说明"))
    print("  " + "-" * 70)
    for label, bucket in (("不分桶", None), ("每 1000 集一个桶", per_dir)):
        for files_per_ep in (2, 4):
            if bucket is None:
                n = episodes * files_per_ep
                note = "ext4 / S3 LIST 都开始难受"
            else:
                n = bucket * files_per_ep
                note = "固定上限，与总量无关"
            print("  %s%s%s%s" % (pad(label if files_per_ep == 2 else "", 18),
                                  pad("%d 个" % files_per_ep, 12),
                                  pad(format(n, ","), 16), note))
    print()
    print("  经验阈值是「单目录一万个上下」。不是文件系统的硬限制，是三个东西的")
    print("  共同作用：目录项要线性扫描、`ls`/LIST 要一次返回全部、并行遍历的")
    print("  元数据请求数会跟着文件数涨。")
    print()
    print("  分桶还有个副作用，恰好是好事：桶号可以由 episode 号直接算出来")
    print("  （`chunk = ep // 1000`），所以「第 3721 集在哪」是一次除法和一次拼接，")
    print("  不需要任何索引。**布局本身就是索引。**")


# ---------------------------------------------------------------- [9] 结论

def part9_conclusion():
    print("\n[9] 一张决策表")
    print("-" * 78)
    rows = [("chunk=100 帧的 HDF5", "随机访问慢 100 倍", "块大小 = 100 帧，不是 1 帧"),
            ("定长记录 + 可 seek", "随机访问 O(1)", "「文件大」不是问题，「块大」才是"),
            ("视频 GOP=30", "体积 ÷69，访问 ×30", "顺序解码时惩罚接近 0"),
            ("图像进 Parquet", "读放大 32900 倍", "一行 72 B 会按 900 KB 付费"),
            ("PNG 无损", "体积 ÷2.7，访问 O(1)", "唯一「无损 + 可随机」的档位"),
            ("巨型转换 job", "成功率 0.67%", "拆成 per-episode 才能重试"),
            ("单目录 40 万文件", "LIST / 遍历退化", "分桶，桶号由 episode 号算出")]
    print(head(("做法", 24), ("代价", 24), ("为什么", 0)))
    print("  " + "-" * 78)
    for a, b, c in rows:
        print("  %s%s%s" % (pad(a, 24), pad(b, 24), c))
    print()
    print("  一句话：格式选择里唯一不能让步的是**读取粒度**，其余都是可以换算的取舍。")


if __name__ == "__main__":
    print("=" * 80)
    print("实验 4：格式选择是物理问题，不是偏好问题")
    print("=" * 80)
    part1_layouts()
    part2_gop()
    part3_size_estimator()
    part4_columnar()
    part5_fidelity()
    part6_conversion()
    part7_lerobot()
    part8_bucketing()
    part9_conclusion()
    print("\n" + "=" * 80)

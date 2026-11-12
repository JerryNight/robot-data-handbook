#!/usr/bin/env python3
"""
实验 1：rosbag2 到底是什么。生成一个结构真实的 bag，然后把它拆开看。

运行：python examples/03_data_pipeline/01_rosbag2_layout.py

这个脚本用标准库 sqlite3 生成一个 **rosbag2 同构** 的目录：

    bag_0.db3      关系库，两张表：topics / messages
    metadata.yaml  袋子的「封面」，索引与统计全在这里

payload 是合成的（不是真的 sensor_msgs CDR），但 **容器结构、表结构、
时间戳单位、分片规则、metadata 字段** 都和真实的 rosbag2 一致。
你可以用 `sqlite3 bag_0.db3 .tables` 自己验证。

看完这个脚本，你应该能回答三个问题：
    1. rosbag2 里「一帧图像」存在哪一列的哪一个字节里
    2. 为什么 metadata.yaml 说 60 秒，实际可能只有 30 秒
    3. 在没有 ROS 环境的机器上，怎么纯 Python 读一个 bag
"""

import math
import os
import random
import shutil
import sqlite3
import struct

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp_bag")

# 三个话题，三种典型频率，凑成一个「一段 60 秒抓取」
TOPICS = [
    ("/camera/image_raw/compressed", "sensor_msgs/msg/CompressedImage", 30.0),
    ("/joint_states", "sensor_msgs/msg/JointState", 100.0),
    ("/imu/data", "sensor_msgs/msg/Imu", 200.0),
]

START_NS = 1_700_000_000_000_000_000   # 2023-11-14 22:13:20 UTC
DURATION_S = 60.0


# ---------------------------------------------------------------- 输出对齐

def width(s: str) -> int:
    """终端里一个汉字占两列。不做这个换算，中文表格一定错位。"""
    return sum(2 if ord(c) > 0x2E80 else 1 for c in s)


def pad(s: str, n: int) -> str:
    return s + " " * max(0, n - width(s))


# ---------------------------------------------------------------- CDR 外壳

def cdr_wrap(payload: bytes) -> bytes:
    """
    CDR 封装头 4 字节：0x00 0x01 0x00 0x00 = 小端 + PLAIN_CDR。
    rosbag2 的 `data` 列存的就是「序列化后的消息体」，
    不同 serialization_format 的区别只在这层字节怎么解。
    """
    return b"\x00\x01\x00\x00" + payload


def fake_image(seq: int, rng: random.Random) -> bytes:
    """合成一张「压缩后」的图：真实 H.264 关键帧约 30-60KB，这里按量级生成。"""
    n = rng.randint(30_000, 38_000)
    return cdr_wrap(rng.randbytes(n))


def fake_joint_state(seq: int, t: float) -> bytes:
    """7 个关节：position / velocity / effort 各 7 个 float64 = 168 字节。"""
    pos = [0.3 * math.sin(2 * math.pi * 0.2 * t + i) for i in range(7)]
    vel = [0.3 * 2 * math.pi * 0.2 * math.cos(2 * math.pi * 0.2 * t + i) for i in range(7)]
    eff = [0.0] * 7
    body = struct.pack("<21d", *(pos + vel + eff))
    return cdr_wrap(body)


def fake_imu(seq: int, t: float, rng: random.Random) -> bytes:
    """9 个 float64 = 72 字节，另加协方差 9 个。IMU 一帧非常小。"""
    ax = 0.02 * math.sin(2 * math.pi * 3 * t) + rng.gauss(0, 0.01)
    body = struct.pack("<16d", ax, 0.0, 9.81, rng.gauss(0, 0.002), 0.0, 0.0,
                       *(0.0,) * 9, 0.0)
    return cdr_wrap(body)


# ---------------------------------------------------------------- 写 bag

def create_bag(path: str, split_s: float = 0.0, inject_bugs: bool = False,
               jump_ms: float = 50.0) -> list:
    """
    按 rosbag2 的 sqlite3 存储格式建库。

    split_s > 0 时按时间切片，产生 bag_0.db3 / bag_1.db3 ……
    真实 rosbag2 的切分条件是 max_bagfile_duration 或 max_bagfile_size。

    inject_bugs 会埋两个缺陷；jump_ms 是 imu 在 t=45s 的时钟跳变量，
    正数是钟被拨快（NTP 追上来），负数是钟被拨回（校正过头）。

    返回 [(文件名, 起始 ns, 时长 ns, 消息数), ...]
    """
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path)

    rng = random.Random(42)
    # 先生成全部消息到内存，再按片写盘（真实实现是流式的，这里为了简单）
    #
    # 注意这里有**两个时间**，它们不是一回事：
    #   t_true  真实的产生时刻，决定写入顺序（也就是 messages.id 的顺序）
    #   ns      消息自报的 timestamp，可能被时钟跳变污染
    # 真实的 bag 里 id 反映硬件到达顺序，timestamp 反映时钟 —— 时钟错了，
    # 两者就会不一致。这正是「必须按 id 查」的原因，所以排序必须用 t_true。
    msgs = []          # (t_true, topic_id, timestamp_ns, blob)
    for tid, (name, _typ, hz) in enumerate(TOPICS, start=1):
        n = int(hz * DURATION_S)
        for i in range(n):
            t = i / hz
            ns = START_NS + int(t * 1e9)
            if inject_bugs and name == "/joint_states" and 30.0 <= t < 30.2:
                continue                                    # 丢 0.2s 的关节数据
            if inject_bugs and name == "/imu/data" and t >= 45.0:
                ns += int(jump_ms * 1e6)                    # 时钟跳变
            if name.startswith("/camera"):
                blob = fake_image(i, rng)
            elif name == "/joint_states":
                blob = fake_joint_state(i, t)
            else:
                blob = fake_imu(i, t, rng)
            msgs.append((t, tid, ns, blob))

    msgs.sort(key=lambda m: m[0])          # 按真实到达顺序写入

    files = []
    if split_s <= 0:
        chunks = [msgs]
    else:
        # 真实的 rosbag2 是循环切分：每片装满 split_s 秒就换下一个文件。
        # 只切一刀在 split_s < 总时长/2 时会切出「一片 20 秒 + 一片 40 秒」，
        # 那不是分片，那是截断。
        chunks, cur = [], []
        for m_ in msgs:
            if cur and (m_[0] - cur[0][0]) >= split_s:
                chunks.append(cur)
                cur = []
            cur.append(m_)
        if cur:
            chunks.append(cur)

    for idx, chunk in enumerate(chunks):
        fname = "bag_%d.db3" % idx
        conn = sqlite3.connect(os.path.join(path, fname))
        cur = conn.cursor()
        # 这两条 CREATE 语句与 rosbag2 的 sqlite3 插件一致
        cur.execute("CREATE TABLE topics("
                    "id INTEGER PRIMARY KEY, name TEXT NOT NULL, type TEXT NOT NULL,"
                    "serialization_format TEXT NOT NULL, offered_qos_profiles TEXT NOT NULL);")
        cur.execute("CREATE TABLE messages("
                    "id INTEGER PRIMARY KEY, topic_id INTEGER NOT NULL,"
                    "timestamp INTEGER NOT NULL, data BLOB NOT NULL);")
        for tid, (name, typ, _hz) in enumerate(TOPICS, start=1):
            cur.execute("INSERT INTO topics VALUES(?,?,?,?,?)",
                        (tid, name, typ, "cdr", ""))
        cur.executemany("INSERT INTO messages(topic_id, timestamp, data) VALUES(?,?,?)",
                        [(m[1], m[2], m[3]) for m in chunk])
        conn.commit()
        conn.close()
        ns_all = [m[2] for m in chunk]
        files.append((fname, min(ns_all), max(ns_all) - min(ns_all), len(chunk)))

    write_metadata(path, files, chunks)
    return files


def write_metadata(path: str, files: list, chunks: list) -> None:
    """
    metadata.yaml 是 bag 的「封面」，rosbag2 靠它索引，不需要打开 .db3。
    它记录：每个 topic 多少条、每个分片起止时间、整体时长。
    """
    total = sum(f[3] for f in files)
    per_topic = []
    for tid, (name, typ, _hz) in enumerate(TOPICS, start=1):
        cnt = sum(1 for chunk in chunks for m in chunk if m[1] == tid)
        per_topic.append((name, typ, cnt))

    lines = [
        "rosbag2_bagfile_information:",
        "  version: 5",
        "  storage_identifier: sqlite3",
        "  duration:",
        "    nanoseconds: %d" % (files[-1][1] + files[-1][2] - files[0][1]),
        "  starting_time:",
        "    nanoseconds_since_epoch: %d" % files[0][1],
        "  message_count: %d" % total,
        "  topics_with_message_count:",
    ]
    for name, typ, cnt in per_topic:
        lines += [
            "    - topic_metadata:",
            "        name: %s" % name,
            "        type: %s" % typ,
            "        serialization_format: cdr",
            '        offered_qos_profiles: ""',
            "      message_count: %d" % cnt,
        ]
    lines += [
        '  compression_format: ""',
        '  compression_mode: ""',
        "  relative_file_paths:",
    ]
    lines += ["    - %s" % f[0] for f in files]
    lines.append("  files:")
    for fname, start, dur, cnt in files:
        lines += [
            "    - path: %s" % fname,
            "      starting_time:",
            "        nanoseconds_since_epoch: %d" % start,
            "      duration:",
            "        nanoseconds: %d" % dur,
            "      message_count: %d" % cnt,
        ]

    with open(os.path.join(path, "metadata.yaml"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------- 读 bag

def read_metadata(path: str) -> dict:
    """没有 PyYAML 也能读：metadata.yaml 是固定两层缩进的简单结构。"""
    info = {"files": []}
    cur_file = None
    with open(os.path.join(path, "metadata.yaml"), encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            s = line.strip()
            if s.startswith("message_count:") and line.startswith("  message_count"):
                info["message_count"] = int(s.split(":")[1])
            elif s.startswith("nanoseconds:") and cur_file is not None:
                pass
            elif s.startswith("- path:"):
                cur_file = s.split(":", 1)[1].strip()
                info["files"].append(cur_file)
    return info


def query_file(path: str, fname: str) -> dict:
    """直接用 sqlite3 查 bag，不需要 ROS，也不需要 rosbag2 的 Python 包。"""
    conn = sqlite3.connect(os.path.join(path, fname))
    cur = conn.cursor()
    topics = {row[0]: row[1] for row in cur.execute("SELECT id, name FROM topics")}
    out = {"topics": topics, "counts": {}, "span": {}, "sizes": {}}
    for tid, name in topics.items():
        row = cur.execute(
            "SELECT COUNT(*), MIN(timestamp), MAX(timestamp), SUM(LENGTH(data)) "
            "FROM messages WHERE topic_id=?", (tid,)).fetchone()
        out["counts"][name] = row[0]
        out["span"][name] = (row[1], row[2]) if row[0] else (0, 0)
        out["sizes"][name] = row[3] or 0
    conn.close()
    return out


# ---------------------------------------------------------------- 各部分

def part1_create():
    print("=" * 78)
    print("实验 1：rosbag2 的容器结构，以及它藏不住的三个问题")
    print("=" * 78)

    print("\n[1] 生成一个 60 秒的 bag（3 个话题）")
    print("-" * 78)
    files = create_bag(OUT)
    print("  目录 %s/" % os.path.basename(OUT))
    for name in sorted(os.listdir(OUT)):
        size = os.path.getsize(os.path.join(OUT, name))
        print("    %-14s %10.2f MB" % (name, size / 1e6))

    print("\n  三个话题各自的频率和数据量：")
    print("  %s%8s%10s%12s" % (pad("topic", 34), "Hz", "条数", "总量"))
    print("  " + "-" * 64)
    q0 = query_file(OUT, "bag_0.db3")
    for name, _typ, hz in TOPICS:
        print("  %s%8.0f%10d%10.2fMB"
              % (pad(name, 34), hz, q0["counts"][name], q0["sizes"][name] / 1e6))

    q = query_file(OUT, "bag_0.db3")
    cam_name = TOPICS[0][0]
    cam_cnt, cam_size = q["counts"][cam_name], q["sizes"][cam_name]
    all_cnt = sum(q["counts"].values())
    all_size = sum(q["sizes"].values())
    print("\n  图像那一行：条数只占 %.1f%%（%d / %d），"
          % (cam_cnt / all_cnt * 100, cam_cnt, all_cnt))
    print("  体积却占 %.1f%%（%.1fMB / %.1fMB）。"
          % (cam_size / all_size * 100, cam_size / 1e6, all_size / 1e6))
    print("  这就是为什么「录了 10 分钟」的重量几乎全是视频，")
    print("  也是为什么数据平台的成本模型必须以图像为基准。")


def part2_schema():
    print("\n[2] 拆开 .db3：只有两张表")
    print("-" * 78)
    conn = sqlite3.connect(os.path.join(OUT, "bag_0.db3"))
    for (sql,) in conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table'"):
        # 按逗号断行，让两条 DDL 在终端里读得出来
        cols = [c.strip() for c in sql.replace("(", "(\n      ").split(",")]
        print("  " + ",\n      ".join(cols))
        print()
    print("  topics：话题元信息，一个话题一行")
    print("  messages：每帧一行，timestamp 单位是 **纳秒**，data 是序列化后的字节")
    print("\n  也就是说：rosbag2 = 一个 SQLite 文件 + 一张时间序列大表。")
    print("  没有列式存储、没有按 topic 分区、图像和 IMU 挤在同一张表里。")

    row = conn.execute(
        "SELECT id, topic_id, timestamp, LENGTH(data) FROM messages "
        "WHERE topic_id=1 LIMIT 3").fetchall()
    print("\n  前三条图像消息（截断到 SQL 层能看到的字段）：")
    print("  %8s %9s %20s %10s" % ("id", "topic_id", "timestamp(ns)", "字节"))
    for r in row:
        print("  %8d %9d %20d %10d" % r)
    print("\n  「一帧图像」= data 列里的一个 BLOB。想看像素？")
    print("  先按 topics.type 找到消息定义，再按 CDR 规则解 data ——")
    print("  中间没有任何元数据告诉你这张图多少分辨率、什么编码。")
    print("  这些信息**只存在于字节流内部**。")
    conn.close()


def part3_metadata_vs_truth():
    print("\n[3] metadata.yaml 说 60 秒，实际可能只有一半")
    print("-" * 78)
    split_dir = OUT + "_split"
    files = create_bag(split_dir, split_s=30.0)
    print("  开启按 30 秒分片后，目录里变成：")
    for fname, start, dur, cnt in files:
        print("    %-12s %6.1fs  %7d 条" % (fname, dur / 1e9, cnt))

    meta = read_metadata(split_dir)
    print("\n  metadata.yaml 的 message_count = %d（两个文件之和）"
          % meta["message_count"])

    only_first = query_file(split_dir, "bag_0.db3")
    first_cnt = sum(only_first["counts"].values())
    print("  但如果你只上传了 bag_0.db3：实际 %d 条，**丢了一半**。" % first_cnt)
    print("  更糟的是：bag_0.db3 自身完全自洽 —— 表结构完整、时间戳连续、")
    print("  能正常回放。它只是安静地少了 30 秒。")
    print("\n  这是数据平台最常见的一类事故：")
    print("  「录制成功」的判据是「文件写完了」，不是「数据完整」。")
    print("  唯一可靠的校验是拿 metadata 的预期值和实际行数/时间跨度对账。")

    span = []
    for name in only_first["counts"]:
        s, e = only_first["span"][name]
        if s:
            span.append((e - s) / 1e9)
    print("\n  对账的写法（3 行）：")
    print("    expected = metadata['message_count']")
    print("    actual   = SELECT COUNT(*) FROM messages")
    print("    assert actual == expected, 'bag 不完整，拒绝入库'")
    print("  实际时间跨度 %.1fs，而 metadata 声称 %.0fs —— 一眼就能看出来。"
          % (max(span), DURATION_S))
    shutil.rmtree(split_dir)


def part4_quality_check():
    print("\n[4] 回放质检：三类缺陷，三种检测方法")
    print("-" * 78)
    bad = OUT + "_bad"
    create_bag(bad, inject_bugs=True)

    conn = sqlite3.connect(os.path.join(bad, "bag_0.db3"))
    print("  埋了两个缺陷，再做一个对照实验：")
    print("    ① /joint_states 在 t=30.0~30.2s 丢了 20 帧（模拟 DDS 丢包）")
    print("    ② /imu/data 在 t=45s 之后时间戳整体 +50ms（模拟时钟跳变）")
    print()

    print("  检测 ①：相邻消息间隔超过 1.5 倍标称周期 -> 判为空洞")
    tid = 2
    ts = [r[0] for r in conn.execute(
        "SELECT timestamp FROM messages WHERE topic_id=? ORDER BY timestamp", (tid,))]
    period = 1e9 / 100.0
    gaps = []
    for a, b in zip(ts, ts[1:]):
        if (b - a) > 1.5 * period:
            gaps.append(((a - START_NS) / 1e9, (b - a) / 1e6))
    for t, dt in gaps:
        print("    t=%.2fs 处空洞 %.0f ms（正常 10.0ms）-> 丢了约 %d 帧"
              % (t, dt, round(dt / 10.0) - 1))

    print("\n  检测 ②：相邻时间戳的增量超出容差 -> 时钟跳变")
    tid = 3
    ts = [r[0] for r in conn.execute(
        "SELECT timestamp FROM messages WHERE topic_id=? ORDER BY id", (tid,))]
    jump = [(i, (b - a) / 1e6) for i, (a, b) in enumerate(zip(ts, ts[1:]))
            if abs((b - a) - 5e6) > 1e6]
    for i, d in jump[:3]:
        print("    第 %d 条到第 %d 条：间隔 %.1f ms（应为 5.0ms）" % (i, i + 1, d))
    print("    （钟被拨快 50ms：后续消息整体前移，间隔从 5ms 变成 55ms）")

    print("\n  检测 ③：同一件事换成「钟被拨回 50ms」，排序方式就开始骗人了")
    back = OUT + "_back"
    create_bag(back, inject_bugs=True, jump_ms=-50.0)
    conn2 = sqlite3.connect(os.path.join(back, "bag_0.db3"))
    for order in ("id", "timestamp"):
        sql = "SELECT timestamp FROM messages WHERE topic_id=? ORDER BY " + order
        ts2 = [r[0] for r in conn2.execute(sql, (tid,))]
        bad_gaps = [(b - a) / 1e6 for a, b in zip(ts2, ts2[1:]) if abs((b - a) - 5e6) > 1e6]
        kinds = {}
        for g in bad_gaps:
            k = round(g, 1)
            kinds[k] = kinds.get(k, 0) + 1
        desc = "  ".join("%+.1f ms × %d" % (k, v) for k, v in sorted(kinds.items()))
        print("    ORDER BY %-9s -> %2d 处异常：%s" % (order, len(bad_gaps), desc))
    conn2.close()
    shutil.rmtree(back)

    print("\n    按 id：一条 -45ms 的负间隔 —— 一眼看出钟被拨回了 50ms（50 - 5 = 45）。")
    print("    按 timestamp：变成 10 处 0.0ms 的并列 —— 排序把回退摊平了，")
    print("    检测器看到的只是「有点挤」，看不出这是一次时钟事件。")
    print()
    print("    所以规则是：**查 id，不查 timestamp**。id 是写入顺序（硬件到达），")
    print("    timestamp 是被怀疑的对象本身 —— 用被怀疑的东西去排序，")
    print("    等于让它自己决定证据的顺序。")
    print()
    print("    顺带注意检测 ②：钟被拨快时两种排序结果完全一样（都是 55ms），")
    print("    所以「排序无关紧要」这个错觉很容易形成 —— 直到第一次遇到回退。")
    print("    单向跳变是单调的，回退不是。质检规则要对两者都成立。")

    print("\n  这三类缺陷的共同点：**都不会让录制失败**。")
    print("  rosbag2 照常写盘、照常关闭、照常报 success。")
    print("  但它们会变成训练集里 200ms 的动作-观测错配，或者 20 帧的静默丢帧。")
    conn.close()
    shutil.rmtree(bad)


def part5_gap_threshold(period=10.0):
    """period 是标称采样周期（ms）。改成 5.0 就是 200 Hz，窗口会窄一半。"""
    print("\n[5] 丢帧阈值：1.5 倍周期这个数是怎么夹出来的")
    print("-" * 78)
    print("  上一节的检测用了一个常数：间隔 > 1.5 × 标称周期就算空洞。")
    print("  这个 1.5 不是拍脑袋，它是两条约束夹出来的一个窗口：")
    print()
    print("    下界：不能小到被采样抖动触发（误报）")
    print("    上界：不能大到漏掉「只丢一帧」这种最该抓的情况")
    print()
    print("  丢一帧 -> 间隔变成 2 × 周期。所以阈值必须 < 2，这是硬上界。")
    print("  下界取决于抖动有多大。下面按三档抖动各跑 10000 个间隔：")
    print()

    n = 10000
    rng = random.Random(7)

    def intervals(sigma, n, drop=0):
        """n 个相邻间隔（ms）。drop>0 表示每次间隔都相当于丢了 drop 帧。

        关键：抖动作用在**时间戳**上，而间隔是两个时间戳之差，
        所以间隔上的噪声 std 是 √2·sigma，不是 sigma。
        写成「period + 一个噪声」会低估误报率，差 √2 倍。
        """
        out = []
        prev = rng.gauss(0, sigma)
        for _ in range(n):
            cur = rng.gauss(0, sigma)
            out.append(period * (1 + drop) + (cur - prev))
            prev = cur
        return out

    print("  %s%s" % (pad("抖动 std", 14), "阈值倍数 -> 误报次数 / 万帧"))
    print("  " + "-" * 66)
    print("  %s%12s%12s%12s%12s" % (pad("", 14), "1.2x", "1.5x", "1.8x", "2.0x"))
    for sigma in (0.5, 1.5, 5.0):
        iv = intervals(sigma, n)
        cells = []
        for k in (1.2, 1.5, 1.8, 2.0):
            hits = sum(1 for x in iv if x > k * period)
            cells.append("%d" % hits)
        print("  %s%12s%12s%12s%12s"
              % (pad("%.1f ms" % sigma, 14), cells[0], cells[1], cells[2], cells[3]))

    print()
    if abs(period - 10.0) < 1e-9:
        print("  读法（三个环境各一条）：")
        print("  · 抖动 0.5 ms（有线 / 实时内核）：1.2 倍只误报 17 次/万帧，")
        print("    可以用，而且漏检率 0% —— 这是唯一能这么激进的情况")
        print("  · 抖动 1.5 ms（普通 Linux + USB 相机）：1.2 倍误报 1710 次，")
        print("    必须退到 1.5 倍（误报 100 次，漏检 0.9%）—— 工程上默认 1.5 就是这里来的")
        print("  · 抖动 5 ms（WiFi / 非实时）：1.5 倍误报 2407 次，退到 2.0 倍")
        print("    误报还有 750 次，而漏检已经 50% —— 没有任何阈值能用")
    print()
    print("  决定窗口宽度的不是抖动，也不是周期，而是两者的比值 σ/T：")
    for sig in (0.5, 1.5, 5.0):
        print("    σ/T = %.3f  ->  %s"
              % (sig / period,
                 "宽裕" if sig / period < 0.10 else
                 ("勉强" if sig / period < 0.20 else "不可用")))
    print()

    # 漏检：丢一帧时，间隔 20ms 是否还能被抓住
    print("  漏检率（丢 1 帧，间隔 2×周期，看还能不能判出来）：")
    print("  %s%12s%12s%12s" % (pad("抖动 std", 14), "1.5x", "1.8x", "2.0x"))
    print("  " + "-" * 50)
    for sigma in (0.5, 1.5, 5.0):
        iv = intervals(sigma, n, drop=1)
        cells = []
        for k in (1.5, 1.8, 2.0):
            missed = sum(1 for x in iv if x <= k * period)
            cells.append("%.1f%%" % (missed / n * 100))
        print("  %s%12s%12s%12s" % (pad("%.1f ms" % sigma, 14), cells[0], cells[1], cells[2]))

    print()
    print("  结论：抖动一旦接近周期的一半，用时间戳做丢帧检测就失效了。")
    print("  5 ms 抖动下，阈值窗口 (1.5, 2) 名义上还在，但窗口两端同时不可用：")
    fa15 = 0.5 * math.erfc((1.5 - 1) * period / (2 * 5.0))
    print("  1.5 倍每万帧误报约 %.0f 次（%.0f%%），2.0 倍漏掉一半的真丢帧。"
          % (fa15 * 10000, fa15 * 100))
    print("  这个检测器根本不能上线。")
    print()
    print("  真正可靠的判据不是时间，是**序号**：")
    print("  消息里带一个自增 seq，丢帧就等于 seq 不连续，与抖动无关。")
    print("  ROS 2 的 sensor_msgs 去掉了 header.seq，所以这一点必须自己做 ——")
    print("  在录制侧加一个计数列，或者在 DDS 层读 sample_lost 统计。")
    print("  这是「质检规则要白名单式」的具体体现：预期什么，就断言什么，")
    print("  不要用时间来推断「应该有几帧」。")


def part6_random_access():
    print("\n[6] 随机访问：想取一个关节角，要读多少字节")
    print("-" * 78)
    conn = sqlite3.connect(os.path.join(OUT, "bag_0.db3"))
    # 实测一行的平均字节数，而不是按结构推算
    avg_blob, cnt = conn.execute(
        "SELECT AVG(LENGTH(data)), COUNT(*) FROM messages WHERE topic_id=2").fetchone()
    conn.close()

    # 行开销：id / topic_id / timestamp 三个整数列，SQLite 各行占 8 字节左右
    row = avg_blob + 24
    print("  /joint_states 一行：payload 平均 %.0f B + 行开销 24 B = %.0f B"
          % (avg_blob, row))
    print("  这一行里，21 个 float64（position 7 + velocity 7 + effort 7）")
    print("  只有 168 字节是「值」，其余是外壳。")
    print()
    print("  现在假设你要做一件很常见的事：把 1 万个 episode 的第 3 个关节角")
    print("  拉成一条时间序列，算它的分布。")
    print()
    rows = 10000 * 60 * 100                  # 60s × 100Hz × 1 万条
    need = 8                                  # 你真正要的：1 个 float64
    print("    总行数                %s 行" % f"{rows:,}")
    print("    行式（rosbag2 现状）   %s × %.0f B = %.1f GB"
          % (f"{rows:,}", row, rows * row / 1e9))
    print("    列式（只读那一列）     %s × %d B = %.1f GB"
          % (f"{rows:,}", need, rows * need / 1e9))
    print("    读放大                 %.1f 倍" % (row / need))
    print()
    print("  这还只是关节角。换成一个更常见的需求 —— 统计图像的平均亮度：")
    print()
    conn = sqlite3.connect(os.path.join(OUT, "bag_0.db3"))
    img_b, img_cnt = conn.execute(
        "SELECT AVG(LENGTH(data)), COUNT(*) FROM messages WHERE topic_id=1").fetchone()
    conn.close()
    print("    一张压缩图 %.1f KB，要算亮度必须解出全部像素" % (img_b / 1024))
    print("    而如果转换时顺手把 mean/std 存成两列，读它就是 16 字节")
    print("    读放大                 %.0f 倍" % (img_b / 16))
    print()
    print("  这就是第四章要讲的 stats.json 为什么必须预先算好：")
    print("  不是「顺便优化」，而是行式存储下这个查询根本跑不动。")


def part7_summary():
    print("\n[7] 所以 rosbag2 为什么不能直接当训练集")

    print("-" * 78)
    rows = [
        ("随机访问单帧", "差：一个 BLOB 要整行读，无法只取第 3 维", "HDF5 / 分片 Parquet"),
        ("跨 episode 查询", "差：每个 bag 是独立库，没有全局索引", "数据集清单 + 索引文件"),
        ("压缩", "无：ZSTD 只能压整个文件", "按模态分别编码"),
        ("增量写入", "好：append 一行即可", "（训练集不需要）"),
        ("元数据", "只有 topic 名和类型", "需要标定、任务、成功率、语言指令"),
        ("版本与血缘", "无", "内容哈希 + manifest"),
    ]
    print("  %s%s%s" % (pad("需求", 16), pad("rosbag2 的能力", 44), "训练集需要什么"))
    print("  " + "-" * 78)
    for a, b, c in rows:
        print("  %s%s%s" % (pad(a, 16), pad(b, 44), c))

    print("\n  结论不是「rosbag2 不好」——它是**录制格式**，设计目标是在机器人上")
    print("  稳定地、低延迟地把字节写进磁盘。用训练集的眼光要求它是错位的。")
    print("  正确的做法是：rosbag2 负责采集，转换层负责重排成列式 + 索引 + 元数据。")
    print("\n" + "=" * 78)


def cleanup():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)


if __name__ == "__main__":
    try:
        part1_create()
        part2_schema()
        part3_metadata_vs_truth()
        part4_quality_check()
        part5_gap_threshold()
        part6_random_access()
        part7_summary()
    finally:
        cleanup()

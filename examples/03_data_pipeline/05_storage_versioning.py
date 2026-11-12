#!/usr/bin/env python3
"""
实验 5：内容寻址、去重、血缘，以及「数据是哪一份」。

运行：python examples/03_data_pipeline/05_storage_versioning.py

机器人数据有两个难受的性质：**大**、**重复**。遥操作采集时，机器人有大量
「回到起始位」「等待下一步」的静止段，这些帧在不同 episode 之间几乎一样。

    [1] 去重的上限由块大小决定：小块抓得多，但块数爆炸
    [2] 插入一帧就崩：固定分块 vs 内容定义分块
    [3] 元数据不是免费的：块数换成索引体积和请求数
    [4] 哈希该截多少位：10 万 episode 下的碰撞概率
    [5] 血缘：改一个参数，传播到哪里，要重算什么
    [6] 校验要花时间：全量 sha256、增量 Merkle、以及校验周期
    [7] 子集：物理复制还是引用
    [8] 冷热分层：100 TB 的月账单
    [9] manifest：一个 bit 就能抓到，但它只管字节
    [10] 决策表

关于外部输入
------------
[3] 的「每块索引 32 B」、[6] 的「sha256 1.5 GB/s」、[8] 的对象存储单价，
都是**外部输入**，不是本脚本测出来的。凡是由它们推出来的结论，都在
对应小节标了敏感性：量级不依赖具体取值，但绝对值会。
"""

import hashlib
import json
import os
import random
import shutil

GB = 1e9
TB = 1e12


def width(s: str) -> int:
    return sum(2 if ord(c) > 0x2E80 else 1 for c in s)


def pad(s: str, n: int) -> str:
    return s + " " * max(0, n - width(s))


def head(*cells) -> str:
    return "  " + "".join(pad(c, w) for c, w in cells)


def _fmt_p(p: float) -> str:
    """概率的可读格式：大的用百分数，小的用科学计数。"""
    if p >= 1.0:
        return "100%"
    if p >= 0.001:
        return "%.2f%%" % (p * 100)
    return "%.1e" % p


FRAME = 16 * 1024          # 一「帧」16 KB，压缩后的量级

# 所有 episode 共用同一段静止内容：模拟「同一个工位反复采集」。
# 如果每个 episode 的静止段都不一样（多个工位、光照在变），去重率会打折 ——
# 这也是为什么去重率必须按任务实测，不能引用别人的数字。
_IDLE = bytes(((i * 31 + 7) % 251) for i in range(FRAME))


def episode_parts(seed, n_frames, idle_ratio, frame=FRAME, group=10):
    """合成一个 episode 的帧列表。

    idle 段用**同一份内容**，模拟「机器人静止 / 回原点」。这正是真实数据里
    重复量的主要来源，也是去重能省钱的唯一原因。

    两个刻意的设计：

    - 静止帧是**连续的 k 帧**（每 group 帧一组），不是零散单帧。遥操作时机器人
      真的会停在原地等几秒，不会隔一帧停一帧。
    - 静止段在一组内的**起点随机**——真实数据里，同一段静止画面在不同 episode
      中的字节偏移不一样。如果偏移总是对齐的，块大小对去重率就没有影响，
      实验会得出「块大小无所谓」的错误结论。
    """
    rng = random.Random(seed)
    k = int(round(idle_ratio * group))       # 每组里连续静止的帧数
    parts = []
    for g in range(0, n_frames, group):
        n = min(group, n_frames - g)
        kk = min(k, n)
        start = rng.randint(0, n - kk) if n > kk else 0
        for j in range(n):
            if start <= j < start + kk:
                parts.append(_IDLE)                      # 完全相同的静止帧
            else:
                parts.append(rng.randbytes(rng.randint(frame * 3 // 4, frame * 5 // 4)))
    return parts


def make_episode(seed, n_frames, idle_ratio, frame=FRAME):
    return b"".join(episode_parts(seed, n_frames, idle_ratio, frame))


def fixed_chunks(blob, size):
    """固定分块：每 size 字节一刀。快、可预测、对插入极其敏感。"""
    return [blob[i:i + size] for i in range(0, len(blob), size)]


# Gear 表：内容定义分块（CDC）用的滚动哈希系数。
# 用固定种子生成，保证每次运行结果完全一致。
_grng = random.Random(20261103)
GEAR = [_grng.getrandbits(64) for _ in range(256)]
M64 = (1 << 64) - 1


def cdc_chunks(blob, bits=12, min_size=2048, max_size=65536):
    """内容定义分块（FastCDC 的极简版）。

    h = (h << 1) + GEAR[byte]，当 h 的低 bits 位全 0 时切一刀。
    切口由**内容**决定，所以前面插入字节不会移动后面的切口。
    """
    mask = (1 << bits) - 1
    out = []
    i, n = 0, len(blob)
    while i < n:
        h = 0
        j = i
        end = min(i + max_size, n)
        while j < end:
            h = ((h << 1) + GEAR[blob[j]]) & M64
            j += 1
            if j - i >= min_size and (h & mask) == 0:
                break
        out.append(blob[i:j])
        i = j
    return out


def dedup_stats(chunks):
    """返回 (原始字节, 唯一字节, 块数, 唯一块数)。"""
    seen = {}
    total = 0
    for c in chunks:
        total += len(c)
        seen[hashlib.sha256(c).digest()] = len(c)
    return total, sum(seen.values()), len(chunks), len(seen)


# ---------------------------------------------------------------- [1] 去重

def part1_dedup(n_ep=12, n_frames=60, idle_ratios=(0.0, 0.2, 0.4, 0.6),
                chunk_sizes=(4096, 16384, 65536, 262144, 1048576)):
    print("\n[1] 去重的上限由块大小决定")
    print("-" * 78)
    blobs = {r: [make_episode(ep, n_frames, r) for ep in range(n_ep)]
             for r in idle_ratios}
    one_ep = len(blobs[idle_ratios[0]][0])
    print("  %d 个 episode × %d 帧（帧长 12~20 KB）= 每集 %.1f MB，总 %.0f MB。"
          % (n_ep, n_frames, one_ep / 1e6, n_ep * one_ep / 1e6))
    print("  按块算 sha256，相同哈希只存一份。")
    print()

    rows = []
    for size in chunk_sizes:
        cells = [size]
        for r in idle_ratios:
            total = 0
            seen = {}                     # 跨 episode 共享：这一步不能写在循环里
            for blob in blobs[r]:
                for c in fixed_chunks(blob, size):
                    total += len(c)
                    seen[hashlib.sha256(c).digest()] = len(c)
            cells.append((1 - sum(seen.values()) / total) * 100)
        rows.append((size, cells[1:]))

    cols = [("idle %.0f%%" % (r * 100) if i == 0 else "%.0f%%" % (r * 100), 12)
            for i, r in enumerate(idle_ratios)]
    print(head(("块大小", 12), *cols))
    print("  " + "-" * (18 + 12 * len(idle_ratios)))
    for size, saved in rows:
        print("  %s%s" % (pad("%d KB" % (size // 1024), 12),
                          "".join(pad("%.0f%%" % v, 12) for v in saved)))

    print()
    print("  两个方向都要读：")
    print("    · idle 越高省得越多 —— 但这是**任务属性**，不是技术选择")
    print("    · 块越大省得越少 —— 块一旦接近静止段长度，整块里只要有一帧在动")
    print("      就独一无二，那些静止帧一个都没抓到")
    print()
    i40 = idle_ratios.index(0.4) if 0.4 in idle_ratios else len(idle_ratios) // 2
    best, worst = rows[0][1][i40], rows[-1][1][i40]
    print("  同样 %.0f%% 的静止帧：%d KB 块省 %.0f%%，%d KB 块省 %.0f%%。"
          % (idle_ratios[i40] * 100, chunk_sizes[0] // 1024, best,
             chunk_sizes[-1] // 1024, worst))
    print("  数据一个字节都没差，**唯一的变化是切块粒度**。")

    print()
    print("  0% 那一行所有块大小都省 0%——去重不会凭空变出压缩。")
    print("  它只做一件事：把**字节级相同**的内容只存一次。")
    print()
    print("  但收益递减得很快：%d KB → %d KB 掉 %.0f 个百分点，%d KB 之后基本归零。"
          % (chunk_sizes[0] // 1024, chunk_sizes[1] // 1024,
             abs(rows[0][1][i40] - rows[1][1][i40]), chunk_sizes[2] // 1024))
    print("  而 [3] 的成本按块数线性上涨。**所以块大小不是越小越好，它是个平衡点，**")
    print("  **而不是「越细越省」的单向旋钮。**")
    return {"rows": rows, "idle": list(idle_ratios)}


# ---------------------------------------------------------------- [2] shift

def part2_shift(n_frames=60, insert_at=10):
    print("\n[2] 插入一帧就崩：固定分块 vs 内容定义分块")
    print("-" * 78)
    parts = episode_parts(7, n_frames, 0.3)
    a = b"".join(parts)
    ins = make_episode(999, 1, 0.0)
    off = sum(len(p) for p in parts[:insert_at])
    b = a[:off] + ins + a[off:]
    print("  A：%d 帧 / %.2f MB。B：在 A 的第 %d 帧后插入 1 帧 / %d KB。"
          % (n_frames, len(a) / 1e6, insert_at, len(ins) // 1024))
    print("  只有 %d KB 不同，占总量 %.1f%%。" % (len(ins) // 1024, len(ins) / len(a) * 100))
    print()

    print(head(("分块方式", 20), ("A 块数", 10), ("B 块数", 10),
               ("重合块", 12), ("B 去重后", 12), ("多存", 0)))
    print("  " + "-" * 78)
    results = []
    for label, fn, kw in (("固定 16 KB", fixed_chunks, {"size": FRAME}),
                          ("内容定义（CDC）", cdc_chunks, {"bits": 12})):
        ca, cb = fn(a, **kw), fn(b, **kw)
        ha = {hashlib.sha256(c).digest(): len(c) for c in ca}
        hb = {hashlib.sha256(c).digest(): len(c) for c in cb}
        common = set(ha) & set(hb)
        uniq = sum(hb.values())
        shared = sum(hb[k] for k in common)
        extra = uniq - shared
        results.append((label, len(ca), len(cb), len(common), uniq, extra))
        print("  %s%s%s%s%s%s" % (pad(label, 20), pad(str(len(ca)), 10),
                                  pad(str(len(cb)), 10), pad(str(len(common)), 12),
                                  pad("%.2f MB" % (uniq / 1e6), 12),
                                  "%.2f MB" % (extra / 1e6)))

    print()
    fx, cdc = results[0], results[1]
    print("  固定分块：插入点之后的切口全部右移 16 KB，%.0f%% 的块变成「新的」——"
          % (100 - fx[3] / fx[1] * 100))
    print("  虽然内容只差了 1 帧。CDC：切口由内容决定，只有插入点附近一两个块变化。")
    print()
    print("  多存 %.2f MB vs %.2f MB，**%.0f 倍**。分母只有 16 KB，但它污染了整个后缀。"
          % (fx[5] / 1e6, cdc[5] / 1e6, fx[5] / max(cdc[5], 1)))
    print()
    print("  这就是「固定分块只能给只追加的数据用」的原因。而机器人数据几乎总是")
    print("  在中间被编辑：删掉失败片段、补采一段、重新对齐时间轴。")
    print("  **只要会插入或删除，固定分块的去重率就会在第一次编辑后腰斩。**")
    return {"rows": results}


# ---------------------------------------------------------------- [3] 元数据

def part3_metadata(total_tb=1.0, index_bytes=32, req_us=50,
                   chunk_sizes=(4096, 16384, 65536, 262144, 1048576)):
    print("\n[3] 元数据不是免费的：块数换成索引体积和请求数")
    print("-" * 78)
    print("  %.0f TB 数据，每块一条索引（%d B：16 B 哈希 + 8 B 偏移 + 8 B 长度）。"
          % (total_tb, index_bytes))
    print("  外部输入：索引项大小、每次请求 %.0f µs 的固定开销。" % req_us)
    print()
    print(head(("块大小", 12), ("块数", 16), ("索引体积", 14),
               ("读一个 92 MB 集", 18), ("请求开销", 0)))
    print("  " + "-" * 82)
    rows = []
    for size in chunk_sizes:
        n = total_tb * TB / size
        idx = n * index_bytes
        reqs = 92.2e6 / size
        overhead = reqs * req_us / 1e6
        rows.append((size, n, idx, reqs, overhead))
        print("  %s%s%s%s%s" % (pad("%d KB" % (size // 1024), 12),
                                pad("%s 块" % format(int(n), ","), 16),
                                pad("%.1f GB" % (idx / GB), 14),
                                pad("%s 次" % format(int(reqs), ","), 18),
                                "%.2f 秒" % overhead))

    print()
    print("  4 KB 块：去重抓得最细（[1]），但每存 1 TB 要多付 %.1f GB 索引，"
          % (rows[0][2] / GB))
    print("  读一集要 %s 次请求，光固定开销就 %.1f 秒。"
          % (format(int(rows[0][3]), ","), rows[0][4]))
    print("  1 MB 块：请求只有 %d 次，但去重率已经归零（见 [1]）。" % rows[-1][3])
    print()
    print("  真实系统取中间：CDC 平均 8~64 KB。**块大小不是压缩参数，是索引参数。**")
    print("  它同时决定三件事：去重率、索引体积、每次读的请求数。三个方向互相拉扯，")
    print("  所以「去重率 60%」这种宣传数字，一定要问清楚块大小。")
    print()
    print("  还有一个常被忽略的：把每块存成一个对象（S3 的一个 key），")
    print("  1 TB / 4 KB = %s 个对象。**对象数本身就是配额**，很多系统先撞这个墙。"
          % format(int(total_tb * TB / 4096), ","))
    return {"rows": rows}


# ---------------------------------------------------------------- [4] 哈希位数

def part4_hash_bits(sizes=(10000, 100000, 1000000), bits=(8, 10, 12, 14, 16)):
    print("\n[4] 哈希该截多少位：碰撞概率是算出来的，不是拍的")
    print("-" * 78)
    print("  清单里存 sha256 的前 b 个十六进制字符。生日悖论下，n 个对象至少")
    print("  一对碰撞的概率约为 n² / (2 · 2^(4b))。")
    print()
    print(head(("截断位数", 14), ("空间", 14),) +
          "".join(pad("%s 条" % format(n, ","), 16) for n in sizes))
    print("  " + "-" * 78)
    for b in bits:
        space = 16 ** b
        cells = [min(1.0, n * n / (2.0 * space)) for n in sizes]
        print("  %s%s%s" % (pad("%d 位" % b, 14),
                            pad("%.1e" % space, 14),
                            "".join(pad(_fmt_p(p), 16) for p in cells)))

    print()
    print("  12 位（48 bit）在 10 万条下是 1e-5 量级——够用。")
    print("  但 8 位（32 bit）在 1 万条下就有 %.1f%%，10 万条直接 100%%：**必然撞**。"
          % (10000 * 10000 / (2.0 * 16 ** 8) * 100))
    print("  这就是 git 短哈希冲突的翻版，只是数据集的条目数比代码仓库大三个数量级。")
    print()
    target = 1e-6
    for n in sizes:
        need = 0
        while n * n / (2.0 * 16 ** need) > target:
            need += 1
        print("  %s 条要做到碰撞概率 < 1e-6，至少要 %d 个十六进制字符（%d bit）。"
              % (pad(format(n, ","), 10), need, need * 4))
    print()
    print("  推论：**短哈希只能当标签用，不能当身份证。**清单里既可以存 12 位")
    print("  方便人眼比对，也存全长 64 位用于校验——两者不冲突，各管一件事。")
    return {"sizes": list(sizes), "bits": list(bits)}


# ---------------------------------------------------------------- [5] 血缘

def part5_lineage(cal_h=3.0, conv_h=5.8, subset_h=0.2):
    print("\n[5] 血缘：改一个参数，传播到哪里，要重算什么")
    print("-" * 78)

    def h(*parts):
        m = hashlib.sha256()
        for p in parts:
            m.update(repr(p).encode())
        return m.hexdigest()[:12]

    def build(crf=23, gop=30, cam="intrinsics_v3", eps=100, raw="bag-v1"):
        r = h("raw", raw)
        c = h(r, {"cam": cam, "hand_eye": "2026-10-28"})
        v = h(c, {"codec": "h264", "crf": crf, "gop": gop, "res": "640x480"})
        s = h(v, {"filter": "success_only", "eps": eps})
        return {"raw": r, "calibrated": c, "converted": v, "subset": s}

    base = build()
    names = ("raw", "calibrated", "converted", "subset")
    cost = {"raw": 0.0, "calibrated": cal_h, "converted": conv_h, "subset": subset_h}
    order = list(names)

    variants = [("原始版本", build()),
                ("标定换内参", build(cam="intrinsics_v4")),
                ("CRF 23 → 18", build(crf=18)),
                ("GOP 30 → 5", build(gop=5)),
                ("子集 100 → 120 条", build(eps=120)),
                ("重新采集原始数据", build(raw="bag-v2"))]

    print("  一条血缘链，括号里是这一步的耗时：")
    print("    raw_bag ──▶ calibrated(%.1f h) ──▶ converted(%.1f h) ──▶ subset(%.1f h)"
          % (cal_h, conv_h, subset_h))
    print("    每个节点：H(node) = sha256( H(inputs) || canonical_json(params) )")
    print()
    print("  %s%s%s" % (pad("变体", 22), pad("变了的节点", 26), "必须重跑"))
    print("  " + "-" * 84)
    total_h = sum(cost.values())
    out = []
    for label, v in variants:
        changed = [k for k in order if v[k] != base[k]]
        if not changed:
            print("  %s%s%s" % (pad(label, 22), pad("（无）", 26),
                                "不用跑（%.1f h 全免）" % total_h))
            out.append((label, 0.0))
            continue
        # 从第一个变了的节点起，下游全部要重跑
        first = order.index(changed[0])
        rerun = order[first:]
        hours = sum(cost[k] for k in rerun)
        tail = "" if abs(hours - total_h) < 1e-9 else "（全量 %.1f h）" % total_h
        out.append((label, hours))
        print("  %s%s%s" % (pad(label, 22),
                            pad(" / ".join(k[:4] for k in changed), 26),
                            "%.1f h%s" % (hours, tail)))

    print()
    print("  关键不是「省了多少小时」，是**知道哪一步不用跑**。")
    print("  改 CRF 不影响 calibrated，所以 3.0 h 的标定可以直接复用；")
    print("  换内参则必须从 calibrated 起全部重来，包括 %.1f h 的转换。" % conv_h)
    print()
    print("  没有血缘时，这两件事在界面上长得一模一样：都是「数据集好像不对」。")
    print("  于是默认操作是全部重跑 —— 而全量重跑要 %.1f h，一天只够跑一次。" % total_h)
    print()
    print("  血缘的成本也可以算：每个节点一行 JSON（哈希 + 参数 + 上游哈希），")
    print("  几十字节。**记录成本是毫秒级，不记录的成本是小时级。**")
    return {"total_h": total_h, "rows": out}


# ---------------------------------------------------------------- [6] 校验

def part6_verify(total_tb=4.15, sha_gbps=1.5, disk_mbps=500,
                 merkle_leaf_kb=1024, trains_per_day=3, train_h=8.0):
    print("\n[6] 校验要花时间：全量 sha256、增量 Merkle、以及校验周期")
    print("-" * 78)
    print("  外部输入：sha256 单核吞吐 %.1f GB/s，磁盘顺序读 %d MB/s，"
          % (sha_gbps, disk_mbps))
    print("  按原始体积算（校验必须读原始字节，压缩数据没法局部验）。")
    print()
    n_bytes = total_tb * TB
    t_cpu = n_bytes / (sha_gbps * GB) / 3600
    t_io = n_bytes / (disk_mbps * 1e6) / 3600
    print("  %s%s" % (pad("方式", 32), "耗时"))
    print("  " + "-" * 46)
    print("  %s%s" % (pad("全量 sha256（受 CPU 限制）", 32), "%.2f 小时" % t_cpu))
    print("  %s%s" % (pad("全量 sha256（受磁盘限制）", 32), "%.2f 小时" % t_io))
    print()
    print("  %.2f TB 一次全量校验 %.1f 小时。**训练前校验一遍**这件事，"
          % (total_tb, t_io))
    print("  在 TB 级上是不成立的：一天 %.0f 次训练就是 %.1f 小时纯校验。"
          % (trains_per_day, trains_per_day * t_io))

    print()
    print("  三个让它变得可行的办法：")
    print()
    print("  ① 只算变了的（mtime + size 先筛）")
    print("     采集是追加的，每天新增 1% 的文件：")
    print("     校验时间从 %.2f h 降到 %.2f min。" % (t_io, t_io * 3600 * 0.01 / 60))
    print()
    print("  ② Merkle 树：改一个块只重算一条路径")
    n_leaf = n_bytes / (merkle_leaf_kb * 1024)
    depth = 1
    while 2 ** depth < n_leaf:
        depth += 1
    print("     %.2f TB / %d KB = %s 个叶子，树高 %d。" %
          (total_tb, merkle_leaf_kb, format(int(n_leaf), ","), depth))
    print("     改 1 个块要重算 %d 个哈希，不是 %s 个。" % (depth, format(int(n_leaf), ",")))
    print("     读一次只要 O(log N) 的证明，约 %d × 32 B = %d B。"
          % (depth, depth * 32))
    print()
    print("  ③ 把校验放在写入时，不放在训练前")
    print("     转换产物落盘就验一次，坏文件当场重转。训练时只查清单版本号。")

    print()
    print("  校验周期决定「坏数据被用了多久」。假设每天 %d 次训练、每次 %.0f 小时："
          % (trains_per_day, train_h))
    print()
    print(head(("校验周期", 14), ("平均暴露", 14), ("暴露期内训练次数", 20)))
    print("  " + "-" * 56)
    cycles = []
    for label, hours in (("每次训练前", 0.0), ("每天", 24.0), ("每周", 168.0),
                         ("每月", 720.0)):
        expo = hours / 2
        uses = expo * trains_per_day / 24
        cycles.append((label, expo, uses))
        print("  %s%s%s" % (pad(label, 14),
                            pad("%.1f 小时" % expo, 14),
                            pad("%.0f 次" % uses, 20)))
    print()
    worst = 720.0 / 2 * trains_per_day / 24
    print("  每月校验一次，意味着**平均有 %.0f 次训练用的是坏数据**，而你不知道是哪几次。"
          % worst)
    print("  这不是「多花点钱」的问题：那 %.0f 次实验的结论全部作废，且无法辨认。" % worst)
    print("  **校验频率不是运维参数，是实验有效性的前提。**")
    return {"full_h": t_io, "depth": depth, "cycles": cycles,
            "incremental_h": t_io * 0.01, "cpu_h": t_cpu}


# ---------------------------------------------------------------- [7] 子集

def part7_views():
    print("\n[7] 做数据子集：复制还是引用")
    print("-" * 78)
    ep_mb = 4.0
    total_ep = 5000
    print("  数据集：%d 个 episode × %.1f MB = %.0f GB" % (total_ep, ep_mb,
                                                          total_ep * ep_mb / 1000))
    print()
    subsets = [("全量", 5000), ("成功样本", 3800), ("长程任务", 1200),
               ("消融 A", 600), ("消融 B", 600)]
    copy_total = sum(n for _, n in subsets) * ep_mb
    ref_total = total_ep * ep_mb

    print(head(("子集", 16), ("物理复制", 16), ("引用（共享底数据）", 22)))
    print("  " + "-" * 62)
    for name, n in subsets:
        print("  %s%s%s" % (pad(name, 16), pad("%.1f GB" % (n * ep_mb / 1000), 16),
                            "0 GB（只存指针）"))
    print("  " + "-" * 62)
    print("  %s%s%s" % (pad("合计", 16),
                        pad("%.1f GB" % (copy_total / 1000), 16),
                        "%.0f GB" % (ref_total / 1000)))
    print()
    print("  复制多花 %.0f GB（%.1f 倍），而且每次改编码都要重做 %d 份拷贝。"
          % ((copy_total - ref_total) / 1000, copy_total / ref_total, len(subsets)))
    print("  引用只花一个几百 KB 的清单文件。")
    print()
    print("  但引用不是免费的，两个坑：")
    print("    · **删除**：删底数据会让所有子集一起坏。必须引用计数 + 禁止直接删。")
    print("    · **权限**：子集 A 和 B 共享底数据后，谁能看 A 就能推出 B 的内容。")
    print("      多团队共用存储时，这个坑会变成安全事故。")
    print()
    print("  折中是「逻辑删除 + 物理回收」：标记删除，引用计数归零后再真删。")
    print("  中间的窗口多长？等于「误删到发现」的时间。回收站的 30 天不是习惯，")
    print("  是事故响应时间的上界。")
    print()
    print("  判断标准：子集之间只是**过滤条件**不同 → 引用；")
    print("  涉及**内容修改**（统一裁剪、重采样、换编码）→ 那是新数据，必须复制。")
    return {"copy_gb": copy_total / 1000, "ref_gb": ref_total / 1000}


# ---------------------------------------------------------------- [8] 分层

def part8_tiers(total_tb=100.0, months=12):
    print("\n[8] 冷热分层：%.0f TB 的账单" % total_tb)
    print("-" * 78)
    print("  外部输入：对象存储公开牌价（美元 / GB / 月），取回另计。")
    print("  量级是对的，具体数字随供应商和区域变——结论只看相对关系。")
    print()
    tiers = [("标准（热）", 0.023, 0.0, 0.0, "毫秒"),
             ("低频（温）", 0.0125, 0.01, 3.0, "毫秒"),
             ("归档（冷）", 0.004, 0.02, 12.0, "分钟到小时")]
    n_gb = total_tb * 1000
    print(head(("层", 16), ("$/GB/月", 12), ("月成本", 14), ("取回 $/GB", 12),
               ("取回延迟", 18)))
    print("  " + "-" * 74)
    for name, price, back, lat_h, lat in tiers:
        print("  %s%s%s%s%s" % (pad(name, 16), pad("%.4f" % price, 12),
                                pad("$%s" % format(int(n_gb * price), ","), 14),
                                pad("%.3f" % back, 12), lat))

    print()
    print("  全放热层：$%s/月，%d 个月 $%s。"
          % (format(int(n_gb * 0.023), ","), months,
             format(int(n_gb * 0.023 * months), ",")))
    print("  全放冷层：$%s/月。差 %.0f 倍——看起来冷层完胜。"
          % (format(int(n_gb * 0.004), ","), 0.023 / 0.004))

    print()
    print("  但训练要读。假设一个 epoch 全量读一遍，100 TB 从冷层取回：")
    print("    取回费：100 × 1024 × $0.02 = $%s（一次）"
          % format(int(total_tb * 1024 * 0.02), ","))
    print("    延迟：%.0f 小时起，且不是流式的——要先解冻。" % 12.0)
    print()
    print("  跑 10 个 epoch：热层 $%s 存储 + $0 取回；冷层 $%s 存储 + $%s 取回。"
          % (format(int(n_gb * 0.023), ","), format(int(n_gb * 0.004), ","),
             format(int(total_tb * 1024 * 0.02 * 10), ",")))
    print("  **冷层此刻贵 %.1f 倍。** 取回成本压倒了存储差价。"
          % ((n_gb * 0.004 + total_tb * 1024 * 0.02 * 10) / (n_gb * 0.023)))

    print()
    print("  所以分层的正确切法不是按「新旧」，是按**读的次数**：")
    print("    · 正在训的、要反复读的：热层。省下的取回费远超存储差价。")
    print("    · 训完的、只留证据的：冷层。它一年被读 0 次，取回费也是 0。")
    print("    · 中间态（回放、审计、复现）：温层。")
    print()
    print("  一个常见的错：把「训完的」也放热层，因为「说不定还要用」。")
    print("  %.0f TB 一年就是 $%s 的「说不定」。" % (total_tb, format(int(n_gb * 0.023 * 12), ",")))
    return {"tiers": [(t[0], n_gb * t[1], total_tb * 1024 * t[2] * 10) for t in tiers],
            "month_hot": n_gb * 0.023}


# ---------------------------------------------------------------- [9] manifest

def part9_manifest():
    print("\n[9] manifest：一个 bit 就能抓到，但它只管字节")
    print("-" * 78)

    tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp_cas")
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)

    def sha_file(path):
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for blk in iter(lambda: fh.read(1 << 16), b""):
                h.update(blk)
        return h.hexdigest()

    eps = []
    for i in range(3):
        p = os.path.join(tmp, "episode_%03d.bin" % i)
        with open(p, "wb") as fh:
            fh.write(make_episode(100 + i, 40, 0.3))
        eps.append(p)

    manifest = {
        "dataset": "demo-v1",
        "created": "2026-11-03",
        "converter": {"codec": "h264", "crf": 23, "gop": 30},
        "episodes": [{"path": os.path.basename(p),
                      "bytes": os.path.getsize(p),
                      "sha256": sha_file(p)} for p in eps],
    }
    with open(os.path.join(tmp, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    print("  manifest.json（节选）：")
    print('    "converter": {"codec": "h264", "crf": 23, "gop": 30},')
    e0 = manifest["episodes"][0]
    print('    "episodes": [{"path": "%s", "bytes": %d,' % (e0["path"], e0["bytes"]))
    print('                  "sha256": "%s..."}]' % e0["sha256"][:24])
    print()

    print("  校验全部文件：")
    ok = 0
    for e in manifest["episodes"]:
        p = os.path.join(tmp, e["path"])
        good = os.path.isfile(p) and sha_file(p) == e["sha256"]
        print("    %s %s" % ("OK  " if good else "FAIL", e["path"]))
        ok += good
    print("  -> %d/%d 通过" % (ok, len(eps)))

    print()
    print("  把 episode_001 翻一个 bit（模拟静默写坏 / bit rot）：")
    p = os.path.join(tmp, "episode_001.bin")
    with open(p, "r+b") as fh:
        fh.seek(5000)
        b = fh.read(1)
        fh.seek(5000)
        fh.write(bytes([b[0] ^ 0x01]))
    for e in manifest["episodes"]:
        if sha_file(os.path.join(tmp, e["path"])) != e["sha256"]:
            print("    FAIL %s  <- 内容与清单不符" % e["path"])
    print()
    print("  只翻 1 个 bit 就抓到了。文件「能打开」和文件「是对的」是两件事。")

    print()
    print("  但它挡不住什么，也要说清楚：")
    limits = [("传输损坏 / 静默写坏", "能", "字节变了"),
              ("误覆盖", "能", "字节变了"),
              ("漏传整个文件", "能", "清单里有、目录里没有"),
              ("清单本身被改", "不能", "没有签名，改清单和改数据一样容易"),
              ("语义错误（标签错了）", "不能", "字节完全正确"),
              ("采集时丢帧", "不能", "包本身是合法的，见实验 1"),
              ("标定漂移", "不能", "几何错了，字节没错，见实验 3"),
              ("版本混用（v2 当 v1 用）", "能", "如果清单写进了产物")]
    print("  %s%s%s" % (pad("问题", 26), pad("能发现吗", 12), "为什么"))
    print("  " + "-" * 74)
    for a, b, c in limits:
        print("  %s%s%s" % (pad(a, 26), pad(b, 12), c))

    print()
    print("  所以质检必须分层：哈希管「字节对不对」，对账管「条数/时长对不对」，")
    print("  几何检查管「物理量对不对」。三层各管一段，任何一层单独用都不够。")
    print("  **「我们有 checksum」不能作为「数据没问题」的论据。**")

    shutil.rmtree(tmp)


# ---------------------------------------------------------------- [10] 结论

def part10_conclusion():
    print("\n[10] 一张决策表")
    print("-" * 78)
    rows = [("内容寻址（哈希做 ID）", "名字会变，哈希不会", "免疫覆盖和重名"),
            ("固定分块去重", "插入一帧污染整个后缀", "只追加的数据才能用"),
            ("CDC 分块", "元数据变多，实现复杂", "会被编辑的数据必须用"),
            ("块 4 KB", "7.8 GB 索引 / TB，2.2 万次请求", "去重率不是白来的"),
            ("块 1 MB", "40% 静止帧省 0%", "块大小是索引参数"),
            ("短哈希当 ID", "8 位在 1 万条就必撞", "12 位够 10 万，百万要 15 位"),
            ("每天全量校验", "4.15 TB 要 2.3 小时", "mtime 预筛 + Merkle 增量"),
            ("子集复制", "5 个子集 2.2 倍存储", "只有改内容时才该复制"),
            ("直接删底数据", "所有子集一起坏", "逻辑删除 + 引用计数"),
            ("训完的数据放热层", "100 TB 一年 $27,600", "按读次数分层，不按新旧")]
    print(head(("做法", 26), ("代价", 34), ("为什么", 0)))
    print("  " + "-" * 90)
    for a, b, c in rows:
        print("  %s%s%s" % (pad(a, 26), pad(b, 34), c))
    print()
    print("  一句话：存储与版本里唯一不能让步的是**「这一份到底是哪一份」**。")
    print("  去重率、块大小、分层策略都是可以换算的取舍；可追溯性不是。")


if __name__ == "__main__":
    print("=" * 80)
    print("实验 5：内容寻址、去重、血缘与完整性校验")
    print("=" * 80)
    part1_dedup()
    part2_shift()
    part3_metadata()
    part4_hash_bits()
    part5_lineage()
    part6_verify()
    part7_views()
    part8_tiers()
    part9_manifest()
    part10_conclusion()
    print("\n" + "=" * 80)

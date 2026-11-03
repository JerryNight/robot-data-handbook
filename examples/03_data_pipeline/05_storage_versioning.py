#!/usr/bin/env python3
"""
实验 5：内容寻址、去重，以及「像管代码一样管数据」。

运行：python examples/03_data_pipeline/05_storage_versioning.py

机器人数据有两个难受的性质：**大**、**重复**。
遥操作采集时，机器人有大量「回到起始位」「等待下一步」的静止段，
这些帧在不同 episode 之间几乎一模一样。

    [1] 按块哈希去重，实测能省多少（这个数字是算出来的，不是估的）
    [2] 版本与血缘：改一次编码参数，哪些哈希会变
    [3] 逻辑视图 vs 物理复制：做数据子集时该复制还是该引用
    [4] 一个 40 行的 manifest 实现，以及它挡得住什么
"""

import hashlib
import json
import os
import random
import shutil
import struct


def width(s: str) -> int:
    return sum(2 if ord(c) > 0x2E80 else 1 for c in s)


def pad(s: str, n: int) -> str:
    return s + " " * max(0, n - width(s))


CHUNK = 16 * 1024          # 16KB 块；对象存储/CAS 的粒度通常在这个量级


# ---------------------------------------------------------------- 数据生成

def make_episode(seed, n_frames, idle_ratio):
    """
    合成一个 episode 的字节流。

    关键在于：idle 段用**同一份内容**，模拟「机器人静止 / 回原点」。
    这正是真实数据里重复量的主要来源。
    """
    rng = random.Random(seed)
    idle = bytes(((i * 31 + seed) % 251) for i in range(CHUNK))
    parts = []
    for i in range(n_frames):
        if rng.random() < idle_ratio:
            parts.append(idle)                       # 完全相同的静止帧
        else:
            parts.append(rng.randbytes(CHUNK))
    return b"".join(parts)


def chunk_hashes(blob, size=CHUNK):
    return [hashlib.sha256(blob[i:i + size]).hexdigest()[:16]
            for i in range(0, len(blob), size)]


# ---------------------------------------------------------------- [1] 去重

def part1_dedup():
    print("\n[1] 内容寻址去重：省多少，取决于 idle 占比")
    print("-" * 78)
    N_EP, FRAMES = 80, 80
    print("  %d 个 episode，每个 %d 帧，每帧 %d KB。" % (N_EP, FRAMES, CHUNK // 1024))
    print("  按 %d KB 切块、算 sha256，相同哈希只存一份。" % (CHUNK // 1024))
    print()
    print("  %s%s%s%s" % (pad("idle 占比", 12), pad("去重前", 14),
                          pad("去重后", 14), "节省"))
    print("  " + "-" * 58)
    for idle_ratio in (0.0, 0.2, 0.4, 0.6):
        total = 0
        seen = set()
        for ep in range(N_EP):
            blob = make_episode(ep, FRAMES, idle_ratio)
            total += len(blob)
            seen.update(chunk_hashes(blob))
        after = len(seen) * CHUNK
        print("  %s%s%s%s" % (pad("%.0f%%" % (idle_ratio * 100), 12),
                              pad("%.1f MB" % (total / 1e6), 14),
                              pad("%.1f MB" % (after / 1e6), 14),
                              "%.0f%%" % ((1 - after / total) * 100)))

    print()
    print("  注意 0% 那一行：节省也是 0 —— 去重不会凭空变出压缩。")
    print("  它只做一件事：把**字节级相同**的内容只存一次。")
    print()
    print("  真实数据集里 idle 段能到 30-50%，所以去重通常能省 25-40%。")
    print("  但这个数字**强依赖任务**：连续流水线作业的重复率高，")
    print("  杂乱桌面抓取的重复率低。不测就估，一定会估错。")

    print()
    print("  两个必须知道的代价：")
    print("    · 读一个 episode 变成「按哈希列表逐个取块」，小文件请求变多")
    print("    · 删除要做引用计数，否则删了 episode A 会让 episode B 缺块")
    print("  第二条比第一条严重：这是所有 CAS 系统出事故的地方。")


# ---------------------------------------------------------------- [2] 血缘

def part2_lineage():
    print("\n[2] 版本与血缘：改一个参数，哪些下游会变")
    print("-" * 78)
    print("  一条最小的血缘链：")
    print("    raw_bag ──▶ calibrated ──▶ converted(mp4) ──▶ subset(train)")
    print()
    print("  每个节点用「输入哈希 + 处理参数」算出自己的哈希：")
    print("    H(node) = sha256( H(inputs) || canonical_json(params) )")
    print()

    def h(*parts):
        m = hashlib.sha256()
        for p in parts:
            m.update(repr(p).encode())
        return m.hexdigest()[:12]

    def build(crf, gop, include_ep):
        raw = h("bag", "v1")
        cal = h(raw, {"cam": "intrinsics_v3", "hand_eye": "2026-10-28"})
        conv = h(cal, {"codec": "h264", "crf": crf, "gop": gop, "res": "640x480"})
        sub = h(conv, {"filter": "success_only", "eps": include_ep})
        return {"raw": raw, "calibrated": cal, "converted": conv, "subset": sub}

    base = build(23, 30, 100)
    variants = [
        ("原始版本", build(23, 30, 100)),
        ("CRF 23 -> 18（提画质）", build(18, 30, 100)),
        ("GOP 30 -> 5（提随机访问）", build(23, 5, 100)),
        ("子集 100 -> 120 条", build(23, 30, 120)),
    ]

    print("  %s%s%s%s%s" % (pad("变体", 28), pad("raw", 16), pad("calibrated", 16),
                            pad("converted", 16), "subset"))
    print("  " + "-" * 82)
    for label, v in variants:
        cells = []
        for k in ("raw", "calibrated", "converted", "subset"):
            same = v[k] == base[k]
            cells.append(pad(("=" if same else "*") + v[k], 16))
        print("  %s%s" % (pad(label, 28), "".join(cells).rstrip()))
    print("  （= 表示哈希未变，* 表示已变）")

    print()
    print("  这张表就是血缘的作用：看到 * 那一列，立刻知道「哪一步被改动了」。")
    print("  CRF 改动影响 converted 和 subset，但 raw 和 calibrated 不受影响 ——")
    print("  所以**不需要重新标定**，只需要重跑转换。这就是能省下几小时的地方。")
    print()
    print("  反过来说：如果没有血缘，改一个参数之后，")
    print("  你根本不知道手上这份训练集是哪次转换产出的。")
    print("  两周后发现指标掉了，也查不出原因 —— 这是复现性事故，不是工程细节。")


# ---------------------------------------------------------------- [3] 视图

def part3_views():
    print("\n[3] 做数据子集：复制还是引用？")
    print("-" * 78)
    ep_size_mb = 4.0          # 一个 episode 转换后的体积（含 3 路视频）
    total_ep = 5000
    print("  数据集：%d 个 episode × %.1f MB = %.0f GB" % (total_ep, ep_size_mb,
                                                          total_ep * ep_size_mb / 1000))
    print()

    subsets = [("全量", 5000), ("成功样本", 3800), ("长程任务", 1200),
               ("消融 A", 600), ("消融 B", 600)]
    copy_total = sum(n for _, n in subsets) * ep_size_mb
    ref_total = total_ep * ep_size_mb

    print("  %s%s%s" % (pad("子集", 16), pad("物理复制", 16), "引用（共享底数据）"))
    print("  " + "-" * 56)
    for name, n in subsets:
        print("  %s%s%s" % (pad(name, 16), pad("%.1f GB" % (n * ep_size_mb / 1000), 16),
                            "0 GB（只存指针）"))
    print("  " + "-" * 56)
    print("  %s%s%s" % (pad("合计", 16),
                        pad("%.1f GB" % (copy_total / 1000), 16),
                        "%.0f GB" % (ref_total / 1000)))
    print()
    print("  复制方案多花 %.0f GB，而且每次改编码都要重做 %d 份拷贝。"
          % ((copy_total - ref_total) / 1000, len(subsets)))
    print("  引用方案只花一个几百 KB 的清单文件。")

    print()
    print("  但引用不是免费的，两个坑：")
    print("    · **删除**：删底数据会让所有子集一起坏。必须做引用计数 + 禁止直接删。")
    print("    · **权限**：子集 A 和 B 共享底数据后，谁能看 A 就能推出 B 的内容。")
    print("      多团队共用存储时，这个坑会变成安全事故。")
    print()
    print("  折中办法是「逻辑删除 + 物理回收」：标记删除，")
    print("  确认引用计数归零后再真正回收，中间留一个可撤销的窗口。")


# ---------------------------------------------------------------- [4] manifest

def part4_manifest():
    print("\n[4] 一个 40 行的 manifest：它挡得住什么，挡不住什么")
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

    # 造 3 个 episode 文件
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
    mpath = os.path.join(tmp, "manifest.json")
    with open(mpath, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    print("  manifest.json 内容（节选）：")
    print('    "converter": {"codec": "h264", "crf": 23, "gop": 30},')
    e0 = manifest["episodes"][0]
    print('    "episodes": [{"path": "%s", "bytes": %d,'
          % (e0["path"], e0["bytes"]))
    print('                  "sha256": "%s..."}]' % e0["sha256"][:24])

    print("\n  校验全部文件：")
    ok = 0
    for e in manifest["episodes"]:
        p = os.path.join(tmp, e["path"])
        good = os.path.isfile(p) and sha_file(p) == e["sha256"]
        print("    %s %s" % ("OK  " if good else "FAIL", e["path"]))
        ok += good
    print("  -> %d/%d 通过" % (ok, len(eps)))

    print("\n  现在把 episode_001 改一个字节（模拟传输损坏 / 盘上 bit rot）：")
    p = os.path.join(tmp, "episode_001.bin")
    with open(p, "r+b") as fh:
        fh.seek(5000)
        b = fh.read(1)
        fh.seek(5000)
        fh.write(bytes([b[0] ^ 0x01]))          # 翻转一个 bit
    for e in manifest["episodes"]:
        path = os.path.join(tmp, e["path"])
        good = sha_file(path) == e["sha256"]
        if not good:
            print("    FAIL %s  <- 内容与清单不符" % e["path"])
    print()
    print("  只翻了 1 个 bit，就抓到了。这就是哈希的作用：")
    print("  文件「能打开」和文件「是对的」是两件完全不同的事。")

    print("\n  但它挡不住什么，也要说清楚：")
    limits = [
        ("传输损坏 / 静默写坏", "能"),
        ("误覆盖", "能"),
        ("漏传整个文件", "能（清单里有、目录里没有）"),
        ("语义错误（标签错了）", "不能"),
        ("采集时丢帧", "不能（需要 metadata 对账，见实验 1）"),
        ("标定漂移", "不能（需要实验 3 的几何检查）"),
    ]
    print("  %s%s" % (pad("问题", 26), "manifest 能发现吗"))
    print("  " + "-" * 50)
    for a, b in limits:
        print("  %s%s" % (pad(a, 26), b))

    print()
    print("  所以质检必须分层：哈希管「字节对不对」，")
    print("  对账管「条数/时长对不对」，几何检查管「物理量对不对」。")
    print("  三层各管一段，任何一层单独用都不够。")

    shutil.rmtree(tmp)


if __name__ == "__main__":
    print("=" * 78)
    print("实验 5：内容寻址、去重、血缘与完整性校验")
    print("=" * 78)
    part1_dedup()
    part2_lineage()
    part3_views()
    part4_manifest()
    print("\n" + "=" * 78)

#!/usr/bin/env python3
"""
检查所有 Markdown 里的相对链接是否指向真实存在的文件。

只检查仓库内部的相对链接；http/https/mailto 和纯锚点跳过。

用法：
    python scripts/check_links.py
    python scripts/check_links.py -v     # 列出所有检查过的链接

有断链时返回 1。
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
VERBOSE = "-v" in sys.argv or "--verbose" in sys.argv

# [文本](目标)  和引用式 [文本][ref] 定义  [ref]: 目标
INLINE = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
REF_DEF = re.compile(r"^\[[^\]]+\]:\s*(\S+)", re.MULTILINE)
SKIP_PREFIX = ("http://", "https://", "mailto:", "#", "ftp:")


def iter_md_files():
    return sorted(p for p in ROOT.rglob("*.md") if ".git" not in p.parts)


def extract_links(text):
    seen = []
    for rx in (INLINE, REF_DEF):
        for m in rx.finditer(text):
            target = m.group(1).strip()
            seen.append(target)
    return seen


def resolve(src_file, target):
    """把链接目标解析成仓库内的路径；越界或外部链接返回 None。"""
    if target.startswith(SKIP_PREFIX):
        return None
    # 去掉锚点和查询串
    path = target.split("#", 1)[0].split("?", 1)[0]
    if not path:
        return None
    if path.startswith("/"):
        # 仓库根相对路径
        return (ROOT / path.lstrip("/")).resolve()
    return (src_file.parent / path).resolve()


def main():
    files = iter_md_files()
    if not files:
        print("没找到任何 Markdown 文件", file=sys.stderr)
        return 1

    checked = 0
    broken = []

    for md in files:
        text = md.read_text(encoding="utf-8", errors="replace")
        for target in extract_links(text):
            dest = resolve(md, target)
            if dest is None:
                continue
            checked += 1
            rel_src = md.relative_to(ROOT).as_posix()
            if VERBOSE:
                print(f"  {rel_src} -> {target}")
            try:
                dest.relative_to(ROOT)
            except ValueError:
                broken.append((rel_src, target, "指向仓库之外"))
                continue
            if not dest.exists():
                broken.append((rel_src, target, "目标不存在"))

    print(f"检查了 {len(files)} 个 Markdown 文件、{checked} 条相对链接")

    if broken:
        print(f"\n发现 {len(broken)} 条断链：")
        for src, target, why in broken:
            print(f"  [{why}] {src} -> {target}")
        return 1

    print("全部有效")
    return 0


if __name__ == "__main__":
    sys.exit(main())

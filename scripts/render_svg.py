#!/usr/bin/env python3
"""
把 SVG 渲染成 PNG，用来人眼检查排版。

为什么需要它
------------
手写 SVG 的坐标是算出来的，算出来的东西一定有排错的地方：文字压在线上、
标签被画布裁掉、两个标注撞在一起。这些错误在代码里看不出来，只有渲染出来
才看得见。所以本仓库的规矩是：**每张图必须渲染检查过才能提交。**

用法
----
    python scripts/render_svg.py 02-sensor-data/figures/fig-cam-01-two-arrays.svg
    python scripts/render_svg.py 02-sensor-data/figures/*.svg
    python scripts/render_svg.py x.svg --out /tmp/preview --scale 2

画布尺寸从 SVG 的 viewBox 自动读，不用手填。

只用标准库 + 本机的 Chrome / Edge / Chromium。
找不到浏览器时脚本会直接报错，而不是静默跳过 —— 静默跳过等于没有检查。
"""

import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

VIEWBOX = re.compile(r'viewBox="\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*"')

# 按优先级排列的浏览器位置。找不到就继续往下试。
CANDIDATES = [
    os.environ.get("CHROME", ""),
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
]


def find_browser():
    for c in CANDIDATES:
        if c and pathlib.Path(c).is_file():
            return c
    for name in ("google-chrome", "chromium", "chromium-browser", "chrome", "msedge"):
        p = shutil.which(name)
        if p:
            return p
    sys.exit(
        "找不到 Chrome / Edge / Chromium。\n"
        "装一个，或者用环境变量指定：CHROME=/path/to/chrome python scripts/render_svg.py ..."
    )


def viewbox(svg_path):
    head = svg_path.read_text(encoding="utf-8", errors="replace")[:2000]
    m = VIEWBOX.search(head)
    if not m:
        sys.exit(f"{svg_path.name}: 前 2000 字符里没有 viewBox，无法确定画布尺寸。")
    return int(round(float(m.group(1)))), int(round(float(m.group(2))))


def render(browser, svg_path, out_dir, scale, user_dir):
    w, h = viewbox(svg_path)
    png = (out_dir / (svg_path.stem + ".png")).resolve()

    # Windows 上必须给 --user-data-dir，否则 headless Chrome 写自己的配置目录时
    # 会 "Failed to write file: 拒绝访问 (0x5)"，截图直接失败。
    cmd = [
        browser,
        "--headless",
        "--disable-gpu",
        "--hide-scrollbars",
        f"--user-data-dir={user_dir}",
        f"--window-size={w},{h}",
        f"--force-device-scale-factor={scale}",
        # --screenshot 和目标文件都必须是绝对路径；相对路径会落到浏览器的
        # 工作目录里，然后你以为失败了，其实文件在别处。
        f"--screenshot={png}",
        svg_path.resolve().as_uri(),
    ]
    subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not png.is_file() or png.stat().st_size == 0:
        sys.exit(f"{svg_path.name}: 渲染失败，没有生成 {png}")
    return png


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        sys.exit(__doc__)

    out_dir = pathlib.Path(tempfile.gettempdir()) / "svgpreview"
    scale = 2
    for i, a in enumerate(sys.argv):
        if a == "--out" and i + 1 < len(sys.argv):
            out_dir = pathlib.Path(sys.argv[i + 1])
            args = [x for x in args if x != sys.argv[i + 1]]
        if a == "--scale" and i + 1 < len(sys.argv):
            scale = int(sys.argv[i + 1])
            args = [x for x in args if x != sys.argv[i + 1]]

    out_dir.mkdir(parents=True, exist_ok=True)
    user_dir = out_dir / "_chrome_profile"
    user_dir.mkdir(exist_ok=True)
    browser = find_browser()

    for a in args:
        p = pathlib.Path(a)
        if not p.is_file():
            sys.exit(f"没有这个文件：{p}")
        png = render(browser, p, out_dir, scale, user_dir)
        print(png)


if __name__ == "__main__":
    main()

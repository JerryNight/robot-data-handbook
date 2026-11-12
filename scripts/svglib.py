# -*- coding: utf-8 -*-
"""
手写 SVG 的公共小工具：把数学坐标映射到画布坐标，避免手填像素。

为什么不直接画 PNG、不引 matplotlib
------------------------------------
本仓库的图有两个硬要求：**中文不糊**、**diff 可读**。

- PNG 在 2x 屏上会糊，而 SVG 是矢量，任何缩放下文字都锐利；
- SVG 是纯文本，改一个数字在 git diff 里看得见，评审时能指着某行说
  "这里 26 cm 应该是 26 mm"。PNG 只能整张替换，等于没有评审。

代价是每张图要自己算坐标，所以有了这个文件。

用法
----
    from svglib import *

    W, H = 760, 430
    s = [head(W, H)]
    s.append(txt(24, 30, "标题", 12.5, INK, weight="600"))

    m = XY(96, 122, 384, 340, 0.0, 1.0, 0.0, 10.0)   # 数学坐标 -> 画布坐标
    s.append(poly([m(x, y) for x, y in data], BLUE, 2.2))

    open("fig.svg", "w", encoding="utf-8").write("\n".join(s) + "\n</svg>\n")

写完必须渲染检查：`python scripts/render_svg.py fig.svg`。
**不渲染就等于没画** —— 见 CONTRIBUTING.md 的"图的自检清单"。
"""

import math

# ---------------------------------------------------------------- 调色板
# 只用这几个色，保证全仓库的图看起来是同一套东西。
# 语义固定，不要为了好看换色：
#   BLUE   主数据 / 正确做法
#   ORANGE 对照数据 / 另一种做法
#   RED    错误、危险、会出事的地方
#   GREEN  可行、健康、结论
#   PURPLE 第三组数据
#   GREY   次要刻度文字
INK = "#1F2328"
MUT = "#57606A"
LINE = "#E4E8EC"
BLUE = "#3C6E9F"
ORANGE = "#C0651A"
RED = "#C0392B"
GREEN = "#2E7D5B"
PURPLE = "#6B5B95"
GREY = "#8A94A0"


def esc(s):
    """转义 XML 特殊字符。不转义的话一个 & 就能让整张图白屏。"""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def head(w, h):
    """SVG 头 + 白底。白底必须显式画，否则深色主题下文字是浅色的会看不见。"""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="{w}" height="{h}" font-family="\'PingFang SC\',\'Microsoft YaHei\','
            f'\'Noto Sans CJK SC\',sans-serif">\n  <rect width="{w}" height="{h}" fill="#ffffff"/>')


def txt(x, y, s, size=11, color=INK, anchor="start", weight="400", op=1.0, ls=0):
    """单行文字。(x, y) 是基线左端；anchor="end" 时是基线右端，"middle" 是中心。"""
    extra = f' letter-spacing="{ls}"' if ls else ""
    return (f'  <text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" '
            f'text-anchor="{anchor}" font-weight="{weight}" opacity="{op}"{extra}>{esc(s)}</text>')


def line(x1, y1, x2, y2, color=INK, w=1, dash=None, op=1.0):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'  <line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-width="{w}" opacity="{op}"{d}/>')


def rect(x, y, w, h, fill="none", stroke=INK, r=6, sw=1.2, dash=None, op=1.0):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'  <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" opacity="{op}"{d}/>')


def arrow(x1, y1, x2, y2, color=INK, w=1.6, hd=6.5, dash=None):
    """带箭头的线段，箭头画在终点。hd 是箭头长度（px）。"""
    ang = math.atan2(y2 - y1, x2 - x1)
    bx, by = x2 - hd * math.cos(ang), y2 - hd * math.sin(ang)
    d = f' stroke-dasharray="{dash}"' if dash else ""
    p1 = (x2, y2)
    p2 = (bx + hd * 0.55 * math.sin(ang), by - hd * 0.55 * math.cos(ang))
    p3 = (bx - hd * 0.55 * math.sin(ang), by + hd * 0.55 * math.cos(ang))
    pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in (p1, p2, p3))
    return (f'  <line x1="{x1:.1f}" y1="{y1:.1f}" x2="{bx:.1f}" y2="{by:.1f}" stroke="{color}" '
            f'stroke-width="{w}"{d}/>\n  <polygon points="{pts}" fill="{color}"/>')


def path(d, color=INK, w=2, fill="none", dash=None, op=1.0, cap="round"):
    dd = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'  <path d="{d}" fill="{fill}" stroke="{color}" stroke-width="{w}" '
            f'stroke-linecap="{cap}" stroke-linejoin="round" opacity="{op}"{dd}/>')


def poly(pts, color=INK, w=2, dash=None, op=1.0):
    """折线。pts 是 [(x, y), ...]，已转换到画布坐标。"""
    d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts)
    return path(d, color, w, "none", dash, op)


def circle(cx, cy, r, fill=INK, stroke="none", sw=0, op=1.0):
    return (f'  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}" opacity="{op}"/>')


def clip(x, y, w, h, rid, r=4):
    """定义裁剪区。配合 <g clip-path="url(#rid)"> 使用，防止曲线冲出面板。"""
    return (f'  <clipPath id="{rid}"><rect x="{x:.1f}" y="{y:.1f}" '
            f'width="{w:.1f}" height="{h:.1f}" rx="{r}"/></clipPath>')


class XY:
    """数学坐标 (x0..x1, y0..y1) -> 画布坐标 (px0..px1, py0..py1)，y 轴自动翻转。

    m = XY(96, 122, 384, 340, 0.0, 1.0, 0.0, 10.0)
    m(0.5, 5.0)   -> (x, y) 像素元组，直接喂给 poly/circle
    m.px(0.5)     -> 只要横坐标
    m.py(5.0)     -> 只要纵坐标

    对数轴不要用它，自己写映射（见 02-sensor-data/figures 里任何一张对数轴图）。
    混用线性和对数映射是"刻度对不上条形"这类 bug 的唯一来源。
    """

    def __init__(self, px0, py0, px1, py1, x0, x1, y0, y1):
        self.px0, self.py0, self.px1, self.py1 = px0, py0, px1, py1
        self.x0, self.x1, self.y0, self.y1 = x0, x1, y0, y1

    def px(self, x):
        return self.px0 + (x - self.x0) / (self.x1 - self.x0) * (self.px1 - self.px0)

    def py(self, y):
        return self.py1 - (y - self.y0) / (self.y1 - self.y0) * (self.py1 - self.py0)

    def __call__(self, x, y):
        return (self.px(x), self.py(y))


def curve(data, m, w=2.2, color=BLUE, dash=None, xkey=0, ykey=1, op=1.0):
    """把 [(x, y), ...] 按映射 m 画成折线。"""
    return poly([m(p[xkey], p[ykey]) for p in data], color, w, dash, op)

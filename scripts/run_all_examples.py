#!/usr/bin/env python3
"""
跑一遍所有示例，确认没有报错。

用法：
    python scripts/run_all_examples.py          # 跑全部，静默输出
    python scripts/run_all_examples.py -v       # 跑全部，显示每个脚本的输出

任何脚本以非零退出码结束，本脚本立即停止并返回 1。
"""

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
VERBOSE = "-v" in sys.argv or "--verbose" in sys.argv


def main():
    files = sorted(ROOT.glob("examples/*/*.py"))
    if not files:
        print("没找到任何示例脚本", file=sys.stderr)
        return 1

    print(f"共 {len(files)} 个示例\n")
    failed = []

    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        print(f"[RUN ] {rel}", flush=True)
        result = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},
        )
        if result.returncode != 0:
            print(f"[FAIL] {rel}  (exit {result.returncode})")
            if result.stderr:
                print(result.stderr[-1500:])
            failed.append(rel)
        else:
            lines = len(result.stdout.splitlines())
            print(f"[ OK ] {rel}  ({lines} 行输出)")
            if VERBOSE:
                print(result.stdout)

    print()
    if failed:
        print(f"失败 {len(failed)} 个：")
        for f in failed:
            print(f"  - {f}")
        return 1

    print(f"全部 {len(files)} 个示例通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())

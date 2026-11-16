"""命令行入口。

示例:
  # 列出 STEP 中的实体
  python lattice_fill.py model.step --list

  # 对 0 号实体做 Gyroid 填充: 单元 6mm, 壁厚 0.8mm, 皮壳 1.2mm
  python lattice_fill.py model.step -s 0 -t gyroid -c 6 -w 0.8 --shell 1.2 -o out.stl

  # 蜂窝填充全部实体, 未选中的实体原样并入输出
  python lattice_fill.py model.step -t hex -c 4 --keep-others
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import trimesh

from .core import FillParams, fill_solid
from .lattices import LATTICE_TYPES, STRUT_TYPES
from .step_io import describe_solids, load_solids, select_solids


def build_parser() -> argparse.ArgumentParser:
    types_help = "; ".join(f"{k}: {v}" for k, v in LATTICE_TYPES.items())
    ap = argparse.ArgumentParser(
        prog="lattice_fill",
        description="STEP 实体晶格参数化填充: 保留内外表面皮壳, 内部填充可调晶格。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("input", help="输入文件 (.step/.stp, 也支持 .stl/.obj/.ply/.glb)")
    ap.add_argument("--list", action="store_true", help="仅列出文件中的实体后退出")
    ap.add_argument(
        "-s", "--solids", default="all",
        help="选取实体: 逗号分隔的序号或名称子串, 如 '0,2' 或 'Body1'; all=全部",
    )
    ap.add_argument(
        "-t", "--type", default="gyroid", choices=sorted(LATTICE_TYPES),
        help=f"晶格类型 ({types_help})",
    )
    ap.add_argument("-c", "--cell", type=float, default=5.0, help="晶格单元大小 (mm)")
    ap.add_argument("-w", "--wall", type=float, default=0.8, help="晶格壁厚 (mm)")
    ap.add_argument(
        "-d", "--strut", type=float, default=None,
        help="连杆直径 (mm), 用于杆状晶格 hex3d/bcc/cubic; 未指定时取 --wall 值",
    )
    ap.add_argument(
        "--layer", type=float, default=None,
        help="层高 (mm), hexcell/hex3d 的 Z 向周期; 默认等于 --cell",
    )
    ap.add_argument(
        "--axis", default="z", choices=["x", "y", "z"],
        help="晶格轴向 (蜂窝孔道/挤出方向)",
    )
    ap.add_argument(
        "--shell", type=float, default=1.0,
        help="保留的表面皮壳厚度 (mm), 内外表面均保留; 0=不留皮壳",
    )
    ap.add_argument(
        "-r", "--res", type=int, default=8,
        help="分辨率: 每个晶格单元的体素数, 越大越精细也越慢",
    )
    ap.add_argument(
        "--blur", type=float, default=0.0,
        help="隐式场高斯平滑 sigma (体素数, 建议 0.8~1.2), 消除表面体素台阶",
    )
    ap.add_argument("--smooth", type=int, default=0, help="Taubin 平滑迭代次数 (0=不平滑)")
    ap.add_argument(
        "--keep-others", action="store_true",
        help="未选中的实体以原始网格并入输出文件",
    )
    ap.add_argument(
        "--preview", action="store_true",
        help="为每个实体额外输出切片预览 PNG (与输出文件同目录)",
    )
    ap.add_argument(
        "--scale", type=float, default=None,
        help="坐标缩放系数 (默认自动: STEP 按 m→mm 乘 1000, 其它格式不缩放)",
    )
    ap.add_argument(
        "-o", "--output", default=None,
        help="输出文件 (.stl/.obj/.ply/.glb), 默认 <输入名>_lattice.stl",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not os.path.isfile(args.input):
        print(f"输入文件不存在: {args.input}", file=sys.stderr)
        return 2

    print(f"载入 {args.input} ...")
    solids = load_solids(args.input, scale=args.scale)
    print(f"共 {len(solids)} 个实体:\n{describe_solids(solids)}\n")

    if args.list:
        return 0

    picked = select_solids(solids, args.solids)
    picked_idx = {i for i, _, _ in picked}
    print(f"选中 {len(picked)} 个实体: {sorted(picked_idx)}")

    output = args.output or os.path.splitext(args.input)[0] + "_lattice.stl"
    out_base = os.path.splitext(output)[0]

    results: list[trimesh.Trimesh] = []
    t0 = time.time()
    # 杆状晶格用连杆直径, 其余用壁厚
    if args.type in STRUT_TYPES:
        thickness = args.strut if args.strut is not None else args.wall
        print(f"杆状晶格 {args.type}: 连杆直径 {thickness} mm")
    else:
        thickness = args.wall
        if args.strut is not None:
            print(f"[提示] --strut 仅对杆状晶格 ({', '.join(sorted(STRUT_TYPES))}) 生效, 已忽略")

    for i, name, mesh in picked:
        print(f"\n处理实体 [{i}] {name}:")
        if not mesh.is_watertight:
            print("  [警告] 实体网格非水密, 内外判定可能不准")
        params = FillParams(
            lattice=args.type,
            cell=args.cell,
            wall=thickness,
            shell=args.shell,
            layer=args.layer,
            axis=args.axis,
            res=args.res,
            blur=args.blur,
            smooth=args.smooth,
            preview=f"{out_base}_s{i}_slices.png" if args.preview else None,
        )
        results.append(fill_solid(mesh, params))

    if args.keep_others:
        for i, (name, mesh) in enumerate(solids):
            if i not in picked_idx:
                print(f"保留原始实体 [{i}] {name}")
                results.append(mesh)

    merged = trimesh.util.concatenate(results) if len(results) > 1 else results[0]
    merged.export(output)
    print(
        f"\n输出: {output}  ({len(merged.faces):,} 面, "
        f"{os.path.getsize(output) / 1e6:.1f} MB, 用时 {time.time() - t0:.1f} s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

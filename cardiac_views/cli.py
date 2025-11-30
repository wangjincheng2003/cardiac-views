from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .core import (
    DEFAULT_COLOR_MAX,
    DEFAULT_COLOR_MIN,
    DEFAULT_HLA_PLAN,
    DEFAULT_SA_PLAN,
    DEFAULT_SIGMA,
    DEFAULT_TARGET_DIMS,
    DEFAULT_VLA_PLAN,
    DEFAULT_VOX_DIMS,
    SlicePlan,
    generate_views,
)


def _slice_plan(values: Sequence[int], label: str) -> SlicePlan:
    if len(values) != 3:
        raise argparse.ArgumentTypeError(f"{label} 需要3个整数: start stop count")
    return SlicePlan(values[0], values[1], values[2])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从体数据绘制VLA/HLA/SA心脏断层图，支持 .dat 和 .dcm 文件，可处理单个文件或目录。",
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="dat",
        help="输入 .dat/.dcm 文件或目录（默认 ./dat）",
    )
    parser.add_argument(
        "-p",
        "--palette",
        type=Path,
        help="DICOM调色板路径（可选，默认使用热金属备选色表）",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="输出目录（默认为输入同级 view 子目录）",
    )
    parser.add_argument(
        "--vox-dims",
        nargs=3,
        type=int,
        default=DEFAULT_VOX_DIMS,
        metavar=("Z", "Y", "X"),
        help=f"原始体素尺寸，默认 {DEFAULT_VOX_DIMS}",
    )
    parser.add_argument(
        "--target-dims",
        nargs=3,
        type=int,
        default=DEFAULT_TARGET_DIMS,
        metavar=("Z", "Y", "X"),
        help=f"插值后尺寸，默认 {DEFAULT_TARGET_DIMS}",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=DEFAULT_SIGMA,
        help=f"插值后高斯平滑sigma，设为0禁用，默认 {DEFAULT_SIGMA}",
    )
    parser.add_argument(
        "--color-min",
        type=float,
        default=DEFAULT_COLOR_MIN,
        help=f"统一颜色最小值，默认 {DEFAULT_COLOR_MIN}",
    )
    parser.add_argument(
        "--color-max",
        type=float,
        default=DEFAULT_COLOR_MAX,
        help=f"统一颜色最大值（若不开启auto），默认 {DEFAULT_COLOR_MAX}",
    )
    parser.add_argument(
        "--auto-color-max",
        action="store_true",
        default=True,
        help="使用VLA第二张上半部分的最大值作为颜色上阈值（默认开启）",
    )
    parser.add_argument(
        "--no-auto-color-max",
        action="store_true",
        help="禁用自动颜色范围，使用 --color-max 指定的值",
    )
    parser.add_argument(
        "--colorbar",
        action="store_true",
        default=True,
        help="在输出图中添加颜色条（默认开启）",
    )
    parser.add_argument(
        "--no-colorbar",
        action="store_true",
        help="不显示颜色条",
    )
    parser.add_argument(
        "--vla",
        nargs=3,
        type=int,
        default=[
            DEFAULT_VLA_PLAN.start,
            DEFAULT_VLA_PLAN.stop,
            DEFAULT_VLA_PLAN.count,
        ],
        metavar=("START", "STOP", "COUNT"),
        help=f"VLA切片范围 start stop count，默认 {DEFAULT_VLA_PLAN}",
    )
    parser.add_argument(
        "--hla",
        nargs=3,
        type=int,
        default=[
            DEFAULT_HLA_PLAN.start,
            DEFAULT_HLA_PLAN.stop,
            DEFAULT_HLA_PLAN.count,
        ],
        metavar=("START", "STOP", "COUNT"),
        help=f"HLA切片范围 start stop count，默认 {DEFAULT_HLA_PLAN}",
    )
    parser.add_argument(
        "--sa",
        nargs=3,
        type=int,
        default=[
            DEFAULT_SA_PLAN.start,
            DEFAULT_SA_PLAN.stop,
            DEFAULT_SA_PLAN.count,
        ],
        metavar=("START", "STOP", "COUNT"),
        help=f"SA切片范围 start stop count，默认 {DEFAULT_SA_PLAN}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    vla_plan = _slice_plan(args.vla, "VLA")
    hla_plan = _slice_plan(args.hla, "HLA")
    sa_plan = _slice_plan(args.sa, "SA")

    # 处理 --no-xxx 参数覆盖默认值
    use_auto = args.auto_color_max and not args.no_auto_color_max
    show_bar = args.colorbar and not args.no_colorbar

    try:
        results = generate_views(
            Path(args.input),
            palette_path=args.palette,
            output_dir=args.output_dir,
            vox_dims=args.vox_dims,
            target_dims=args.target_dims,
            sigma=args.sigma,
            color_min=args.color_min,
            color_max=args.color_max,
            use_auto_color_max=use_auto,
            vla_plan=vla_plan,
            hla_plan=hla_plan,
            sa_plan=sa_plan,
            show_colorbar=show_bar,
        )
    except Exception as exc:  # pragma: no cover - CLI友好错误输出
        parser.exit(status=1, message=f"错误: {exc}\n")

    print(f"完成处理，共生成 {len(results)} 个文件：")
    for path in results:
        print(f"  - {path}")


if __name__ == "__main__":
    main()

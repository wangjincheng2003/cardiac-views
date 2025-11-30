from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pydicom
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter, zoom

# 统一字体配置，兼容中文标题
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = True

DEFAULT_VOX_DIMS: Tuple[int, int, int] = (48, 48, 48)
DEFAULT_TARGET_DIMS: Tuple[int, int, int] = (256, 256, 256)
DEFAULT_DTYPE_CANDIDATES = (np.float32, np.float64, np.uint16, np.int16, np.uint8)
DEFAULT_COLOR_MIN: float = 0.0
DEFAULT_COLOR_MAX: float = 10000.0
DEFAULT_SIGMA: float = 4.0


@dataclass(frozen=True)
class SlicePlan:
    start: int
    stop: int
    count: int

    def indices(self) -> np.ndarray:
        return np.linspace(self.start, self.stop, self.count, dtype=int)


DEFAULT_VLA_PLAN = SlicePlan(117, 144, 8)
DEFAULT_HLA_PLAN = SlicePlan(110, 145, 8)
DEFAULT_SA_PLAN = SlicePlan(80, 160, 8)


def load_raw_volume(
    path: Path,
    vox_dims: Sequence[int] = DEFAULT_VOX_DIMS,
    target_dims: Sequence[int] = DEFAULT_TARGET_DIMS,
    dtype_candidates: Iterable[np.dtype] = DEFAULT_DTYPE_CANDIDATES,
    sigma: float = DEFAULT_SIGMA,
) -> tuple[np.ndarray, float, float]:
    """读取 .dat 原始体数据并插值到指定尺寸，返回体数据与原始最小/最大值。"""
    frame_size = int(np.prod(vox_dims))

    for dtype in dtype_candidates:
        try:
            bytes_per_frame = frame_size * np.dtype(dtype).itemsize
            with path.open("rb") as f:
                raw = f.read(int(bytes_per_frame))
            data = np.frombuffer(raw, dtype=dtype)
            if data.size != frame_size:
                continue

            vol = data.reshape(tuple(vox_dims))
            original_min, original_max = vol.min(), vol.max()
            if original_max > original_min:
                vol = (vol - original_min) / (original_max - original_min)

            zoom_factors = tuple(t / v for t, v in zip(target_dims, vox_dims))
            vol = zoom(vol, zoom_factors, order=0)
            if sigma and sigma > 0:
                vol = gaussian_filter(vol, sigma=sigma)
            return vol, float(original_min), float(original_max)
        except Exception:
            continue

    # 若全部尝试失败则报错
    raise ValueError(f"无法以候选数据类型读取 {path}")


def load_dicom_volume(
    path: Path,
    target_dims: Sequence[int] = DEFAULT_TARGET_DIMS,
    sigma: float = DEFAULT_SIGMA,
) -> tuple[np.ndarray, float, float]:
    """读取 DICOM 体数据并插值到指定尺寸，返回体数据与原始最小/最大值。"""
    ds = pydicom.dcmread(path)
    pixel_array = ds.pixel_array.astype(np.float64)

    # 应用 RescaleSlope 和 RescaleIntercept（如果存在）
    slope = getattr(ds, "RescaleSlope", 1.0)
    intercept = getattr(ds, "RescaleIntercept", 0.0)
    pixel_array = pixel_array * slope + intercept

    # 获取体数据维度
    if pixel_array.ndim == 2:
        # 单帧 DICOM，扩展为 3D
        vol = pixel_array[np.newaxis, :, :]
    elif pixel_array.ndim == 3:
        vol = pixel_array
    else:
        raise ValueError(f"不支持的 DICOM 数据维度: {pixel_array.ndim}")

    original_min, original_max = float(vol.min()), float(vol.max())

    # 归一化
    if original_max > original_min:
        vol = (vol - original_min) / (original_max - original_min)

    # 插值到目标尺寸
    current_dims = vol.shape
    zoom_factors = tuple(t / c for t, c in zip(target_dims, current_dims))
    vol = zoom(vol, zoom_factors, order=0)

    if sigma and sigma > 0:
        vol = gaussian_filter(vol, sigma=sigma)

    return vol, original_min, original_max


def load_dicom_palette(dicom_path: Path) -> LinearSegmentedColormap | None:
    """从DICOM文件读取调色板，失败返回None。"""
    try:
        ds = pydicom.dcmread(dicom_path, force=True)
        if not hasattr(ds, "RedPaletteColorLookupTableData"):
            return None

        red_lut = ds.RedPaletteColorLookupTableData
        green_lut = ds.GreenPaletteColorLookupTableData
        blue_lut = ds.BluePaletteColorLookupTableData
        red_desc = ds.RedPaletteColorLookupTableDescriptor
        bits_per_entry = red_desc[2]

        if bits_per_entry == 16:
            scale = 65535.0
            dtype = np.uint16
        else:
            scale = 255.0
            dtype = np.uint8

        red_array = np.frombuffer(red_lut, dtype=dtype).astype(np.float32) / scale
        green_array = np.frombuffer(green_lut, dtype=dtype).astype(np.float32) / scale
        blue_array = np.frombuffer(blue_lut, dtype=dtype).astype(np.float32) / scale
        colors = [
            (red_array[i], green_array[i], blue_array[i]) for i in range(len(red_array))
        ]
        return LinearSegmentedColormap.from_list("dicom_palette", colors, N=len(colors))
    except Exception:
        return None


def _get_default_palette_path() -> Path:
    """获取默认调色板路径（包内的 pet.dcm）"""
    return Path(__file__).parent / "pet.dcm"


def _resolve_cmap(palette_path: Path | None) -> LinearSegmentedColormap:
    if palette_path is None:
        palette_path = _get_default_palette_path()
    cmap = load_dicom_palette(palette_path)
    if cmap is not None:
        return cmap
    raise ValueError(f"无法加载调色板: {palette_path}")


def _derive_output_path(dat_path: Path, output_dir: Path | None) -> Path:
    output_root = Path(output_dir) if output_dir else dat_path.parent / "view"
    output_root.mkdir(parents=True, exist_ok=True)

    base_name = dat_path.stem
    if base_name.startswith("ReSampleImage_"):
        suffix = base_name.replace("ReSampleImage_", "")
        filename = f"views_{suffix}.png"
    else:
        filename = f"views_{base_name}.png"
    return output_root / filename


def _compute_display_range(
    vol: np.ndarray,
    original_min: float,
    original_max: float,
    color_min: float,
    color_max: float,
    use_auto_color_max: bool,
    vla_indices: np.ndarray,
) -> tuple[float, float, float]:
    if use_auto_color_max:
        vla_slice = np.flipud(vol[:, :, vla_indices[1]])
        vla_slice = np.rot90(vla_slice, k=1)
        upper_half = vla_slice[vla_slice.shape[0] // 2 :, :]
        vla_slice_2_upper_max = float(np.max(upper_half))
        dynamic_color_max = (
            vla_slice_2_upper_max * (original_max - original_min) + original_min
        )
    else:
        dynamic_color_max = color_max

    if original_max > original_min:
        vmin_display = (color_min - original_min) / (original_max - original_min)
        vmax_display = (dynamic_color_max - original_min) / (
            original_max - original_min
        )
        vmin_display = max(0.0, min(1.0, vmin_display))
        vmax_display = max(0.0, min(1.0, vmax_display))
    else:
        vmin_display, vmax_display = 0.0, 1.0

    return vmin_display, vmax_display, dynamic_color_max


def make_figure(
    vol: np.ndarray,
    original_min: float,
    original_max: float,
    save_path: Path,
    *,
    cmap: LinearSegmentedColormap,
    color_min: float = DEFAULT_COLOR_MIN,
    color_max: float = DEFAULT_COLOR_MAX,
    use_auto_color_max: bool = False,
    vla_plan: SlicePlan = DEFAULT_VLA_PLAN,
    hla_plan: SlicePlan = DEFAULT_HLA_PLAN,
    sa_plan: SlicePlan = DEFAULT_SA_PLAN,
    show_colorbar: bool = False,
) -> None:
    vla_indices = vla_plan.indices()
    hla_indices = hla_plan.indices()
    sa_indices = sa_plan.indices()

    vmin_display, vmax_display, dynamic_color_max = _compute_display_range(
        vol,
        original_min,
        original_max,
        color_min,
        color_max,
        use_auto_color_max,
        vla_indices,
    )

    fig, axes = plt.subplots(3, 8, figsize=(20, 8), dpi=150, facecolor="black")
    fig.patch.set_facecolor("black")
    for ax in axes.flat:
        ax.set_facecolor("black")

    for i, slice_idx in enumerate(vla_indices):
        vla_slice = np.rot90(np.flipud(vol[:, :, slice_idx]), k=1)
        im = axes[0, i].imshow(
            vla_slice,
            cmap=cmap,
            origin="lower",
            vmin=vmin_display,
            vmax=vmax_display,
            aspect="equal",
        )
        axes[0, i].set_title(f"VLA Slice{i+1}", fontsize=10, pad=5, color="white")
        axes[0, i].axis("off")

    for i, slice_idx in enumerate(hla_indices):
        hla_slice = np.rot90(vol[:, slice_idx, :].T, k=1)
        axes[1, i].imshow(
            hla_slice,
            cmap=cmap,
            origin="lower",
            vmin=vmin_display,
            vmax=vmax_display,
            aspect="equal",
        )
        axes[1, i].set_title(f"HLA Slice{i+1}", fontsize=10, pad=5, color="white")
        axes[1, i].axis("off")

    for i, slice_idx in enumerate(sa_indices):
        sa_slice = vol[slice_idx, :, :]
        axes[2, i].imshow(
            sa_slice,
            cmap=cmap,
            origin="upper",
            vmin=vmin_display,
            vmax=vmax_display,
            aspect="equal",
        )
        axes[2, i].set_title(f"SA Slice{i+1}", fontsize=10, pad=5, color="white")
        axes[2, i].axis("off")

    axes[0, 0].text(
        -0.15,
        0.5,
        "VLA",
        transform=axes[0, 0].transAxes,
        rotation=90,
        fontsize=14,
        fontweight="bold",
        va="center",
        color="white",
    )
    axes[1, 0].text(
        -0.15,
        0.5,
        "HLA",
        transform=axes[1, 0].transAxes,
        rotation=90,
        fontsize=14,
        fontweight="bold",
        va="center",
        color="white",
    )
    axes[2, 0].text(
        -0.15,
        0.5,
        "SA",
        transform=axes[2, 0].transAxes,
        rotation=90,
        fontsize=14,
        fontweight="bold",
        va="center",
        color="white",
    )

    plt.tight_layout()

    if show_colorbar:
        cbar = fig.colorbar(
            im, ax=axes.ravel().tolist(), shrink=0.8, aspect=40, pad=0.02, fraction=0.05
        )
        tick_interval = max(500, int(dynamic_color_max / 10))
        desired_fixed_values = list(
            range(int(color_min), int(dynamic_color_max) + 1, tick_interval)
        )
        normalized_ticks = []
        fixed_ticks = []
        for val in desired_fixed_values:
            if vmax_display > vmin_display:
                normalized_val = vmin_display + (val - color_min) / (
                    dynamic_color_max - color_min
                ) * (vmax_display - vmin_display)
            else:
                normalized_val = vmin_display
            if 0 <= normalized_val <= 1:
                normalized_ticks.append(normalized_val)
                fixed_ticks.append(val)
        cbar.set_ticks(normalized_ticks)
        cbar.set_ticklabels([f"{val}" for val in fixed_ticks])
        cbar.ax.tick_params(colors="white", labelsize=10)
        cbar.ax.yaxis.label.set_color("white")
        label_suffix = "Auto" if use_auto_color_max else "Manual"
        cbar.set_label(
            f"Intensity ({color_min:.0f}~{dynamic_color_max:.0f} {label_suffix} Range)",
            color="white",
            fontsize=12,
        )

    fig.savefig(save_path, bbox_inches="tight", dpi=150, facecolor="black", edgecolor="black")
    plt.close(fig)


def process_dat_file(
    dat_path: Path,
    *,
    palette_path: Path | None = None,
    output_dir: Path | None = None,
    vox_dims: Sequence[int] = DEFAULT_VOX_DIMS,
    target_dims: Sequence[int] = DEFAULT_TARGET_DIMS,
    dtype_candidates: Iterable[np.dtype] = DEFAULT_DTYPE_CANDIDATES,
    sigma: float = DEFAULT_SIGMA,
    color_min: float = DEFAULT_COLOR_MIN,
    color_max: float = DEFAULT_COLOR_MAX,
    use_auto_color_max: bool = False,
    vla_plan: SlicePlan = DEFAULT_VLA_PLAN,
    hla_plan: SlicePlan = DEFAULT_HLA_PLAN,
    sa_plan: SlicePlan = DEFAULT_SA_PLAN,
    show_colorbar: bool = False,
) -> Path:
    vol, original_min, original_max = load_raw_volume(
        dat_path,
        vox_dims=vox_dims,
        target_dims=target_dims,
        dtype_candidates=dtype_candidates,
        sigma=sigma,
    )

    save_path = _derive_output_path(dat_path, output_dir)
    cmap = _resolve_cmap(palette_path)

    make_figure(
        vol,
        original_min,
        original_max,
        save_path,
        cmap=cmap,
        color_min=color_min,
        color_max=color_max,
        use_auto_color_max=use_auto_color_max,
        vla_plan=vla_plan,
        hla_plan=hla_plan,
        sa_plan=sa_plan,
        show_colorbar=show_colorbar,
    )
    return save_path


def process_dicom_file(
    dicom_path: Path,
    *,
    palette_path: Path | None = None,
    output_dir: Path | None = None,
    target_dims: Sequence[int] = DEFAULT_TARGET_DIMS,
    sigma: float = DEFAULT_SIGMA,
    color_min: float = DEFAULT_COLOR_MIN,
    color_max: float = DEFAULT_COLOR_MAX,
    use_auto_color_max: bool = False,
    vla_plan: SlicePlan = DEFAULT_VLA_PLAN,
    hla_plan: SlicePlan = DEFAULT_HLA_PLAN,
    sa_plan: SlicePlan = DEFAULT_SA_PLAN,
    show_colorbar: bool = False,
) -> Path:
    vol, original_min, original_max = load_dicom_volume(
        dicom_path,
        target_dims=target_dims,
        sigma=sigma,
    )

    save_path = _derive_output_path(dicom_path, output_dir)
    cmap = _resolve_cmap(palette_path)

    make_figure(
        vol,
        original_min,
        original_max,
        save_path,
        cmap=cmap,
        color_min=color_min,
        color_max=color_max,
        use_auto_color_max=use_auto_color_max,
        vla_plan=vla_plan,
        hla_plan=hla_plan,
        sa_plan=sa_plan,
        show_colorbar=show_colorbar,
    )
    return save_path


DICOM_EXTENSIONS = {".dcm", ".dicom"}


def _is_dicom_file(path: Path) -> bool:
    """判断文件是否为 DICOM 格式"""
    return path.suffix.lower() in DICOM_EXTENSIONS


def generate_views(
    input_path: Path,
    *,
    palette_path: Path | None = None,
    output_dir: Path | None = None,
    vox_dims: Sequence[int] = DEFAULT_VOX_DIMS,
    target_dims: Sequence[int] = DEFAULT_TARGET_DIMS,
    dtype_candidates: Iterable[np.dtype] = DEFAULT_DTYPE_CANDIDATES,
    sigma: float = DEFAULT_SIGMA,
    color_min: float = DEFAULT_COLOR_MIN,
    color_max: float = DEFAULT_COLOR_MAX,
    use_auto_color_max: bool = False,
    vla_plan: SlicePlan = DEFAULT_VLA_PLAN,
    hla_plan: SlicePlan = DEFAULT_HLA_PLAN,
    sa_plan: SlicePlan = DEFAULT_SA_PLAN,
    show_colorbar: bool = False,
) -> List[Path]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"输入路径不存在: {path}")

    if path.is_dir():
        dat_files = sorted(path.glob("*.dat"))
        dcm_files = sorted(path.glob("*.dcm")) + sorted(path.glob("*.dicom"))
        all_files = dat_files + dcm_files
        if not all_files:
            raise FileNotFoundError(f"目录中未找到 .dat 或 .dcm 文件: {path}")
    else:
        all_files = [path]

    results: List[Path] = []
    for file_path in all_files:
        if _is_dicom_file(file_path):
            save_path = process_dicom_file(
                file_path,
                palette_path=palette_path,
                output_dir=output_dir,
                target_dims=target_dims,
                sigma=sigma,
                color_min=color_min,
                color_max=color_max,
                use_auto_color_max=use_auto_color_max,
                vla_plan=vla_plan,
                hla_plan=hla_plan,
                sa_plan=sa_plan,
                show_colorbar=show_colorbar,
            )
        else:
            save_path = process_dat_file(
                file_path,
                palette_path=palette_path,
                output_dir=output_dir,
                vox_dims=vox_dims,
                target_dims=target_dims,
                dtype_candidates=dtype_candidates,
                sigma=sigma,
                color_min=color_min,
                color_max=color_max,
                use_auto_color_max=use_auto_color_max,
                vla_plan=vla_plan,
                hla_plan=hla_plan,
                sa_plan=sa_plan,
                show_colorbar=show_colorbar,
            )
        results.append(save_path)
    return results

# cardiac-views

从体数据生成心脏 VLA/HLA/SA 三视图。

## 安装

```bash
pip install git+https://github.com/wangjincheng2003/cardiac-views.git
```

## 使用

### 命令行

```bash
# 处理单个文件
cardiac-views input.dat

# 处理目录下所有 .dat 文件
cardiac-views ./dat_folder

# 指定输出目录和调色板
cardiac-views input.dat -o ./output -p palette.dcm

# 自动颜色范围
cardiac-views input.dat --auto-color-max --colorbar
```

### Python API

```python
from pathlib import Path
from cardiac_views import generate_views, process_dat_file

# 处理目录或单个文件
results = generate_views(Path("./dat"))

# 自定义参数
results = generate_views(
    Path("input.dat"),
    output_dir=Path("./output"),
    color_min=0,
    color_max=3000,
    use_auto_color_max=True,
    show_colorbar=True,
)
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--vox-dims` | 48 48 48 | 原始体素尺寸 |
| `--target-dims` | 256 256 256 | 插值后尺寸 |
| `--sigma` | 4.0 | 高斯平滑 sigma |
| `--color-min` | 0.0 | 颜色最小值 |
| `--color-max` | 4000.0 | 颜色最大值 |
| `--auto-color-max` | - | 自动计算颜色上限 |
| `--colorbar` | - | 显示颜色条 |
| `--vla/--hla/--sa` | - | 切片范围 (start stop count) |

## License

MIT

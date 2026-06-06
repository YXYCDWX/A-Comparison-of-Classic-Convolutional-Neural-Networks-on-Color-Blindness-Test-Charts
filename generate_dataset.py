#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
色盲测试图表数据集生成器
Ishihara-style Color Blindness Test Chart Dataset Generator

核心思路：
1. 用0/1矩阵定义数字图案（1=数字区域，0=背景区域）
2. 在整个画面上密集铺设小圆点
3. 数字区域的圆点使用一组颜色（如绿色系）
4. 背景区域的圆点使用另一组颜色（如红色系）
5. 两组颜色在色相上接近但可区分，模拟真实石原色盲检测图

生成格式与原数据集完全一致：
- 图像尺寸: 128x128 RGB PNG
- 目录结构: {类别}/{类别}_{序号}.png
- 类别: 0-9
"""

import os
import sys
import io
import math
import random
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Windows控制台UTF-8编码
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


# ============================================================
# 数字矩阵定义 (7x5 点阵，1=数字区域, 0=背景区域)
# ============================================================
DIGIT_PATTERNS = {
    0: [
        [0, 1, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [0, 1, 1, 1, 0],
    ],
    1: [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [1, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
    ],
    2: [
        [0, 1, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 1, 1, 0],
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
    ],
    3: [
        [0, 1, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [0, 1, 1, 1, 0],
    ],
    4: [
        [0, 0, 0, 1, 0],
        [0, 0, 1, 1, 0],
        [0, 1, 0, 1, 0],
        [1, 0, 0, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0],
    ],
    5: [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0],
        [1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [0, 1, 1, 1, 0],
    ],
    6: [
        [0, 1, 1, 1, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 1, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [0, 1, 1, 1, 0],
    ],
    7: [
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ],
    8: [
        [0, 1, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [0, 1, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [0, 1, 1, 1, 0],
    ],
    9: [
        [0, 1, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 1, 1, 1, 0],
    ],
}


@dataclass
class ColorScheme:
    """颜色方案：前景(数字)和背景的颜色范围"""
    fg_hue_range: Tuple[float, float]    # 前景色相范围
    fg_sat_range: Tuple[float, float]    # 前景饱和度范围
    fg_val_range: Tuple[float, float]    # 前景明度范围
    bg_hue_range: Tuple[float, float]    # 背景色相范围
    bg_sat_range: Tuple[float, float]    # 背景饱和度范围
    bg_val_range: Tuple[float, float]    # 背景明度范围


# 预定义颜色方案（模拟不同类型的色盲测试）
COLOR_SCHEMES = [
    # 方案0: 红绿对比（经典石原测试 - 红绿色盲）
    ColorScheme(
        fg_hue_range=(0.25, 0.42),   # 绿色系
        fg_sat_range=(0.4, 0.8),
        fg_val_range=(0.4, 0.75),
        bg_hue_range=(0.95, 0.08),   # 红色系（跨越0/1边界）
        bg_sat_range=(0.4, 0.8),
        bg_val_range=(0.4, 0.75),
    ),
    # 方案1: 绿棕对比
    ColorScheme(
        fg_hue_range=(0.22, 0.38),
        fg_sat_range=(0.3, 0.7),
        fg_val_range=(0.35, 0.65),
        bg_hue_range=(0.06, 0.12),
        bg_sat_range=(0.4, 0.75),
        bg_val_range=(0.35, 0.65),
    ),
    # 方案2: 橙绿对比
    ColorScheme(
        fg_hue_range=(0.28, 0.42),
        fg_sat_range=(0.35, 0.75),
        fg_val_range=(0.4, 0.7),
        bg_hue_range=(0.05, 0.12),
        bg_sat_range=(0.45, 0.8),
        bg_val_range=(0.45, 0.75),
    ),
    # 方案3: 黄绿对比
    ColorScheme(
        fg_hue_range=(0.12, 0.22),
        fg_sat_range=(0.4, 0.75),
        fg_val_range=(0.5, 0.8),
        bg_hue_range=(0.28, 0.40),
        bg_sat_range=(0.3, 0.65),
        bg_val_range=(0.35, 0.65),
    ),
    # 方案4: 紫红对比（蓝紫色盲测试）
    ColorScheme(
        fg_hue_range=(0.72, 0.85),
        fg_sat_range=(0.3, 0.65),
        fg_val_range=(0.35, 0.65),
        bg_hue_range=(0.92, 0.02),
        bg_sat_range=(0.35, 0.7),
        bg_val_range=(0.4, 0.7),
    ),
]


def hsv_to_rgb(h: float, s: float, v: float) -> Tuple[int, int, int]:
    """HSV转RGB，h范围[0,1], s范围[0,1], v范围[0,1]"""
    # 处理h跨越0/1边界的情况
    h = h % 1.0
    c = v * s
    x = c * (1 - abs((h * 6) % 2 - 1))
    m = v - c

    if h < 1/6:
        r, g, b = c, x, 0
    elif h < 2/6:
        r, g, b = x, c, 0
    elif h < 3/6:
        r, g, b = 0, c, x
    elif h < 4/6:
        r, g, b = 0, x, c
    elif h < 5/6:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x

    return (
        int((r + m) * 255),
        int((g + m) * 255),
        int((b + m) * 255)
    )


def random_color_in_range(
    hue_range: Tuple[float, float],
    sat_range: Tuple[float, float],
    val_range: Tuple[float, float]
) -> Tuple[int, int, int]:
    """在指定HSV范围内随机生成一个颜色，加入额外抖动增加多样性"""
    h_min, h_max = hue_range
    s_min, s_max = sat_range
    v_min, v_max = val_range

    # 处理色相跨越0/1边界
    if h_max < h_min:
        h_max += 1.0
    h = random.uniform(h_min, h_max) % 1.0
    s = random.uniform(s_min, s_max)
    v = random.uniform(v_min, v_max)

    # 额外随机抖动，增加颜色多样性
    h = (h + random.gauss(0, 0.02)) % 1.0
    s = max(0.05, min(1.0, s + random.gauss(0, 0.08)))
    v = max(0.1, min(0.95, v + random.gauss(0, 0.06)))

    return hsv_to_rgb(h, s, v)


def create_digit_mask(digit: int, img_size: int = 128) -> np.ndarray:
    """
    根据数字的点阵模式创建0/1掩码矩阵
    
    Args:
        digit: 数字0-9
        img_size: 图像尺寸
    
    Returns:
        0/1矩阵，1表示数字区域，0表示背景区域
    """
    pattern = DIGIT_PATTERNS[digit]
    pattern_rows = len(pattern)
    pattern_cols = len(pattern[0])

    # 计算数字在图像中的位置和缩放
    # 数字占据图像中心区域，留出边距
    margin = img_size * 0.15
    digit_area_size = img_size - 2 * margin

    cell_w = digit_area_size / pattern_cols
    cell_h = digit_area_size / pattern_rows

    mask = np.zeros((img_size, img_size), dtype=np.int8)

    for row_idx, row in enumerate(pattern):
        for col_idx, val in enumerate(row):
            if val == 1:
                # 计算该点阵单元在图像中的区域
                x_start = int(margin + col_idx * cell_w)
                x_end = int(margin + (col_idx + 1) * cell_w)
                y_start = int(margin + row_idx * cell_h)
                y_end = int(margin + (row_idx + 1) * cell_h)

                # 添加随机扰动使边缘不规则
                jitter = int(cell_w * 0.2)
                x_start = max(0, x_start + random.randint(-jitter, 0))
                x_end = min(img_size, x_end + random.randint(0, jitter))
                y_start = max(0, y_start + random.randint(-jitter, 0))
                y_end = min(img_size, y_end + random.randint(0, jitter))

                mask[y_start:y_end, x_start:x_end] = 1

    return mask


def generate_ishihara_chart(
    digit: int,
    img_size: int = 128,
    color_scheme: Optional[ColorScheme] = None,
    dot_size_range: Tuple[int, int] = (3, 7),
    bg_color: Tuple[int, int, int] = (245, 243, 245),
) -> Image.Image:
    """
    生成一张石原色盲检测图
    
    核心流程：
    1. 创建0/1掩码矩阵定义数字区域
    2. 在整个画面上密集铺设随机大小的圆点
    3. 数字区域用前景色系，背景区域用背景色系
    4. 每个圆点内部像素带有独立的颜色抖动，增加颜色多样性
    5. 圆形裁剪，模拟真实石原测试图
    
    Args:
        digit: 要显示的数字(0-9)
        img_size: 图像尺寸
        color_scheme: 颜色方案，None则随机选择
        dot_size_range: 圆点半径范围
        bg_color: 画布底色
    
    Returns:
        PIL Image对象
    """
    if color_scheme is None:
        color_scheme = random.choice(COLOR_SCHEMES)

    # 步骤1: 创建数字掩码
    mask = create_digit_mask(digit, img_size)

    # 步骤2: 用numpy数组直接操作（比PIL逐像素灵活得多）
    canvas = np.full((img_size, img_size, 3), bg_color, dtype=np.uint8)
    painted = np.zeros((img_size, img_size), dtype=bool)  # 记录已绘制区域

    # 步骤3: 生成圆点网格位置（带随机偏移）
    min_dot_r, max_dot_r = dot_size_range
    avg_dot_r = (min_dot_r + max_dot_r) / 2
    grid_step = max(3, int(avg_dot_r * 1.3))

    dots = []
    for y in range(0, img_size + grid_step, grid_step):
        for x in range(0, img_size + grid_step, grid_step):
            ox = x + random.randint(-grid_step // 2, grid_step // 2)
            oy = y + random.randint(-grid_step // 2, grid_step // 2)
            r = random.randint(min_dot_r, max_dot_r)
            if 0 <= ox < img_size and 0 <= oy < img_size:
                dots.append((ox, oy, r))

    random.shuffle(dots)

    # 步骤4: 绘制每个圆点（逐像素带颜色抖动）
    for (cx, cy, r) in dots:
        # 判断圆点中心在哪个区域
        region = mask[cy, cx]

        # 生成该圆点的基础颜色
        if region == 1:
            base_color = random_color_in_range(
                color_scheme.fg_hue_range,
                color_scheme.fg_sat_range,
                color_scheme.fg_val_range
            )
        else:
            base_color = random_color_in_range(
                color_scheme.bg_hue_range,
                color_scheme.bg_sat_range,
                color_scheme.bg_val_range
            )

        # 计算圆点覆盖的像素区域
        y_min = max(0, cy - r)
        y_max = min(img_size, cy + r + 1)
        x_min = max(0, cx - r)
        x_max = min(img_size, cx + r + 1)

        # 生成网格坐标
        yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
        # 计算到圆心的距离
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        # 圆内掩码
        circle_mask_local = dist <= r

        if not circle_mask_local.any():
            continue

        # 为圆内每个像素生成带抖动的颜色
        n_pixels = circle_mask_local.sum()
        # 基础颜色 + 每像素独立随机抖动（模拟真实石原图的颜色丰富度）
        jitter = np.random.randint(-20, 21, size=(n_pixels, 3))
        pixel_colors = np.array(base_color, dtype=np.int16) + jitter
        pixel_colors = np.clip(pixel_colors, 0, 255).astype(np.uint8)

        # 写入画布
        local_y = yy[circle_mask_local]
        local_x = xx[circle_mask_local]
        canvas[local_y, local_x] = pixel_colors

    # 步骤5: 圆形裁剪（模拟真实石原测试图的圆形边框）
    center = img_size // 2
    radius = int(img_size * 0.47)
    yy, xx = np.mgrid[0:img_size, 0:img_size]
    circle_mask_global = (xx - center) ** 2 + (yy - center) ** 2 <= radius ** 2
    # 圆外区域填充底色
    canvas[~circle_mask_global] = bg_color

    return Image.fromarray(canvas)


def safe_augment(img: Image.Image, digit: int) -> Image.Image:
    """
    对数字图像进行安全的数据增强
    
    安全的增强（不改变数字语义）：
    - 颜色抖动（亮度/对比度/饱和度微调）
    - 轻微高斯噪声
    - 轻微缩放（0.92~1.08倍）
    - 轻微平移（±5%）
    - 小幅斜向拉伸（0.85~1.15倍，让数字变胖/变瘦/变长/变短）
    
    危险的增强（会改变数字语义，绝对禁止）：
    - 水平翻转：3→反3(像E)，9→反9
    - 垂直翻转：9→6，6→9
    - 180°旋转：9→6，6→9
    - 90°/270°旋转：所有数字都变形
    - 大角度旋转(>15°)：数字变形无法辨认
    """
    arr = np.array(img)
    
    # 增强1: 亮度/对比度微调（±10%）
    brightness = random.uniform(0.9, 1.1)
    arr = np.clip(arr.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
    
    # 增强2: 轻微高斯噪声
    noise_sigma = random.uniform(0, 3)
    if noise_sigma > 0:
        noise = np.random.normal(0, noise_sigma, arr.shape).astype(np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # 增强3: 斜向拉伸（让数字变胖/变瘦/变长/变短）
    # 沿随机角度方向做非等比缩放，数字形状有几何变化但不改变语义
    stretch_angle = random.uniform(0, 180)  # 拉伸方向（任意角度）
    stretch_factor = random.uniform(0.85, 1.15)  # 拉伸倍数
    if abs(stretch_factor - 1.0) > 0.02:
        img_pil = Image.fromarray(arr)
        h, w = img_pil.size
        # 将图像旋转到拉伸方向，沿水平拉伸，再旋转回来
        rad = math.radians(stretch_angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        # 旋转→水平拉伸→旋转回来，等价于沿任意方向拉伸
        # 仿射矩阵：R(-θ) · Scale(factor,1) · R(θ)
        # 展开为六元组 (a,b,c,d,e,f)
        a = cos_a * cos_a * stretch_factor + sin_a * sin_a
        b = cos_a * sin_a * (stretch_factor - 1)
        c = (w / 2) * (1 - a) - (h / 2) * b
        d = cos_a * sin_a * (stretch_factor - 1)
        e = sin_a * sin_a * stretch_factor + cos_a * cos_a
        f = (h / 2) * (1 - e) - (w / 2) * d
        img_pil = img_pil.transform(
            img_pil.size,
            Image.AFFINE,
            (a, b, c, d, e, f),
            resample=Image.BILINEAR,
            fillcolor=tuple(arr[0, 0])
        )
        arr = np.array(img_pil)
    
    # 增强4: 轻微缩放 + 平移（模拟数字位置偏移）
    scale = random.uniform(0.92, 1.08)
    if abs(scale - 1.0) > 0.01:
        h, w = arr.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        img_scaled = Image.fromarray(arr).resize((new_w, new_h), Image.BILINEAR)
        # 放回原尺寸，随机偏移
        canvas = Image.new('RGB', (w, h), tuple(arr[0, 0]))
        offset_x = random.randint(-int(w * 0.05), int(w * 0.05))
        offset_y = random.randint(-int(h * 0.05), int(h * 0.05))
        # 居中放置 + 偏移
        paste_x = (w - new_w) // 2 + offset_x
        paste_y = (h - new_h) // 2 + offset_y
        canvas.paste(img_scaled, (paste_x, paste_y))
        arr = np.array(canvas)
    
    return Image.fromarray(arr)


# 危险增强黑名单：明确禁止的操作及其原因
DANGEROUS_AUGMENTATIONS = {
    'horizontal_flip': '9→反9, 3→反3(像E), 2→反2',
    'vertical_flip':   '9→6, 6→9, 3→反3',
    'rotate_180':      '9→6, 6→9',
    'rotate_90':       '所有数字变形',
    'rotate_270':      '所有数字变形',
    'large_rotation':  '大角度旋转(>15°)使数字无法辨认',
}


def generate_dataset(
    output_dir: str,
    num_per_class: int = 100,
    img_size: int = 128,
    digits: Optional[List[int]] = None,
    seed: Optional[int] = None,
    augment: bool = False,
    augment_ratio: float = 0.3,
):
    """
    生成完整数据集
    
    Args:
        output_dir: 输出目录
        num_per_class: 每个类别生成的图片数量
        img_size: 图像尺寸
        digits: 要生成的数字列表，None则生成0-9
        seed: 随机种子
        augment: 是否对部分图片做安全的数据增强
        augment_ratio: 增强图片占比(0~1)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if digits is None:
        digits = list(range(10))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    total = len(digits) * num_per_class
    count = 0

    print(f"{'='*60}")
    print(f"  色盲测试图表数据集生成器")
    print(f"{'='*60}")
    print(f"  输出目录: {output_path}")
    print(f"  图像尺寸: {img_size}x{img_size}")
    print(f"  类别数量: {len(digits)}")
    print(f"  每类数量: {num_per_class}")
    print(f"  总计图片: {total}")
    print(f"  数据增强: {'开启(' + str(int(augment_ratio*100)) + '%)' if augment else '关闭'}")
    print(f"  随机种子: {seed or '未设置'}")
    if augment:
        print(f"  [注意] 禁止翻转/旋转: 9倒转=6, 6倒转=9")
    print(f"{'='*60}\n")

    for digit in digits:
        digit_dir = output_path / str(digit)
        digit_dir.mkdir(exist_ok=True)

        print(f"[生成] 类别 {digit} ...", end=" ", flush=True)

        for i in range(num_per_class):
            # 随机选择颜色方案，增加多样性
            scheme = random.choice(COLOR_SCHEMES)

            # 随机变化圆点大小范围
            base_dot = random.randint(3, 5)
            dot_range = (base_dot, base_dot + random.randint(2, 4))

            # 随机背景底色（浅色系）
            bg_brightness = random.randint(235, 250)
            bg_color = (
                bg_brightness + random.randint(-5, 5),
                bg_brightness + random.randint(-5, 5),
                bg_brightness + random.randint(-5, 5)
            )
            bg_color = tuple(max(0, min(255, c)) for c in bg_color)

            # 生成图像
            img = generate_ishihara_chart(
                digit=digit,
                img_size=img_size,
                color_scheme=scheme,
                dot_size_range=dot_range,
                bg_color=bg_color,
            )

            # 安全的数据增强（不翻转、不旋转）
            if augment and random.random() < augment_ratio:
                img = safe_augment(img, digit)

            # 保存（命名格式与原数据集一致: {类别}_{序号}.png）
            filename = f"{digit}_{i}.png"
            filepath = digit_dir / filename
            img.save(filepath, 'PNG')

            count += 1

        print(f"完成 ({num_per_class} 张)")

    print(f"\n{'='*60}")
    print(f"  生成完成！")
    print(f"  总计: {count} 张图片")
    print(f"  保存位置: {output_path}")
    print(f"{'='*60}")

    # 验证生成结果
    verify_dataset(output_path, digits, num_per_class)


def verify_dataset(dataset_dir: Path, digits: List[int], expected_count: int):
    """验证生成的数据集格式是否正确"""
    print(f"\n[验证] 检查数据集格式...")

    errors = []

    for digit in digits:
        digit_dir = dataset_dir / str(digit)

        if not digit_dir.exists():
            errors.append(f"类别 {digit} 目录不存在")
            continue

        files = list(digit_dir.glob('*.png'))
        if len(files) != expected_count:
            errors.append(f"类别 {digit}: 期望 {expected_count} 张，实际 {len(files)} 张")

        # 抽查第一张图片的格式
        if files:
            sample = Image.open(files[0])
            if sample.size != (128, 128):
                errors.append(f"类别 {digit}: 图像尺寸 {sample.size}，期望 (128, 128)")
            if sample.mode != 'RGB':
                errors.append(f"类别 {digit}: 图像模式 {sample.mode}，期望 RGB")

    if errors:
        print("[验证] 发现问题:")
        for e in errors:
            print(f"  ✗ {e}")
    else:
        print("[验证] 全部通过 ✓")
        print(f"  - 目录结构: {len(digits)} 个类别目录")
        print(f"  - 每类数量: {expected_count} 张")
        print(f"  - 图像格式: 128x128 RGB PNG")


def main():
    parser = argparse.ArgumentParser(
        description='色盲测试图表数据集生成器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python generate_dataset.py                           # 默认生成1000张(每类100)
  python generate_dataset.py -n 6000                   # 生成60000张(每类6000，与原数据集等量)
  python generate_dataset.py -n 50 -d 0 1 2            # 仅生成0/1/2三个类别
  python generate_dataset.py -o ./my_data -n 200       # 指定输出目录
  python generate_dataset.py --seed 42                 # 固定随机种子
  python generate_dataset.py --augment                 # 开启安全数据增强(不翻转/旋转)
  python generate_dataset.py --augment --aug-ratio 0.5 # 开启增强，50%图片做增强

注意:
  数字图像禁止翻转/旋转！9倒转=6，6倒转=9，3翻转像E
  本工具仅使用安全的增强：亮度抖动、高斯噪声、轻微缩放/平移
        """
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='输出目录 (默认: 项目data/ishihara_dataset/generated/)'
    )
    parser.add_argument(
        '-n', '--num-per-class',
        type=int,
        default=100,
        help='每个类别生成的图片数量 (默认: 100)'
    )
    parser.add_argument(
        '-d', '--digits',
        type=int,
        nargs='+',
        default=None,
        help='要生成的数字类别 (默认: 0-9)'
    )
    parser.add_argument(
        '-s', '--size',
        type=int,
        default=128,
        help='图像尺寸 (默认: 128)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='随机种子 (用于可重复生成)'
    )
    parser.add_argument(
        '--augment',
        action='store_true',
        help='开启安全的数据增强 (不翻转/旋转，仅亮度抖动+噪声+轻微缩放)'
    )
    parser.add_argument(
        '--aug-ratio',
        type=float,
        default=0.3,
        help='增强图片占比 (默认: 0.3)'
    )

    args = parser.parse_args()

    # 默认输出到项目的data目录下
    if args.output is None:
        project_root = Path(__file__).parent
        args.output = str(project_root / 'data' / 'ishihara_dataset' / 'generated')

    generate_dataset(
        output_dir=args.output,
        num_per_class=args.num_per_class,
        img_size=args.size,
        digits=args.digits,
        seed=args.seed,
        augment=args.augment,
        augment_ratio=args.aug_ratio,
    )


if __name__ == "__main__":
    main()

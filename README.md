# GradientAV

<div align="center">

**Gradient-Preserving Retinal Artery-Vein Classification**

基于梯度保留的视网膜动静脉分类后处理方法

[![Python](https://img.shields.io/badge/Python-3.7+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.19+-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

[English](#english) | [中文](#中文)

</div>

---

## English

### Overview

GradientAV is a post-processing tool for retinal vessel segmentation that converts neural network predictions (multi-colored: red/blue/green/purple/cyan) into standardized artery-vein classification maps (red=artery, blue=vein, black=background).

**Key Innovation**: Instead of the traditional "binarize-then-smooth" paradigm, we preserve the original gradient information from neural network outputs, achieving naturally smooth vessel edges without shape distortion.

### Problem & Motivation

| Traditional Methods | Problems |
|---------------------|----------|
| Morphological smoothing | Jagged edges remain, over-processing causes deformation |
| Gaussian blur + threshold | Vessels become thicker, details lost |
| B-spline contour interpolation | Computationally complex, prone to distortion |
| Super-sampling anti-aliasing | Severe structural distortion |

**Root Cause**: All methods follow the **"binarize → smooth"** paradigm. Once hard edges are created, subsequent smoothing inevitably leads to information loss or deformation.

### Our Approach

**Core Insight**: Neural network outputs inherently contain natural edge gradients (from softmax probabilities or sub-pixel responses). This gradient IS the natural anti-aliasing.

```
Input Image → HSV Conversion → Soft Mask Extraction → Hue Classification → Gradient Modulation → Output
                    ↓
         Preserve original brightness gradient (NO binarization)
```

**Mathematical Formulation**:

For input image $I$ with HSV components $(H, S, V)$, output image $O$:

$$O_R = \alpha(H) \cdot \frac{V}{255} \cdot 255$$

$$O_B = (1 - \alpha(H)) \cdot \frac{V}{255} \cdot 255$$

Where $\alpha(H)$ is the hue-based artery weight function with smooth boundary transitions.

### Installation

```bash
git clone https://github.com/Changhuaishui/GradientAV.git
cd GradientAV
pip install -r requirements.txt
```

### Usage

**Single file:**
```bash
python gradient_av.py -i input.png -o output.png
```

**Batch processing:**
```bash
python gradient_av.py -i ./input_folder -o ./output_folder
```

**With comparison output:**
```bash
python gradient_av.py -i input.png -o output.png --compare
```

**Parameters:**
| Parameter | Description | Default |
|-----------|-------------|---------|
| `-i, --input` | Input file or directory | Required |
| `-o, --output` | Output file or directory | Required |
| `-c, --compare` | Generate side-by-side comparison | False |
| `-m, --min-intensity` | Minimum intensity threshold | 5 |

### Results

| Metric | Traditional Methods | GradientAV |
|--------|---------------------|------------|
| Edge Smoothness | Obvious jagging | Natural gradient |
| Vessel Shape Preservation | Deformation/thickening | Fully preserved |
| Processing Speed | Slow (multiple iterations) | Fast (single mapping) |

---

## 中文

### 概述

GradientAV 是一个视网膜血管分割的后处理工具，用于将神经网络预测结果（多色：红/蓝/绿/紫/青）转换为标准化的动静脉分类图（红色=动脉，蓝色=静脉，黑色=背景）。

**核心创新**：摒弃传统的"二值化-再平滑"范式，直接保留神经网络输出的梯度信息，实现自然平滑的血管边缘且不产生形变。

### 问题与动机

| 传统方法 | 问题 |
|----------|------|
| 形态学平滑（开闭运算） | 边缘仍有锯齿，过度处理导致血管变形 |
| 高斯模糊 + 阈值化 | 血管变粗，细节丢失 |
| 轮廓B样条插值 | 计算复杂，易产生形变 |
| 超采样抗锯齿 | 血管结构严重失真 |

**根本原因**：所有方法都遵循 **"二值化→平滑"** 范式。硬边缘一旦产生，后续平滑必然导致信息损失或形变。

### 我们的方法

**核心洞察**：神经网络的原始预测输出本身就包含自然的边缘渐变信息（来自 softmax 概率或网络的亚像素响应），这种渐变就是天然的抗锯齿。

```
输入图像 → HSV转换 → 软掩码提取 → 色相分类 → 梯度调制 → 输出
                ↓
         保留原始亮度渐变（不二值化）
```

### 安装

```bash
git clone https://github.com/Changhuaishui/GradientAV.git
cd GradientAV
pip install -r requirements.txt
```

### 使用方法

**单文件处理：**
```bash
python gradient_av.py -i input.png -o output.png
```

**批量处理：**
```bash
python gradient_av.py -i ./input_folder -o ./output_folder
```

**生成对比图：**
```bash
python gradient_av.py -i input.png -o output.png --compare
```

### 实验结果

| 指标 | 传统方法 | GradientAV |
|------|---------|------------|
| 边缘平滑度 | 锯齿明显 | 自然渐变 |
| 血管形态保持 | 变形/变粗 | 完全保持 |
| 处理速度 | 较慢（多次迭代） | 快速（单次映射） |

---

## Project Structure

```
GradientAV/
├── README.md
├── LICENSE
├── requirements.txt
├── gradient_av.py          # Main program
├── examples/
│   ├── input/              # Example inputs
│   └── output/             # Example outputs
└── docs/
    └── method.md           # Detailed methodology
```

---

## Star History

<a href="https://star-history.com/#Changhuaishui/GradientAV&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Changhuaishui/GradientAV&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=Changhuaishui/GradientAV&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=Changhuaishui/GradientAV&type=Date" />
 </picture>
</a>

---

## Citation

If you find this work useful, please consider citing:

```bibtex
@software{gradientav2024,
  author = {Changhuaishui},
  title = {GradientAV: Gradient-Preserving Retinal Artery-Vein Classification},
  year = {2024},
  url = {https://github.com/Changhuaishui/GradientAV}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**If this project helps you, please give it a star!**

</div>

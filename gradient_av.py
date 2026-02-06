#!/usr/bin/env python
"""
GradientAV - Gradient-Preserving Retinal Artery-Vein Classification

A post-processing tool for retinal vessel segmentation that converts neural
network predictions into standardized artery-vein classification maps.

Core Innovation: Preserve original gradient information instead of binarization,
achieving naturally smooth vessel edges without shape distortion.

Author: Changhuaishui
License: MIT
"""
import argparse
import cv2
import numpy as np
from pathlib import Path


def gradient_av_convert(img_bgr, min_intensity=5):
    """
    Gradient-Preserving Artery-Vein Classification

    Instead of binarizing then smoothing, we preserve the original gradient
    information from neural network outputs.

    Args:
        img_bgr: Input BGR image (neural network prediction)
        min_intensity: Minimum intensity threshold for vessel detection

    Returns:
        result: Output image (red=artery, blue=vein, black=background)
        stats: Dictionary containing artery/vein percentage statistics
    """
    # Extract HSV channels
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.float32)  # Hue [0, 180]
    v = hsv[:, :, 2].astype(np.float32)  # Value [0, 255]

    # Normalize brightness to [0, 1]
    v_norm = v / 255.0

    # Soft vessel mask (preserves gradient, no hard binarization)
    vessel_mask = v_norm * (v > min_intensity).astype(np.float32)

    # Hue-based classification
    # Artery: red hues (h < 30 or h > 140)
    # Vein: blue/cyan hues (30 <= h <= 140)
    artery_weight = np.where((h < 30) | (h > 140), 1.0, 0.0)
    vein_weight = np.where((h >= 30) & (h <= 140), 1.0, 0.0)

    # Smooth transition at hue boundaries
    transition_low = np.clip((30 - h) / 10, 0, 1)   # Gradient near h=30
    transition_high = np.clip((h - 140) / 10, 0, 1)  # Gradient near h=140

    artery_weight = np.maximum(artery_weight, transition_low)
    artery_weight = np.maximum(artery_weight, transition_high)
    vein_weight = 1 - artery_weight

    # Apply vessel mask with gradient modulation
    artery_intensity = vessel_mask * artery_weight
    vein_intensity = vessel_mask * vein_weight

    # Create output image using original brightness gradient
    result = np.zeros_like(img_bgr, dtype=np.float32)
    result[:, :, 2] = artery_intensity * 255  # Red channel (artery)
    result[:, :, 0] = vein_intensity * 255    # Blue channel (vein)

    result = np.clip(result, 0, 255).astype(np.uint8)

    # Calculate statistics
    total = img_bgr.shape[0] * img_bgr.shape[1]
    stats = {
        'artery': np.sum(artery_intensity > 0.5) / total * 100,
        'vein': np.sum(vein_intensity > 0.5) / total * 100,
    }

    return result, stats


def create_comparison(original, converted):
    """Create side-by-side comparison image"""
    h, w = original.shape[:2]
    gap = 20
    canvas = np.zeros((h + 40, w * 2 + gap, 3), dtype=np.uint8)
    canvas[40:40+h, :w] = original
    canvas[40:40+h, w+gap:] = converted

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, 'Input', (w//2-30, 30), font, 0.8, (255, 255, 255), 2)
    cv2.putText(canvas, 'GradientAV', (w+gap+w//2-60, 30), font, 0.8, (255, 255, 255), 2)

    return canvas


def imread_unicode(path):
    """Read image with unicode path support"""
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite_unicode(path, img):
    """Write image with unicode path support"""
    ext = Path(path).suffix
    cv2.imencode(ext, img)[1].tofile(str(path))


def process_file(input_path, output_path, compare=False, **kwargs):
    """Process a single image file"""
    img = imread_unicode(input_path)
    if img is None:
        print(f"[ERROR] Cannot read: {input_path}")
        return False

    result, stats = gradient_av_convert(img, **kwargs)
    output_img = create_comparison(img, result) if compare else result
    imwrite_unicode(output_path, output_img)

    print(f"[OK] {input_path.name}")
    print(f"     Artery: {stats['artery']:.2f}% | Vein: {stats['vein']:.2f}%")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='GradientAV - Gradient-Preserving Retinal Artery-Vein Classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single file:    python gradient_av.py -i input.png -o output.png
  Batch process:  python gradient_av.py -i ./input_dir -o ./output_dir
  With compare:   python gradient_av.py -i input.png -o output.png -c
        """
    )
    parser.add_argument('--input', '-i', required=True,
                        help='Input file or directory')
    parser.add_argument('--output', '-o', required=True,
                        help='Output file or directory')
    parser.add_argument('--compare', '-c', action='store_true',
                        help='Generate side-by-side comparison')
    parser.add_argument('--min-intensity', '-m', type=int, default=5,
                        help='Minimum intensity threshold (default: 5)')

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    kwargs = {'min_intensity': args.min_intensity}

    if input_path.is_file():
        process_file(input_path, output_path, args.compare, **kwargs)
    elif input_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']
        files = []
        for ext in extensions:
            files.extend(input_path.glob(ext))

        if not files:
            print(f"[WARNING] No image files found in {input_path}")
            return

        print(f"Processing {len(files)} files...")
        for img_file in sorted(files):
            out_file = output_path / img_file.name
            process_file(img_file, out_file, args.compare, **kwargs)
        print(f"\nDone! Output saved to {output_path}")
    else:
        print(f"[ERROR] Input path does not exist: {input_path}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
""" TFLite Export Script

Export PyTorch timm models to TensorFlow Lite format using Google's AI Edge Torch.

This script converts timm models to TFLite format for deployment on mobile
and edge devices (Android, iOS, IoT). It uses the ai-edge-torch library which
provides a direct PyTorch to TFLite conversion path.

Requirements:
    pip install ai-edge-torch

Note: The model must be compatible with torch.export() (PyTorch 2.1+)

Copyright 2025 timm contributors
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import torch

import timm
from timm.utils.model import reparameterize_model


def tflite_export(
    model: torch.nn.Module,
    output_file: str,
    input_size: Tuple[int, int, int],
    batch_size: int = 1,
    quantize: bool = False,
) -> None:
    """Export model to TFLite format using AI Edge Torch.

    Args:
        model: PyTorch model to export
        output_file: Output TFLite file path
        input_size: Input size tuple (C, H, W)
        batch_size: Batch size for export
        quantize: Enable dynamic range quantization
    """
    try:
        import ai_edge_torch
    except ImportError:
        raise ImportError(
            "ai-edge-torch is required for TFLite export. "
            "Install with: pip install ai-edge-torch"
        )

    model.eval()

    # Create sample input for tracing
    sample_input = (torch.randn((batch_size,) + input_size),)

    print(f"Exporting model to TFLite: {output_file}")
    print(f"  Input shape: {sample_input[0].shape}")
    print(f"  Quantization: {quantize}")

    # Convert to TFLite
    try:
        if quantize:
            # Dynamic range quantization for smaller model size
            print("Applying dynamic range quantization...")
            edge_model = ai_edge_torch.convert(
                model.eval(),
                sample_input,
                _quantize=True,
            )
        else:
            edge_model = ai_edge_torch.convert(model.eval(), sample_input)

        # Export to file
        edge_model.export(output_file)
        print(f"✓ Successfully exported to {output_file}")

        # Print model info
        import os
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"  Model size: {file_size_mb:.2f} MB")

    except Exception as e:
        print(f"✗ Export failed: {e}")
        print("\nTroubleshooting:")
        print("  - Ensure model is compatible with torch.export()")
        print("  - Try setting exportable=True when creating the model")
        print("  - Some operators may not be supported by TFLite")
        raise


def main():
    parser = argparse.ArgumentParser(
        description='Export timm models to TensorFlow Lite format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('output', metavar='OUTPUT_FILE',
                        help='Output TFLite file path (.tflite)')
    parser.add_argument('--model', '-m', default='mobilenetv3_large_100',
                        help='Model architecture name')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights')
    parser.add_argument('--checkpoint', default='', type=str,
                        help='Path to model checkpoint')
    parser.add_argument('--num-classes', type=int, default=None,
                        help='Number of classes (overrides pretrained)')
    parser.add_argument('--img-size', type=int, default=None,
                        help='Input image size (square)')
    parser.add_argument('--input-size', nargs=3, type=int, default=None,
                        metavar=('C', 'H', 'W'),
                        help='Input size as C H W (e.g., 3 224 224)')
    parser.add_argument('--batch-size', '-b', type=int, default=1,
                        help='Batch size for export')
    parser.add_argument('--quantize', '-q', action='store_true', default=False,
                        help='Enable dynamic range quantization (smaller model)')
    parser.add_argument('--reparam', action='store_true', default=False,
                        help='Reparameterize model before export')

    args = parser.parse_args()

    # Ensure output has .tflite extension
    output_path = Path(args.output)
    if output_path.suffix != '.tflite':
        output_path = output_path.with_suffix('.tflite')
        print(f"Note: Output file renamed to {output_path}")

    # Determine if using pretrained weights
    pretrained = args.pretrained and not args.checkpoint

    print(f"Creating model: {args.model}")
    model = timm.create_model(
        args.model,
        pretrained=pretrained,
        num_classes=args.num_classes,
        checkpoint_path=args.checkpoint if args.checkpoint else None,
        exportable=True,
    )

    if args.reparam:
        print("Reparameterizing model...")
        model = reparameterize_model(model)

    # Determine input size
    if args.input_size:
        input_size = tuple(args.input_size)
    elif args.img_size:
        input_size = (3, args.img_size, args.img_size)
    else:
        input_size = model.default_cfg.get('input_size', (3, 224, 224))

    # Export model
    tflite_export(
        model=model,
        output_file=str(output_path),
        input_size=input_size,
        batch_size=args.batch_size,
        quantize=args.quantize,
    )


if __name__ == '__main__':
    main()

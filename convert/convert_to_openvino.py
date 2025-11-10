#!/usr/bin/env python3
""" OpenVINO Export Script

Export PyTorch timm models to Intel OpenVINO IR (Intermediate Representation) format
for optimized inference on Intel hardware (CPU, GPU, VPU, GNA).

This script converts timm models to OpenVINO IR format which can be used with
OpenVINO Runtime for high-performance inference across various Intel platforms.

Requirements:
    pip install openvino openvino-dev

Copyright 2025 timm contributors
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import torch

import timm
from timm.utils.model import reparameterize_model


def openvino_export(
    model: torch.nn.Module,
    output_file: str,
    input_size: Tuple[int, int, int],
    batch_size: int = 1,
    compress_to_fp16: bool = False,
    dynamic_shapes: bool = False,
) -> None:
    """Export model to OpenVINO IR format.

    Args:
        model: PyTorch model to export
        output_file: Output file path (without extension, .xml and .bin will be added)
        input_size: Input size tuple (C, H, W)
        batch_size: Batch size for export
        compress_to_fp16: Compress weights to FP16 for smaller model size
        dynamic_shapes: Enable dynamic input shapes
    """
    try:
        import openvino as ov
    except ImportError:
        raise ImportError(
            "OpenVINO is required for OpenVINO export. "
            "Install with: pip install openvino openvino-dev"
        )

    model.eval()

    # Create sample input for tracing
    example_input = torch.randn((batch_size,) + input_size)

    print(f"Exporting model to OpenVINO IR: {output_file}")
    print(f"  Input shape: {example_input.shape}")
    print(f"  FP16 compression: {compress_to_fp16}")
    print(f"  Dynamic shapes: {dynamic_shapes}")

    # Convert to OpenVINO IR
    try:
        # Direct conversion from PyTorch
        print("Converting PyTorch model to OpenVINO IR...")
        ov_model = ov.convert_model(
            model,
            example_input=example_input,
            input=[example_input.shape] if not dynamic_shapes else None,
        )

        # Optionally compress to FP16
        if compress_to_fp16:
            print("Compressing model to FP16...")
            from openvino.runtime import serialize
            from openvino.tools.ovc import convert_model

            # FP16 compression happens during serialization
            ov.save_model(ov_model, output_file, compress_to_fp16=True)
        else:
            ov.save_model(ov_model, output_file)

        print(f"✓ Successfully exported to {output_file}")

        # Print model info
        import os
        xml_file = output_file if output_file.endswith('.xml') else f"{output_file}.xml"
        bin_file = output_file.replace('.xml', '.bin') if output_file.endswith('.xml') else f"{output_file}.bin"

        xml_size_mb = os.path.getsize(xml_file) / (1024 * 1024) if os.path.exists(xml_file) else 0
        bin_size_mb = os.path.getsize(bin_file) / (1024 * 1024) if os.path.exists(bin_file) else 0
        total_size_mb = xml_size_mb + bin_size_mb

        print(f"  Model size: {total_size_mb:.2f} MB")
        print(f"    XML: {xml_size_mb:.2f} MB")
        print(f"    BIN: {bin_size_mb:.2f} MB")

        # Verify model can be loaded
        print("Verifying exported model...")
        core = ov.Core()
        compiled_model = core.compile_model(ov_model, "CPU")
        print("✓ Model verification successful")

    except Exception as e:
        print(f"✗ Export failed: {e}")
        print("\nTroubleshooting:")
        print("  - Ensure model is compatible with torch.export() or TorchScript")
        print("  - Try setting exportable=True when creating the model")
        print("  - Some operations may require additional conversion parameters")
        raise


def benchmark_model(model_path: str, num_iterations: int = 100) -> None:
    """Benchmark the exported OpenVINO model.

    Args:
        model_path: Path to the OpenVINO IR model
        num_iterations: Number of inference iterations
    """
    try:
        import openvino as ov
        import numpy as np
        import time
    except ImportError:
        print("Cannot benchmark: OpenVINO not installed")
        return

    print(f"\nBenchmarking model: {model_path}")

    # Load model
    core = ov.Core()
    model = core.read_model(model_path)
    compiled_model = core.compile_model(model, "CPU")

    # Get input shape
    input_layer = compiled_model.input(0)
    input_shape = input_layer.shape

    # Create random input
    random_input = np.random.randn(*input_shape).astype(np.float32)

    # Warmup
    for _ in range(10):
        compiled_model([random_input])

    # Benchmark
    start_time = time.time()
    for _ in range(num_iterations):
        compiled_model([random_input])
    end_time = time.time()

    avg_time_ms = (end_time - start_time) / num_iterations * 1000
    fps = 1000 / avg_time_ms

    print(f"  Average inference time: {avg_time_ms:.2f} ms")
    print(f"  Throughput: {fps:.2f} FPS")


def main():
    parser = argparse.ArgumentParser(
        description='Export timm models to OpenVINO IR format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('output', metavar='OUTPUT_FILE',
                        help='Output file path (without .xml/.bin extension)')
    parser.add_argument('--model', '-m', default='resnet50',
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
    parser.add_argument('--fp16', action='store_true', default=False,
                        help='Compress weights to FP16 (smaller model, slight accuracy loss)')
    parser.add_argument('--dynamic-shapes', action='store_true', default=False,
                        help='Enable dynamic input shapes')
    parser.add_argument('--reparam', action='store_true', default=False,
                        help='Reparameterize model before export')
    parser.add_argument('--benchmark', action='store_true', default=False,
                        help='Benchmark the exported model')
    parser.add_argument('--benchmark-iterations', type=int, default=100,
                        help='Number of iterations for benchmarking')

    args = parser.parse_args()

    # Remove .xml extension if provided
    output_path = Path(args.output)
    if output_path.suffix == '.xml':
        output_path = output_path.with_suffix('')
    output_file = str(output_path)

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
    openvino_export(
        model=model,
        output_file=output_file,
        input_size=input_size,
        batch_size=args.batch_size,
        compress_to_fp16=args.fp16,
        dynamic_shapes=args.dynamic_shapes,
    )

    # Benchmark if requested
    if args.benchmark:
        xml_file = f"{output_file}.xml"
        benchmark_model(xml_file, args.benchmark_iterations)


if __name__ == '__main__':
    main()

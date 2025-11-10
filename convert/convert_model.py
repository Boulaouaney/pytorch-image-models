#!/usr/bin/env python3
""" Unified Model Conversion Script

Convert PyTorch timm models to various deployment formats with a single command.

Supported formats:
  - ONNX: Cross-platform inference
  - TFLite: TensorFlow Lite for mobile (Android, iOS)
  - CoreML: Apple devices (iOS, macOS, iPadOS, watchOS, tvOS)
  - OpenVINO: Intel hardware optimization
  - TensorRT: NVIDIA GPU acceleration
  - ExecuTorch: PyTorch on-device inference

Usage examples:
  # Export to ONNX
  python convert_model.py resnet50 --format onnx --output model.onnx

  # Export to TFLite with quantization
  python convert_model.py mobilenetv3_large_100 --format tflite --quantize

  # Export to CoreML for iOS
  python convert_model.py efficientnet_b0 --format coreml --output model.mlpackage

Copyright 2025 timm contributors
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch

import timm
from timm.utils.model import reparameterize_model


def get_default_output_name(model_name: str, format: str) -> str:
    """Generate default output filename based on model and format."""
    extensions = {
        'onnx': '.onnx',
        'tflite': '.tflite',
        'coreml': '.mlpackage',
        'openvino': '',  # No extension (creates .xml and .bin)
        'tensorrt': '.engine',
        'executorch': '.pte',
    }
    base_name = model_name.replace('/', '_').replace('\\', '_')
    ext = extensions.get(format, '')
    return f"{base_name}{ext}"


def convert_to_onnx(model, output_file, input_size, batch_size, **kwargs):
    """Convert model to ONNX format."""
    from convert_to_onnx import onnx_export

    onnx_export(
        model=model,
        output_file=output_file,
        input_size=input_size,
        batch_size=batch_size,
        opset=kwargs.get('opset'),
        dynamic_size=kwargs.get('dynamic_shapes', False),
        use_dynamo=kwargs.get('use_dynamo', True),
        verbose=kwargs.get('verbose', False),
    )


def convert_to_tflite(model, output_file, input_size, batch_size, **kwargs):
    """Convert model to TFLite format."""
    from convert_to_tflite import tflite_export

    tflite_export(
        model=model,
        output_file=output_file,
        input_size=input_size,
        batch_size=batch_size,
        quantize=kwargs.get('quantize', False),
    )


def convert_to_coreml(model, output_file, input_size, batch_size, **kwargs):
    """Convert model to CoreML format."""
    from convert_to_coreml import coreml_export

    coreml_export(
        model=model,
        output_file=output_file,
        input_size=input_size,
        batch_size=batch_size,
        compute_precision=kwargs.get('compute_precision', 'float32'),
        minimum_deployment_target=kwargs.get('min_ios_version', 'iOS15'),
    )


def convert_to_openvino(model, output_file, input_size, batch_size, **kwargs):
    """Convert model to OpenVINO IR format."""
    from convert_to_openvino import openvino_export

    openvino_export(
        model=model,
        output_file=output_file,
        input_size=input_size,
        batch_size=batch_size,
        compress_to_fp16=kwargs.get('fp16', False),
        dynamic_shapes=kwargs.get('dynamic_shapes', False),
    )


def convert_to_tensorrt(model, output_file, input_size, batch_size, **kwargs):
    """Convert model to TensorRT format."""
    from convert_to_tensorrt import tensorrt_export_via_onnx

    tensorrt_export_via_onnx(
        model=model,
        output_file=output_file,
        input_size=input_size,
        batch_size=batch_size,
        fp16_mode=kwargs.get('fp16', False),
        int8_mode=kwargs.get('int8', False),
        dynamic_shapes=kwargs.get('dynamic_shapes', False),
    )


def convert_to_executorch(model, output_file, input_size, batch_size, **kwargs):
    """Convert model to ExecuTorch format."""
    from convert_to_executorch import executorch_export

    executorch_export(
        model=model,
        output_file=output_file,
        input_size=input_size,
        batch_size=batch_size,
        use_xnnpack=not kwargs.get('no_xnnpack', False),
        quantize=kwargs.get('quantize', False),
    )


def main():
    parser = argparse.ArgumentParser(
        description='Convert timm models to various deployment formats',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert ResNet50 to ONNX
  python convert_model.py resnet50 --format onnx

  # Convert MobileNetV3 to TFLite with quantization
  python convert_model.py mobilenetv3_large_100 --format tflite --quantize

  # Convert EfficientNet to CoreML for iOS
  python convert_model.py efficientnet_b0 --format coreml

  # Convert to multiple formats at once
  python convert_model.py resnet50 --format onnx tflite coreml
        """
    )

    # Required arguments
    parser.add_argument('model', help='Model architecture name from timm')
    parser.add_argument('--format', '-f', nargs='+', required=True,
                        choices=['onnx', 'tflite', 'coreml', 'openvino', 'tensorrt', 'executorch'],
                        help='Target format(s) for conversion')

    # Output configuration
    parser.add_argument('--output', '-o', default=None,
                        help='Output file path (auto-generated if not specified)')
    parser.add_argument('--output-dir', default='.',
                        help='Output directory for converted models')

    # Model configuration
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights')
    parser.add_argument('--checkpoint', default='', type=str,
                        help='Path to model checkpoint')
    parser.add_argument('--num-classes', type=int, default=None,
                        help='Number of classes (overrides pretrained)')
    parser.add_argument('--reparam', action='store_true', default=False,
                        help='Reparameterize model before export')

    # Input configuration
    parser.add_argument('--img-size', type=int, default=None,
                        help='Input image size (square)')
    parser.add_argument('--input-size', nargs=3, type=int, default=None,
                        metavar=('C', 'H', 'W'),
                        help='Input size as C H W (e.g., 3 224 224)')
    parser.add_argument('--batch-size', '-b', type=int, default=1,
                        help='Batch size for export')

    # Optimization options
    parser.add_argument('--quantize', '-q', action='store_true', default=False,
                        help='Enable quantization (TFLite, ExecuTorch)')
    parser.add_argument('--fp16', action='store_true', default=False,
                        help='Enable FP16 precision (OpenVINO, TensorRT)')
    parser.add_argument('--int8', action='store_true', default=False,
                        help='Enable INT8 precision (TensorRT only)')
    parser.add_argument('--dynamic-shapes', action='store_true', default=False,
                        help='Enable dynamic input shapes (ONNX, OpenVINO, TensorRT)')

    # Format-specific options
    parser.add_argument('--compute-precision', default='float32',
                        choices=['float32', 'float16', 'mixed'],
                        help='CoreML compute precision')
    parser.add_argument('--min-ios-version', default='iOS15',
                        help='Minimum iOS version for CoreML (e.g., iOS15)')
    parser.add_argument('--opset', type=int, default=None,
                        help='ONNX opset version')
    parser.add_argument('--use-dynamo', action='store_true', default=True,
                        help='Use Dynamo exporter for ONNX (PyTorch 2.5+)')
    parser.add_argument('--no-xnnpack', action='store_true', default=False,
                        help='Disable XNNPACK backend for ExecuTorch')

    # General options
    parser.add_argument('--verbose', '-v', action='store_true', default=False,
                        help='Verbose output')

    args = parser.parse_args()

    # Validate output argument
    if args.output and len(args.format) > 1:
        print("Error: --output cannot be used when converting to multiple formats")
        print("Use --output-dir to specify output directory instead")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine if using pretrained weights
    pretrained = args.pretrained and not args.checkpoint

    # Create model
    print(f"\nCreating model: {args.model}")
    print(f"  Pretrained: {pretrained}")

    model = timm.create_model(
        args.model,
        pretrained=pretrained,
        num_classes=args.num_classes,
        checkpoint_path=args.checkpoint if args.checkpoint else None,
        exportable=True,
    )

    if args.reparam:
        print("  Reparameterizing model...")
        model = reparameterize_model(model)

    # Determine input size
    if args.input_size:
        input_size = tuple(args.input_size)
    elif args.img_size:
        input_size = (3, args.img_size, args.img_size)
    else:
        input_size = model.default_cfg.get('input_size', (3, 224, 224))

    print(f"  Input size: {input_size}")
    print(f"  Batch size: {args.batch_size}")

    # Conversion function mapping
    converters = {
        'onnx': convert_to_onnx,
        'tflite': convert_to_tflite,
        'coreml': convert_to_coreml,
        'openvino': convert_to_openvino,
        'tensorrt': convert_to_tensorrt,
        'executorch': convert_to_executorch,
    }

    # Convert to each requested format
    # Prepare kwargs, excluding parameters we pass explicitly
    kwargs = vars(args).copy()
    for key in ['model', 'output', 'output_dir', 'format', 'pretrained',
                'checkpoint', 'num_classes', 'reparam', 'img_size',
                'input_size', 'batch_size']:
        kwargs.pop(key, None)

    success_count = 0
    fail_count = 0

    for fmt in args.format:
        print(f"\n{'='*60}")
        print(f"Converting to {fmt.upper()}...")
        print('='*60)

        # Determine output file
        if args.output:
            output_file = args.output
        else:
            output_file = str(output_dir / get_default_output_name(args.model, fmt))

        try:
            converter = converters[fmt]
            converter(
                model=model,
                output_file=output_file,
                input_size=input_size,
                batch_size=args.batch_size,
                **kwargs
            )
            success_count += 1
        except Exception as e:
            print(f"\n✗ Failed to convert to {fmt}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            fail_count += 1

    # Print summary
    print(f"\n{'='*60}")
    print("Conversion Summary")
    print('='*60)
    print(f"Successful: {success_count}/{len(args.format)}")
    print(f"Failed: {fail_count}/{len(args.format)}")

    if fail_count > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()

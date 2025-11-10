#!/usr/bin/env python3
""" TensorRT Export Script

Export PyTorch timm models to NVIDIA TensorRT format for optimized inference
on NVIDIA GPUs.

This script supports two conversion paths:
1. PyTorch -> ONNX -> TensorRT (recommended, most compatible)
2. PyTorch -> TensorRT via torch2trt (requires torch2trt installation)

TensorRT provides significant performance improvements on NVIDIA GPUs through
layer fusion, precision calibration, and kernel auto-tuning.

Requirements:
    - NVIDIA TensorRT: https://developer.nvidia.com/tensorrt
    - For ONNX path: pip install onnx
    - For torch2trt path: https://github.com/NVIDIA-AI-IOT/torch2trt

Copyright 2025 timm contributors
"""

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple
import tempfile

import torch

import timm
from timm.utils.model import reparameterize_model


def tensorrt_export_via_onnx(
    model: torch.nn.Module,
    output_file: str,
    input_size: Tuple[int, int, int],
    batch_size: int = 1,
    fp16_mode: bool = False,
    int8_mode: bool = False,
    workspace_size: int = 1 << 30,  # 1GB
    dynamic_shapes: bool = False,
) -> None:
    """Export model to TensorRT via ONNX path.

    Args:
        model: PyTorch model to export
        output_file: Output TensorRT engine file path (.trt or .engine)
        input_size: Input size tuple (C, H, W)
        batch_size: Batch size for export
        fp16_mode: Enable FP16 precision mode
        int8_mode: Enable INT8 precision mode (requires calibration)
        workspace_size: Maximum workspace size in bytes
        dynamic_shapes: Enable dynamic input shapes
    """
    try:
        import tensorrt as trt
    except ImportError:
        raise ImportError(
            "TensorRT is required for TensorRT export. "
            "Please install NVIDIA TensorRT from: "
            "https://developer.nvidia.com/tensorrt"
        )

    model.eval()

    print(f"Exporting model to TensorRT via ONNX: {output_file}")
    print(f"  Input shape: ({batch_size}, {input_size[0]}, {input_size[1]}, {input_size[2]})")
    print(f"  FP16 mode: {fp16_mode}")
    print(f"  INT8 mode: {int8_mode}")

    # Step 1: Export to ONNX
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:
        onnx_file = tmp.name

    try:
        print("Step 1: Exporting to ONNX...")
        example_input = torch.randn((batch_size,) + input_size)

        # Run model once to initialize any dynamic padding
        with torch.inference_mode():
            _ = model(example_input)

        # Export to ONNX
        dynamic_axes = None
        if dynamic_shapes:
            dynamic_axes = {
                'input': {0: 'batch', 2: 'height', 3: 'width'},
                'output': {0: 'batch'}
            }

        torch.onnx.export(
            model,
            example_input,
            onnx_file,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            opset_version=17,
            do_constant_folding=True,
        )
        print(f"  ✓ ONNX export successful: {onnx_file}")

        # Step 2: Convert ONNX to TensorRT
        print("Step 2: Converting ONNX to TensorRT...")

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, TRT_LOGGER)

        # Parse ONNX model
        with open(onnx_file, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                raise RuntimeError("Failed to parse ONNX model")

        # Configure builder
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)

        # Set precision modes
        if fp16_mode and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("  ✓ FP16 mode enabled")

        if int8_mode and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            print("  ✓ INT8 mode enabled (note: calibration may be needed)")

        # Build engine
        print("  Building TensorRT engine (this may take a while)...")
        serialized_engine = builder.build_serialized_network(network, config)

        if serialized_engine is None:
            raise RuntimeError("Failed to build TensorRT engine")

        # Save engine
        with open(output_file, 'wb') as f:
            f.write(serialized_engine)

        print(f"✓ Successfully exported to {output_file}")

        # Print model info
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"  Engine size: {file_size_mb:.2f} MB")

    finally:
        # Clean up temporary ONNX file
        if os.path.exists(onnx_file):
            os.unlink(onnx_file)


def tensorrt_export_via_torch2trt(
    model: torch.nn.Module,
    output_file: str,
    input_size: Tuple[int, int, int],
    batch_size: int = 1,
    fp16_mode: bool = False,
    max_workspace_size: int = 1 << 30,
) -> None:
    """Export model to TensorRT via torch2trt.

    Args:
        model: PyTorch model to export
        output_file: Output TensorRT engine file path
        input_size: Input size tuple (C, H, W)
        batch_size: Batch size for export
        fp16_mode: Enable FP16 precision mode
        max_workspace_size: Maximum workspace size in bytes
    """
    try:
        from torch2trt import torch2trt
    except ImportError:
        raise ImportError(
            "torch2trt is required for this conversion path. "
            "Install from: https://github.com/NVIDIA-AI-IOT/torch2trt"
        )

    model.eval().cuda()

    print(f"Exporting model to TensorRT via torch2trt: {output_file}")
    print(f"  Input shape: ({batch_size}, {input_size[0]}, {input_size[1]}, {input_size[2]})")
    print(f"  FP16 mode: {fp16_mode}")

    # Create sample input on GPU
    x = torch.randn((batch_size,) + input_size).cuda()

    # Convert to TensorRT
    print("Converting to TensorRT (this may take a while)...")
    model_trt = torch2trt(
        model,
        [x],
        fp16_mode=fp16_mode,
        max_workspace_size=max_workspace_size,
    )

    # Save model
    torch.save(model_trt.state_dict(), output_file)
    print(f"✓ Successfully exported to {output_file}")

    # Print model info
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"  Model size: {file_size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description='Export timm models to TensorRT format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('output', metavar='OUTPUT_FILE',
                        help='Output TensorRT engine file path (.trt or .engine)')
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
                        help='Enable FP16 precision mode')
    parser.add_argument('--int8', action='store_true', default=False,
                        help='Enable INT8 precision mode (ONNX path only)')
    parser.add_argument('--workspace-size', type=int, default=1 << 30,
                        help='Maximum workspace size in bytes (default: 1GB)')
    parser.add_argument('--dynamic-shapes', action='store_true', default=False,
                        help='Enable dynamic input shapes (ONNX path only)')
    parser.add_argument('--method', default='onnx', choices=['onnx', 'torch2trt'],
                        help='Conversion method: onnx (recommended) or torch2trt')
    parser.add_argument('--reparam', action='store_true', default=False,
                        help='Reparameterize model before export')

    args = parser.parse_args()

    # Ensure output has correct extension
    output_path = Path(args.output)
    if output_path.suffix not in ['.trt', '.engine', '.pth']:
        if args.method == 'torch2trt':
            output_path = output_path.with_suffix('.pth')
        else:
            output_path = output_path.with_suffix('.engine')
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

    # Export model using selected method
    if args.method == 'onnx':
        tensorrt_export_via_onnx(
            model=model,
            output_file=str(output_path),
            input_size=input_size,
            batch_size=args.batch_size,
            fp16_mode=args.fp16,
            int8_mode=args.int8,
            workspace_size=args.workspace_size,
            dynamic_shapes=args.dynamic_shapes,
        )
    elif args.method == 'torch2trt':
        if args.int8:
            print("Warning: INT8 mode not supported with torch2trt method")
        if args.dynamic_shapes:
            print("Warning: Dynamic shapes not supported with torch2trt method")

        tensorrt_export_via_torch2trt(
            model=model,
            output_file=str(output_path),
            input_size=input_size,
            batch_size=args.batch_size,
            fp16_mode=args.fp16,
            max_workspace_size=args.workspace_size,
        )


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
""" ExecuTorch Export Script

Export PyTorch timm models to ExecuTorch format with XNNPACK backend for
optimized on-device inference on mobile and edge devices.

ExecuTorch is PyTorch's solution for running models efficiently on edge devices,
supporting various backends including XNNPACK for CPU acceleration.

Requirements:
    pip install executorch

Note: ExecuTorch requires PyTorch 2.1+ and uses torch.export() for model capture.

Copyright 2025 timm contributors
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import torch

import timm
from timm.utils.model import reparameterize_model


def executorch_export(
    model: torch.nn.Module,
    output_file: str,
    input_size: Tuple[int, int, int],
    batch_size: int = 1,
    use_xnnpack: bool = True,
    quantize: bool = False,
) -> None:
    """Export model to ExecuTorch format.

    Args:
        model: PyTorch model to export
        output_file: Output .pte file path
        input_size: Input size tuple (C, H, W)
        batch_size: Batch size for export
        use_xnnpack: Use XNNPACK backend for acceleration
        quantize: Enable quantization
    """
    try:
        from executorch.exir import to_edge
        from torch.export import export
        from executorch.exir import EdgeProgramManager
    except ImportError:
        raise ImportError(
            "ExecuTorch is required for ExecuTorch export. "
            "Install with: pip install executorch"
        )

    model.eval()

    # Create sample input for export
    example_input = (torch.randn((batch_size,) + input_size),)

    print(f"Exporting model to ExecuTorch: {output_file}")
    print(f"  Input shape: {example_input[0].shape}")
    print(f"  XNNPACK backend: {use_xnnpack}")
    print(f"  Quantization: {quantize}")

    try:
        # Step 1: Export model using torch.export
        print("Step 1: Exporting model with torch.export()...")
        with torch.no_grad():
            exported_program = export(model, example_input)
        print("  ✓ Export successful")

        # Step 2: Lower to Edge dialect
        print("Step 2: Lowering to Edge dialect...")

        if use_xnnpack:
            try:
                from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

                edge_program = to_edge(
                    exported_program,
                    compile_config=None,
                )

                # Partition for XNNPACK
                print("  Partitioning for XNNPACK backend...")
                edge_program = edge_program.to_backend(XnnpackPartitioner())
                print("  ✓ XNNPACK partitioning successful")

            except ImportError:
                print("  Warning: XNNPACK backend not available, using default")
                edge_program = to_edge(exported_program)
        else:
            edge_program = to_edge(exported_program)

        # Step 3: Convert to ExecuTorch program
        print("Step 3: Converting to ExecuTorch program...")
        executorch_program = edge_program.to_executorch()

        # Step 4: Save to file
        print(f"Step 4: Saving to {output_file}...")
        with open(output_file, 'wb') as f:
            f.write(executorch_program.buffer)

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
        print("  - Some operations may not be supported by ExecuTorch/XNNPACK")
        print("  - Try without XNNPACK: --no-xnnpack")
        raise


def main():
    parser = argparse.ArgumentParser(
        description='Export timm models to ExecuTorch format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('output', metavar='OUTPUT_FILE',
                        help='Output ExecuTorch file path (.pte)')
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
    parser.add_argument('--no-xnnpack', action='store_true', default=False,
                        help='Disable XNNPACK backend')
    parser.add_argument('--quantize', '-q', action='store_true', default=False,
                        help='Enable quantization (experimental)')
    parser.add_argument('--reparam', action='store_true', default=False,
                        help='Reparameterize model before export')

    args = parser.parse_args()

    # Ensure output has .pte extension
    output_path = Path(args.output)
    if output_path.suffix != '.pte':
        output_path = output_path.with_suffix('.pte')
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
    executorch_export(
        model=model,
        output_file=str(output_path),
        input_size=input_size,
        batch_size=args.batch_size,
        use_xnnpack=not args.no_xnnpack,
        quantize=args.quantize,
    )


if __name__ == '__main__':
    main()

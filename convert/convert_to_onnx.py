#!/usr/bin/env python3
""" ONNX Export Script

Export PyTorch timm models to ONNX format with modern Dynamo-based exporter support.

This script provides an enhanced interface for converting timm models to ONNX,
supporting both legacy torch.onnx.export and the new Dynamo-based exporter
(recommended for PyTorch 2.5+).

Copyright 2025 timm contributors
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

# Check dependencies before importing
try:
    from convert_utils import check_and_install_dependencies, get_framework_dependencies, print_framework_info
    packages, specs = get_framework_dependencies('onnx')
    if not check_and_install_dependencies(packages, specs):
        sys.exit(1)
except ImportError:
    print("Warning: convert_utils not found, skipping dependency check")

import torch

import timm
from timm.utils.model import reparameterize_model


def onnx_export(
    model: torch.nn.Module,
    output_file: str,
    input_size: Tuple[int, int, int],
    batch_size: int = 1,
    opset: Optional[int] = None,
    dynamic_size: bool = False,
    use_dynamo: bool = True,
    verbose: bool = False,
) -> None:
    """Export model to ONNX format.

    Args:
        model: PyTorch model to export
        output_file: Output ONNX file path
        input_size: Input size tuple (C, H, W)
        batch_size: Batch size for export
        opset: ONNX opset version
        dynamic_size: Enable dynamic input sizes
        use_dynamo: Use new Dynamo-based exporter (recommended)
        verbose: Verbose output
    """
    import onnx

    model.eval()
    example_input = torch.randn((batch_size,) + input_size)

    # Run model once to initialize any dynamic padding
    with torch.inference_mode():
        _ = model(example_input)

    print(f"Exporting model to {output_file}")
    print(f"  Input shape: {example_input.shape}")
    print(f"  Using Dynamo exporter: {use_dynamo}")

    if use_dynamo:
        # Use new Dynamo-based exporter (PyTorch 2.1+)
        try:
            # PyTorch 2.5+ API
            export_options = torch.onnx.ExportOptions(
                dynamic_shapes=dynamic_size,
                opset_version=opset,
            )
            export_output = torch.onnx.dynamo_export(
                model,
                example_input,
                export_options=export_options,
            )
            export_output.save(output_file)
        except AttributeError:
            # Fallback for PyTorch 2.1-2.4 using torch.export
            print("  Note: Using torch.export API (PyTorch 2.1-2.4)")
            import torch._dynamo as dynamo
            exported_program = torch.export.export(model, (example_input,))
            torch.onnx.export(
                exported_program,
                (example_input,),
                output_file,
                opset_version=opset or 17,
            )
    else:
        # Legacy exporter
        input_names = ["input"]
        output_names = ["output"]
        dynamic_axes = {'input': {0: 'batch'}, 'output': {0: 'batch'}}

        if dynamic_size:
            dynamic_axes['input'][2] = 'height'
            dynamic_axes['input'][3] = 'width'

        torch.onnx.export(
            model,
            example_input,
            output_file,
            training=torch.onnx.TrainingMode.EVAL,
            export_params=True,
            verbose=verbose,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
        )

    # Validate exported model
    print("Validating ONNX model...")
    onnx_model = onnx.load(output_file)
    onnx.checker.check_model(onnx_model, full_check=True)
    print(f"✓ Successfully exported to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Export timm models to ONNX format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('output', metavar='OUTPUT_FILE',
                        help='Output ONNX file path')
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
    parser.add_argument('--opset', type=int, default=None,
                        help='ONNX opset version (default: latest)')
    parser.add_argument('--dynamic-size', action='store_true', default=False,
                        help='Enable dynamic input dimensions')
    parser.add_argument('--use-dynamo', action='store_true', default=True,
                        help='Use Dynamo-based exporter (PyTorch 2.5+)')
    parser.add_argument('--legacy', action='store_true', default=False,
                        help='Use legacy exporter instead of Dynamo')
    parser.add_argument('--reparam', action='store_true', default=False,
                        help='Reparameterize model before export')
    parser.add_argument('--verbose', '-v', action='store_true', default=False,
                        help='Verbose output')

    args = parser.parse_args()

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
    use_dynamo = args.use_dynamo and not args.legacy

    onnx_export(
        model=model,
        output_file=args.output,
        input_size=input_size,
        batch_size=args.batch_size,
        opset=args.opset,
        dynamic_size=args.dynamic_size,
        use_dynamo=use_dynamo,
        verbose=args.verbose,
    )


if __name__ == '__main__':
    main()

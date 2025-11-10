#!/usr/bin/env python3
""" CoreML Export Script

Export PyTorch timm models to Apple CoreML format for deployment on iOS, macOS,
iPadOS, watchOS, and tvOS devices.

This script converts timm models to CoreML format using Apple's coremltools.
The converted models can be used with Vision framework or directly with CoreML.

Requirements:
    pip install coremltools

Copyright 2025 timm contributors
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple, List

import torch

import timm
from timm.utils.model import reparameterize_model


def coreml_export(
    model: torch.nn.Module,
    output_file: str,
    input_size: Tuple[int, int, int],
    batch_size: int = 1,
    classifier_config: Optional[Tuple[List[str], str]] = None,
    compute_precision: str = 'float32',
    minimum_deployment_target: Optional[str] = None,
) -> None:
    """Export model to CoreML format.

    Args:
        model: PyTorch model to export
        output_file: Output CoreML file path (.mlpackage or .mlmodel)
        input_size: Input size tuple (C, H, W)
        batch_size: Batch size for export
        classifier_config: Optional tuple of (class_labels, predicted_feature_name)
        compute_precision: Precision for compute ('float32', 'float16', or 'mixed')
        minimum_deployment_target: Minimum iOS version (e.g., 'iOS15')
    """
    try:
        import coremltools as ct
    except ImportError:
        raise ImportError(
            "coremltools is required for CoreML export. "
            "Install with: pip install coremltools"
        )

    model.eval()

    # Create sample input for tracing
    example_input = torch.randn((batch_size,) + input_size)

    print(f"Exporting model to CoreML: {output_file}")
    print(f"  Input shape: {example_input.shape}")
    print(f"  Compute precision: {compute_precision}")

    # Trace the model using TorchScript
    print("Tracing model with torch.jit.trace...")
    try:
        traced_model = torch.jit.trace(model, example_input)
    except Exception as e:
        print(f"✗ Tracing failed: {e}")
        print("\nTrying torch.jit.script instead...")
        traced_model = torch.jit.script(model)

    # Set up input type with optional shape constraints
    input_shape = ct.Shape(shape=example_input.shape)
    inputs = [ct.TensorType(name="input", shape=input_shape)]

    # Set up classifier config if provided
    if classifier_config:
        class_labels, predicted_feature_name = classifier_config
        classifier_config_ct = ct.ClassifierConfig(
            class_labels=class_labels,
            predicted_feature_name=predicted_feature_name
        )
    else:
        classifier_config_ct = None

    # Set compute precision
    compute_precision_map = {
        'float32': ct.precision.FLOAT32,
        'float16': ct.precision.FLOAT16,
        'mixed': None,  # Default mixed precision
    }
    compute_precision_ct = compute_precision_map.get(compute_precision.lower())

    # Convert to CoreML
    print("Converting to CoreML format...")
    try:
        if minimum_deployment_target:
            # Parse minimum deployment target
            target_map = {
                'iOS13': ct.target.iOS13,
                'iOS14': ct.target.iOS14,
                'iOS15': ct.target.iOS15,
                'iOS16': ct.target.iOS16,
                'iOS17': ct.target.iOS17,
                'iOS18': ct.target.iOS18,
            }
            min_deployment = target_map.get(minimum_deployment_target, ct.target.iOS15)
        else:
            min_deployment = ct.target.iOS15

        coreml_model = ct.convert(
            traced_model,
            inputs=inputs,
            classifier_config=classifier_config_ct,
            minimum_deployment_target=min_deployment,
            compute_precision=compute_precision_ct,
        )

        # Save the model
        coreml_model.save(output_file)
        print(f"✓ Successfully exported to {output_file}")

        # Print model info
        import os
        if os.path.isdir(output_file):  # .mlpackage is a directory
            # Calculate total size of directory
            total_size = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, _, filenames in os.walk(output_file)
                for filename in filenames
            )
            file_size_mb = total_size / (1024 * 1024)
        else:
            file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"  Model size: {file_size_mb:.2f} MB")

    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        print("\nTroubleshooting:")
        print("  - Ensure model doesn't use unsupported operations")
        print("  - Try using --compute-precision float32 for compatibility")
        print("  - Some custom layers may require CoreML custom layer implementation")
        raise


def main():
    parser = argparse.ArgumentParser(
        description='Export timm models to Apple CoreML format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('output', metavar='OUTPUT_FILE',
                        help='Output CoreML file path (.mlpackage or .mlmodel)')
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
    parser.add_argument('--compute-precision', default='float32',
                        choices=['float32', 'float16', 'mixed'],
                        help='Compute precision for CoreML')
    parser.add_argument('--minimum-deployment-target', default='iOS15',
                        choices=['iOS13', 'iOS14', 'iOS15', 'iOS16', 'iOS17', 'iOS18'],
                        help='Minimum deployment target iOS version')
    parser.add_argument('--class-labels', type=str, default=None,
                        help='Path to text file with class labels (one per line)')
    parser.add_argument('--reparam', action='store_true', default=False,
                        help='Reparameterize model before export')

    args = parser.parse_args()

    # Ensure output has correct extension
    output_path = Path(args.output)
    if output_path.suffix not in ['.mlpackage', '.mlmodel']:
        output_path = output_path.with_suffix('.mlpackage')
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

    # Load class labels if provided
    classifier_config = None
    if args.class_labels:
        with open(args.class_labels, 'r') as f:
            class_labels = [line.strip() for line in f if line.strip()]
        classifier_config = (class_labels, 'class_label')
        print(f"Loaded {len(class_labels)} class labels")

    # Export model
    coreml_export(
        model=model,
        output_file=str(output_path),
        input_size=input_size,
        batch_size=args.batch_size,
        classifier_config=classifier_config,
        compute_precision=args.compute_precision,
        minimum_deployment_target=args.minimum_deployment_target,
    )


if __name__ == '__main__':
    main()

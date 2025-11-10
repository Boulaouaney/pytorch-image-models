#!/usr/bin/env python3
"""Test conversion scripts with multiple models.

Tests ONNX, TFLite, and CoreML conversions with common models
to ensure proper functionality.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


# Test models known to work well with conversions
TEST_MODELS = [
    'mobilenetv3_large_100',
    'resnet50',
    'efficientnet_b0',
]


def run_conversion(script: str, model: str, output: str, extra_args: list = None) -> tuple[bool, str]:
    """Run a conversion script and return success status and output.

    Args:
        script: Path to conversion script
        model: Model name
        output: Output file path
        extra_args: Additional arguments for the script

    Returns:
        Tuple of (success, output_message)
    """
    cmd = ['uv', 'run', 'python', script, output, '--model', model]
    if extra_args:
        cmd.extend(extra_args)

    try:
        print(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode == 0:
            return True, "Success"
        else:
            return False, f"Exit code {result.returncode}: {result.stderr[:200]}"

    except subprocess.TimeoutExpired:
        return False, "Timeout (>300s)"
    except Exception as e:
        return False, f"Exception: {str(e)}"


def test_onnx_conversion(model: str, output_dir: Path) -> bool:
    """Test ONNX conversion for a model."""
    output_file = output_dir / f"{model}.onnx"
    success, msg = run_conversion(
        'convert/convert_to_onnx.py',
        model,
        str(output_file)
    )

    if success and output_file.exists():
        size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"    ✓ ONNX: {size_mb:.2f} MB")
        return True
    else:
        print(f"    ✗ ONNX: {msg}")
        return False


def test_tflite_conversion(model: str, output_dir: Path) -> bool:
    """Test TFLite conversion for a model."""
    output_file = output_dir / f"{model}.tflite"
    success, msg = run_conversion(
        'convert/convert_to_tflite.py',
        model,
        str(output_file)
    )

    if success and output_file.exists():
        size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"    ✓ TFLite: {size_mb:.2f} MB")
        return True
    else:
        print(f"    ✗ TFLite: {msg}")
        return False


def test_coreml_conversion(model: str, output_dir: Path) -> bool:
    """Test CoreML conversion for a model."""
    output_file = output_dir / f"{model}.mlpackage"
    success, msg = run_conversion(
        'convert/convert_to_coreml.py',
        model,
        str(output_file)
    )

    if success and output_file.exists():
        # Calculate directory size for .mlpackage
        total_size = sum(
            f.stat().st_size
            for f in output_file.rglob('*')
            if f.is_file()
        )
        size_mb = total_size / (1024 * 1024)
        print(f"    ✓ CoreML: {size_mb:.2f} MB")
        return True
    else:
        print(f"    ✗ CoreML: {msg}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Test model conversions across multiple formats'
    )
    parser.add_argument('--models', nargs='+', default=TEST_MODELS,
                        help='Models to test')
    parser.add_argument('--formats', nargs='+', default=['onnx'],
                        choices=['onnx', 'tflite', 'coreml'],
                        help='Formats to test')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (default: temp dir)')
    parser.add_argument('--keep', action='store_true',
                        help='Keep output files after testing')

    args = parser.parse_args()

    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = None
    else:
        temp_dir = tempfile.TemporaryDirectory()
        output_dir = Path(temp_dir.name)

    print(f"\nTesting model conversions")
    print(f"Output directory: {output_dir}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Formats: {', '.join(args.formats)}")
    print("=" * 60)

    # Test each model with each format
    results = {}

    for model in args.models:
        print(f"\n{model}:")
        results[model] = {}

        if 'onnx' in args.formats:
            results[model]['onnx'] = test_onnx_conversion(model, output_dir)

        if 'tflite' in args.formats:
            results[model]['tflite'] = test_tflite_conversion(model, output_dir)

        if 'coreml' in args.formats:
            results[model]['coreml'] = test_coreml_conversion(model, output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    for model in args.models:
        successes = sum(1 for v in results[model].values() if v)
        total = len(results[model])
        status = "✓" if successes == total else "✗"
        print(f"{status} {model}: {successes}/{total} formats successful")

    # Calculate overall statistics
    total_tests = sum(len(r) for r in results.values())
    total_success = sum(sum(1 for v in r.values() if v) for r in results.values())

    print(f"\nOverall: {total_success}/{total_tests} tests passed")

    # Cleanup
    if temp_dir and not args.keep:
        temp_dir.cleanup()

    # Exit with error if any tests failed
    if total_success < total_tests:
        sys.exit(1)


if __name__ == '__main__':
    main()

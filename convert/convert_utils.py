"""Common utilities for conversion scripts.

Handles dynamic dependency checking and installation using uv.
"""

import importlib
import subprocess
import sys
from typing import List, Dict, Optional


def check_and_install_dependencies(
    packages: List[str],
    package_specs: Optional[Dict[str, str]] = None,
    use_uv: bool = True
) -> bool:
    """Check if required packages are installed, install if missing.

    Args:
        packages: List of package names to check
        package_specs: Optional dict mapping package names to version specs
        use_uv: Use uv for installation (faster, recommended)

    Returns:
        True if all dependencies are satisfied, False otherwise
    """
    missing_packages = []
    package_specs = package_specs or {}

    for package in packages:
        try:
            importlib.import_module(package)
        except ImportError:
            missing_packages.append(package)

    if not missing_packages:
        return True

    # Ask user for confirmation
    print("\nMissing required dependencies:")
    for pkg in missing_packages:
        spec = package_specs.get(pkg, pkg)
        print(f"  - {spec}")

    response = input("\nInstall missing dependencies? [Y/n]: ").strip().lower()
    if response and response != 'y':
        print("Aborting: Required dependencies not installed")
        return False

    # Install missing packages
    print("\nInstalling dependencies...")
    install_cmd = "uv pip install" if use_uv else f"{sys.executable} -m pip install"

    for pkg in missing_packages:
        spec = package_specs.get(pkg, pkg)
        try:
            cmd = f"{install_cmd} {spec}"
            print(f"  Running: {cmd}")
            subprocess.check_call(cmd, shell=True)
            print(f"  ✓ Installed {spec}")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Failed to install {spec}: {e}")
            return False

    return True


def get_framework_dependencies(framework: str) -> tuple[List[str], Dict[str, str]]:
    """Get required packages and version specs for a framework.

    Args:
        framework: Framework name (onnx, tflite, coreml, etc.)

    Returns:
        Tuple of (package_names, package_specs)
    """
    dependencies = {
        'onnx': (
            ['onnx'],
            {'onnx': 'onnx>=1.12.0'}
        ),
        'tflite': (
            ['ai_edge_torch'],
            {'ai_edge_torch': 'ai-edge-torch'}
        ),
        'coreml': (
            ['coremltools'],
            {'coremltools': 'coremltools>=8.0'}
        ),
        'openvino': (
            ['openvino'],
            {'openvino': 'openvino>=2023.0'}
        ),
        'tensorrt': (
            ['tensorrt'],
            {'tensorrt': 'tensorrt'}  # Note: Usually requires manual install
        ),
        'executorch': (
            ['executorch'],
            {'executorch': 'executorch'}
        ),
    }

    return dependencies.get(framework, ([], {}))


def check_pytorch_version(min_version: str = "2.0.0") -> bool:
    """Check if PyTorch version meets minimum requirement.

    Args:
        min_version: Minimum required PyTorch version

    Returns:
        True if version is sufficient
    """
    try:
        import torch
        from packaging import version

        current = version.parse(torch.__version__.split('+')[0])
        required = version.parse(min_version)

        if current < required:
            print(f"\nWarning: PyTorch {min_version}+ required, found {torch.__version__}")
            print("Some features may not work correctly")
            return False
        return True
    except ImportError:
        print("\nError: PyTorch is not installed")
        print("Install with: uv pip install torch torchvision")
        return False


def check_gpu_available() -> bool:
    """Check if CUDA GPU is available.

    Returns:
        True if GPU is available
    """
    try:
        import torch
        return torch.cuda.is_available()
    except:
        return False


def print_framework_info(framework: str):
    """Print information about framework requirements and limitations."""
    info = {
        'onnx': {
            'description': 'ONNX - Universal format for model interchange',
            'requirements': ['onnx'],
            'optional': ['onnxruntime', 'onnxruntime-gpu'],
            'notes': [
                'PyTorch 2.5+ supports new Dynamo-based exporter (recommended)',
                'Legacy exporter works with PyTorch 1.6+',
                'Most models are well supported'
            ]
        },
        'tflite': {
            'description': 'TensorFlow Lite - Mobile and edge deployment',
            'requirements': ['ai-edge-torch', 'torch>=2.1.0'],
            'optional': [],
            'notes': [
                'Requires torch.export() compatible models (PyTorch 2.1+)',
                'Google AI Edge Torch provides direct PyTorch → TFLite',
                'Not all PyTorch ops are supported',
                'MobileNet and EfficientNet families work well'
            ]
        },
        'coreml': {
            'description': 'CoreML - Apple ecosystem (iOS, macOS, etc.)',
            'requirements': ['coremltools>=8.0'],
            'optional': [],
            'notes': [
                '~70% PyTorch operator coverage',
                'Tracing recommended over scripting',
                'FP16 support for smaller models',
                'Best for standard CNN architectures'
            ]
        },
        'openvino': {
            'description': 'OpenVINO - Intel hardware optimization',
            'requirements': ['openvino>=2023.0'],
            'optional': ['openvino-dev'],
            'notes': [
                'Direct PyTorch conversion supported (2023.0+)',
                'Excellent performance on Intel CPUs/GPUs',
                'FP16 compression recommended',
                'Can also convert via ONNX'
            ]
        },
        'tensorrt': {
            'description': 'TensorRT - NVIDIA GPU acceleration',
            'requirements': ['tensorrt', 'onnx'],
            'optional': ['torch2trt'],
            'notes': [
                'Requires NVIDIA GPU and TensorRT installation',
                'ONNX → TensorRT path is most stable',
                'FP16 provides 2x speedup with minimal accuracy loss',
                'INT8 requires calibration dataset',
                'Manual TensorRT install often needed'
            ]
        },
        'executorch': {
            'description': 'ExecuTorch - PyTorch on-device inference',
            'requirements': ['executorch', 'torch>=2.1.0'],
            'optional': [],
            'notes': [
                'Requires torch.export() compatible models',
                'XNNPACK backend for CPU acceleration',
                'BF16 support on newer devices',
                'Latest PyTorch (2.9+) recommended'
            ]
        }
    }

    if framework not in info:
        return

    data = info[framework]
    print(f"\n{'='*60}")
    print(f"{data['description']}")
    print('='*60)
    print("\nRequired packages:")
    for req in data['requirements']:
        print(f"  - {req}")

    if data['optional']:
        print("\nOptional packages:")
        for opt in data['optional']:
            print(f"  - {opt}")

    print("\nKey notes:")
    for note in data['notes']:
        print(f"  • {note}")
    print()

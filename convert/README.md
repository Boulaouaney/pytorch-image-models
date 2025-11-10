# Model Conversion Scripts

This directory contains scripts for converting PyTorch Image Models (timm) to various deployment formats for optimized inference across different platforms and hardware.

## Supported Formats

| Format | Target Platform | Script | Key Features |
|--------|----------------|--------|--------------|
| **ONNX** | Cross-platform | `convert_to_onnx.py` | Universal format, Dynamo-based export (PyTorch 2.5+) |
| **TFLite** | Mobile (Android/iOS) | `convert_to_tflite.py` | Google AI Edge Torch, quantization support |
| **CoreML** | Apple Devices | `convert_to_coreml.py` | iOS/macOS/iPadOS, FP16 support |
| **OpenVINO** | Intel Hardware | `convert_to_openvino.py` | CPU/GPU/VPU optimization, FP16 compression |
| **TensorRT** | NVIDIA GPUs | `convert_to_tensorrt.py` | FP16/INT8 precision, ONNX or direct path |
| **ExecuTorch** | Edge Devices | `convert_to_executorch.py` | PyTorch on-device, XNNPACK backend |

## Quick Start

### Unified Conversion Tool

The easiest way to convert models is using the unified conversion script:

```bash
# Convert to ONNX
python convert/convert_model.py resnet50 --format onnx

# Convert to TFLite with quantization
python convert/convert_model.py mobilenetv3_large_100 --format tflite --quantize

# Convert to multiple formats at once
python convert/convert_model.py efficientnet_b0 --format onnx tflite coreml
```

### Individual Conversion Scripts

Each format has a dedicated script with format-specific options:

```bash
# ONNX with dynamic shapes
python convert/convert_to_onnx.py model.onnx --model resnet50 --dynamic-size

# CoreML for iOS 16+
python convert/convert_to_coreml.py model.mlpackage --model mobilenetv3_large_100 \
    --minimum-deployment-target iOS16

# OpenVINO with FP16 compression
python convert/convert_to_openvino.py model --model efficientnet_b0 --fp16
```

## Installation Requirements

### Core Requirements
```bash
pip install torch torchvision timm
```

### Format-Specific Requirements

**ONNX:**
```bash
pip install onnx onnxruntime
```

**TFLite:**
```bash
pip install ai-edge-torch
```

**CoreML:**
```bash
pip install coremltools
```

**OpenVINO:**
```bash
pip install openvino openvino-dev
```

**TensorRT:**
- Install NVIDIA TensorRT from [developer.nvidia.com/tensorrt](https://developer.nvidia.com/tensorrt)
- Optional: `pip install torch2trt` for direct conversion

**ExecuTorch:**
```bash
pip install executorch
```

## Usage Examples

### ONNX Conversion

```bash
# Basic ONNX export with Dynamo (recommended for PyTorch 2.5+)
python convert/convert_to_onnx.py resnet50.onnx --model resnet50

# Legacy exporter
python convert/convert_to_onnx.py resnet50.onnx --model resnet50 --legacy

# Custom input size
python convert/convert_to_onnx.py model.onnx --model efficientnet_b0 \
    --input-size 3 224 224

# Dynamic batch and spatial dimensions
python convert/convert_to_onnx.py model.onnx --model resnet50 --dynamic-size
```

### TFLite Conversion

```bash
# Basic TFLite export
python convert/convert_to_tflite.py mobilenetv3.tflite --model mobilenetv3_large_100

# With dynamic range quantization (smaller model)
python convert/convert_to_tflite.py mobilenetv3.tflite --model mobilenetv3_large_100 --quantize

# Custom checkpoint
python convert/convert_to_tflite.py model.tflite --model resnet50 \
    --checkpoint path/to/checkpoint.pth
```

### CoreML Conversion

```bash
# Basic CoreML export
python convert/convert_to_coreml.py mobilenetv3.mlpackage --model mobilenetv3_large_100

# FP16 precision for smaller model
python convert/convert_to_coreml.py model.mlpackage --model efficientnet_b0 \
    --compute-precision float16

# With class labels
python convert/convert_to_coreml.py model.mlpackage --model resnet50 \
    --class-labels imagenet_classes.txt \
    --minimum-deployment-target iOS17
```

### OpenVINO Conversion

```bash
# Basic OpenVINO export
python convert/convert_to_openvino.py resnet50 --model resnet50

# FP16 compression
python convert/convert_to_openvino.py resnet50_fp16 --model resnet50 --fp16

# With benchmarking
python convert/convert_to_openvino.py resnet50 --model resnet50 --benchmark
```

### TensorRT Conversion

```bash
# ONNX-based conversion (recommended)
python convert/convert_to_tensorrt.py resnet50.engine --model resnet50

# FP16 precision
python convert/convert_to_tensorrt.py resnet50_fp16.engine --model resnet50 --fp16

# INT8 precision (requires calibration)
python convert/convert_to_tensorrt.py resnet50_int8.engine --model resnet50 --int8

# Direct conversion via torch2trt
python convert/convert_to_tensorrt.py resnet50.pth --model resnet50 --method torch2trt
```

### ExecuTorch Conversion

```bash
# Basic ExecuTorch export with XNNPACK
python convert/convert_to_executorch.py mobilenetv3.pte --model mobilenetv3_large_100

# Without XNNPACK backend
python convert/convert_to_executorch.py model.pte --model resnet50 --no-xnnpack

# With quantization (experimental)
python convert/convert_to_executorch.py model.pte --model mobilenetv3_large_100 --quantize
```

## Common Options

All conversion scripts support these common options:

```
--model, -m          Model architecture name from timm
--pretrained         Use pretrained weights (default: True)
--checkpoint         Path to custom checkpoint file
--num-classes        Override number of output classes
--img-size           Square input image size (e.g., 224)
--input-size C H W   Explicit input dimensions (e.g., 3 224 224)
--batch-size, -b     Batch size for export (default: 1)
--reparam            Reparameterize model before export
```

## Model Compatibility

### Highly Compatible Models
These models work well across all formats:
- MobileNetV2/V3
- EfficientNet (B0-B7)
- ResNet family
- RegNet family

### Format-Specific Notes

**TFLite & ExecuTorch:**
- Require models compatible with `torch.export()` (PyTorch 2.1+)
- May have limited operator support
- Use `exportable=True` flag (automatically set by scripts)

**CoreML:**
- ~70% PyTorch operator coverage
- Best with standard CNN architectures
- Some dynamic control flow limitations

**TensorRT:**
- ONNX path is most compatible
- Best performance with FP16 on modern GPUs
- INT8 requires calibration dataset

**OpenVINO:**
- Direct PyTorch conversion supported (2023.0+)
- Excellent Intel CPU/GPU performance
- FP16 compression recommended for deployment

## Troubleshooting

### Common Issues

**"Module not found" errors:**
```bash
# Install the required package for your target format
pip install <package-name>
```

**Export compatibility issues:**
- Add `--reparam` flag to reparameterize model
- Try different input sizes with `--img-size`
- Some models require specific PyTorch versions

**TFLite/ExecuTorch torch.export() failures:**
- Not all models are compatible with `torch.export()`
- Try simpler architectures (MobileNet, EfficientNet)
- Check PyTorch version (2.1+ required)

**TensorRT build failures:**
- Use ONNX path instead of torch2trt
- Reduce `--workspace-size` if out of memory
- Some operations may need custom plugins

### Validation

Validate converted models:

```bash
# ONNX validation (built-in)
python convert/convert_to_onnx.py model.onnx --model resnet50

# OpenVINO benchmarking
python convert/convert_to_openvino.py model --model resnet50 --benchmark

# Use the root-level onnx_validate.py script
python onnx_validate.py --model resnet50 --onnx-input model.onnx
```

## Performance Tips

1. **Quantization**: Use `--quantize` (TFLite) or `--int8` (TensorRT) for smaller models
2. **FP16**: Enable with `--fp16` for 2x smaller models with minimal accuracy loss
3. **Dynamic Shapes**: Only enable if needed, can impact performance
4. **Batch Size**: Use batch_size > 1 for throughput-optimized inference
5. **Reparameterization**: Use `--reparam` to fold batch norm for faster inference

## Advanced Usage

### Batch Conversion

Convert multiple models:

```bash
# Bash loop
for model in resnet50 mobilenetv3_large_100 efficientnet_b0; do
    python convert/convert_model.py $model --format onnx tflite
done
```

### Custom Model Configurations

```bash
# Custom number of classes
python convert/convert_model.py resnet50 --format onnx \
    --num-classes 10 --checkpoint my_checkpoint.pth

# Reparameterized model
python convert/convert_model.py repvgg_a0 --format onnx --reparam
```

## Contributing

When adding new conversion scripts:
1. Follow the existing script structure
2. Use argparse for CLI interface
3. Include comprehensive error handling
4. Add documentation to this README
5. Test with at least 3 different models

## References

- **ONNX**: [pytorch.org/docs/stable/onnx.html](https://pytorch.org/docs/stable/onnx.html)
- **TFLite**: [ai.google.dev/edge/litert](https://ai.google.dev/edge/litert)
- **CoreML**: [apple.github.io/coremltools](https://apple.github.io/coremltools)
- **OpenVINO**: [docs.openvino.ai](https://docs.openvino.ai)
- **TensorRT**: [developer.nvidia.com/tensorrt](https://developer.nvidia.com/tensorrt)
- **ExecuTorch**: [pytorch.org/executorch](https://pytorch.org/executorch)

## License

Copyright 2025 timm contributors. See main repository LICENSE file.

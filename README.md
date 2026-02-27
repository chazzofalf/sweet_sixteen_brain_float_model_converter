# Sweet Sixteen Brain Float Model Converter

A Python tool to convert any type of safetensors model file into bf16 (brain float 16) or float16 precision.

## Overview

This tool allows you to convert AI/ML models stored in the safetensors format to use bfloat16 (brain float 16) or float16 precision. Converting models to lower precision can significantly reduce memory usage while maintaining good numerical precision for deep learning workloads.

### Why bf16?

- **Reduced Memory Usage**: bf16 uses half the memory of fp32 models
- **Faster Inference**: Lower precision operations can be faster on modern hardware
- **Good Numerical Stability**: bf16 maintains the same exponent range as fp32, providing better numerical stability than fp16
- **Hardware Acceleration**: Modern GPUs (like NVIDIA's Ampere architecture) have dedicated bf16 processing units

### Why float16?

- **Reduced Memory Usage**: float16 uses half the memory of fp32 models
- **Faster Inference**: Lower precision operations can be faster on modern hardware
- **Wider Hardware Support**: float16 is supported on a wider range of hardware including CPUs with SIMD instructions
- **Smaller Model Size**: float16 models are smaller for storage and transfer

## Installation

```bash
pip install sweet-sixteen
```

### From Source

```bash
git clone https://github.com/chazzofalf/sweet_sixteen_brain_float_model_converter.git
cd sweet_sixteen_brain_float_model_converter
pip install .
```

## Prerequisites

- Python 3.8 or higher
- PyTorch 2.10.0 or compatible version
- A safetensors model file to convert

## Usage

### Command Line Interface

Convert to bf16 (default):
```bash
sweet-sixteen <input_model_path> <output_model_path>
```

Convert to float16:
```bash
sweet-sixteen --float16 <input_model_path> <output_model_path>
```

### Python API

Convert to bf16:
```python
from sweet_sixteen import convert_safetensors_to_bf16

stats = convert_safetensors_to_bf16(
    "path/to/input_model.safetensors",
    "path/to/output_model_bf16.safetensors"
)

print(f"Converted {stats['converted']} tensors")
```

Convert to float16:
```python
from sweet_sixteen import convert_safetensors_to_float16

stats = convert_safetensors_to_float16(
    "path/to/input_model.safetensors",
    "path/to/output_model_float16.safetensors"
)

print(f"Converted {stats['converted']} tensors")
```

### Command Line Arguments

| Argument | Description |
|----------|-------------|
| `input_model_path` | Path to the source safetensors model file |
| `output_model_path` | Path where the converted model will be saved |
| `--float16` | Convert to float16 instead of bf16 (optional) |

## Features

- **Universal Format Support**: Convert any safetensors model regardless of architecture
- **Multiple Precision Support**: Convert to bf16 or float16 precision
- **Automatic dtype Detection**: Automatically detects tensor data types
- **Smart Conversion**: Only converts non-converted tensors, preserving already converted ones
- **Device-Aware**: Handles both CPU and GPU tensors
- **Conversion Statistics**: Provides detailed conversion summary

## Conversion Process

### For bf16 conversion:
1. Loads the safetensors file
2. Iterates through all tensors in the model
3. Converts each tensor to bf16 through float32 (to preserve precision)
4. Saves the converted tensors to a new safetensors file

### For float16 conversion:
1. Loads the safetensors file
2. Iterates through all tensors in the model
3. Converts each tensor to float16 through float32 (to preserve precision)
4. Saves the converted tensors to a new safetensors file

## Output

The tool provides detailed progress information and a conversion summary including:
- Total number of tensors processed
- Number of tensors converted (to bf16 or float16 based on option)
- Number of tensors already in the target format
- List of data types encountered
- Output file location

## Example Output

### bf16 conversion:
```
Loading model from: /path/to/input_model.safetensors
Total tensors found: 245
Converting 'model.layers.0.self_attn.q_proj.weight' from torch.float32 to bf16
Converting 'model.layers.0.self_attn.k_proj.weight' from torch.float32 to bf16
...

==================================================
Conversion Summary
==================================================
Total tensors: 245
Converted to bf16: 238
Already bf16: 7
Data types seen: torch.float32, torch.bfloat16
==================================================

Success! Converted model saved to: /path/to/output_model_bf16.safetensors
```

### float16 conversion:
```
Loading model from: /path/to/input_model.safetensors
Total tensors found: 245
Converting 'model.layers.0.self_attn.q_proj.weight' from torch.float32 to float16
Converting 'model.layers.0.self_attn.k_proj.weight' from torch.float32 to float16
...

==================================================
Conversion Summary
==================================================
Total tensors: 245
Converted to float16: 238
Already float16: 7
Data types seen: torch.float32, torch.float16
==================================================

Success! Converted model saved to: /path/to/output_model_float16.safetensors
```

## Requirements

- Python 3.8+
- PyTorch >= 2.10.0
- safetensors >= 0.7.0

## License

This project is provided as-is for educational and practical use.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

If you encounter any issues or have questions, please open an issue on GitHub.

## Acknowledgments

- Built with [PyTorch](https://pytorch.org/)
- Safetensors support via [safetensors](https://github.com/huggingface/safetensors)

---

**Note**: Always keep a backup of your original model before conversion. While bf16 conversion is generally safe, it's good practice to preserve your original weights.
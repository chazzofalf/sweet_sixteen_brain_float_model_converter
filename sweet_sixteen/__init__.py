"""
Sweet Sixteen - Brain Float 16 Model Converter

A Python tool to convert safetensors model files into bf16 (brain float 16) precision.
"""

__version__ = "0.1.0"
__author__ = "Charles Montgomery"
__email__ = "chazzofalf@gmail.com"

from sweet_sixteen.converter import (
    convert_to_bf16,
    convert_safetensors_to_bf16,
    detect_dtype,
    get_bf16_dtype_name,
)

__all__ = [
    "convert_to_bf16",
    "convert_safetensors_to_bf16",
    "detect_dtype",
    "get_bf16_dtype_name",
    "__version__",
]
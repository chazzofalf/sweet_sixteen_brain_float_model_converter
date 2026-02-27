"""
Converter module for Sweet Sixteen - Brain Float 16 Model Converter

This module provides functionality to convert safetensors model files into bf16 (brain float 16) precision.
"""

import torch
from safetensors.torch import load_file, save_file


def get_bf16_dtype_name() -> str:
    """Return the string representation of bf16 dtype."""
    return "bf16"


def get_float16_dtype_name() -> str:
    """Return the string representation of float16 dtype."""
    return "float16"


def detect_dtype(tensor: torch.Tensor) -> str:
    """Detect and return the dtype name of a tensor."""
    return str(tensor.dtype)


def convert_to_bf16(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert a tensor to bf16 precision.
    
    Args:
        tensor: Input tensor of any dtype
        
    Returns:
        Tensor converted to bf16
        
    Raises:
        ValueError: If tensor cannot be converted to bf16
    """
    if tensor.dtype == torch.bfloat16:
        return tensor
    
    # Move to CPU for conversion if on GPU
    device = tensor.device
    tensor_cpu = tensor.cpu()
    
    # Convert through float32 to preserve precision when possible
    if tensor_cpu.dtype.is_floating_point:
        # For floating point tensors, convert through float32
        tensor_float32 = tensor_cpu.to(torch.float32)
        tensor_bf16 = tensor_float32.to(torch.bfloat16)
    else:
        # For non-floating point tensors, convert to bf16
        tensor_bf16 = tensor_cpu.to(torch.bfloat16)
    
    # Move back to original device if needed
    if device != torch.device('cpu'):
        tensor_bf16 = tensor_bf16.to(device)
    
    return tensor_bf16


def convert_to_float16(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert a tensor to float16 precision.
    
    Args:
        tensor: Input tensor of any dtype
        
    Returns:
        Tensor converted to float16
        
    Raises:
        ValueError: If tensor cannot be converted to float16
    """
    if tensor.dtype == torch.float16:
        return tensor
    
    # Move to CPU for conversion if on GPU
    device = tensor.device
    tensor_cpu = tensor.cpu()
    
    # Convert through float32 to preserve precision when possible
    if tensor_cpu.dtype.is_floating_point:
        # For floating point tensors, convert through float32
        tensor_float32 = tensor_cpu.to(torch.float32)
        tensor_float16 = tensor_float32.to(torch.float16)
    else:
        # For non-floating point tensors, convert to float16
        tensor_float16 = tensor_cpu.to(torch.float16)
    
    # Move back to original device if needed
    if device != torch.device('cpu'):
        tensor_float16 = tensor_float16.to(device)
    
    return tensor_float16


def convert_safetensors_to_bf16(input_path: str, output_path: str) -> dict:
    """
    Convert a safetensors model file to bf16 precision.
    
    Args:
        input_path: Path to input safetensors file
        output_path: Path to save converted bf16 model
        
    Returns:
        Dictionary with conversion statistics
    """
    print(f"Loading model from: {input_path}")
    
    # Load the safetensors file
    state_dict = load_file(input_path)
    
    # Track statistics
    stats = {
        'total_tensors': len(state_dict),
        'converted': 0,
        'already_bf16': 0,
        'types_seen': set()
    }
    
    print(f"Total tensors found: {stats['total_tensors']}")
    
    # Convert each tensor to bf16
    for key, tensor in state_dict.items():
        original_dtype = detect_dtype(tensor)
        stats['types_seen'].add(original_dtype)
        
        if tensor.dtype == torch.bfloat16:
            stats['already_bf16'] += 1
            continue
        
        print(f"Converting '{key}' from {original_dtype} to {get_bf16_dtype_name()}")
        
        # Convert to bf16
        try:
            state_dict[key] = convert_to_bf16(tensor)
            stats['converted'] += 1
        except Exception as e:
            print(f"Error converting '{key}': {e}")
            raise
    
    # Save the converted model
    print(f"\nSaving converted model to: {output_path}")
    save_file(state_dict, output_path)
    
    return stats


def convert_safetensors_to_float16(input_path: str, output_path: str) -> dict:
    """
    Convert a safetensors model file to float16 precision.
    
    Args:
        input_path: Path to input safetensors file
        output_path: Path to save converted float16 model
        
    Returns:
        Dictionary with conversion statistics
    """
    print(f"Loading model from: {input_path}")
    
    # Load the safetensors file
    state_dict = load_file(input_path)
    
    # Track statistics
    stats = {
        'total_tensors': len(state_dict),
        'converted': 0,
        'already_float16': 0,
        'types_seen': set()
    }
    
    print(f"Total tensors found: {stats['total_tensors']}")
    
    # Convert each tensor to float16
    for key, tensor in state_dict.items():
        original_dtype = detect_dtype(tensor)
        stats['types_seen'].add(original_dtype)
        
        if tensor.dtype == torch.float16:
            stats['already_float16'] += 1
            continue
        
        print(f"Converting '{key}' from {original_dtype} to {get_float16_dtype_name()}")
        
        # Convert to float16
        try:
            state_dict[key] = convert_to_float16(tensor)
            stats['converted'] += 1
        except Exception as e:
            print(f"Error converting '{key}': {e}")
            raise
    
    # Save the converted model
    print(f"\nSaving converted model to: {output_path}")
    save_file(state_dict, output_path)
    
    return stats

"""
Command-line interface for Sweet Sixteen - Brain Float 16 Model Converter

Convert safetensors model files to bf16 or float16 precision.

Usage:
    python -m sweet_sixteen <input_model_path> <output_model_path>          Convert to bf16
    python -m sweet_sixteen --float16 <input_model_path> <output_model_path>  Convert to float16

Arguments:
    input_model_path   Path to the source safetensors model file
    output_model_path  Path where the converted model will be saved

Options:
    --float16    Convert to float16 instead of bf16
"""

import sys
from sweet_sixteen.converter import convert_safetensors_to_bf16, convert_safetensors_to_float16


def print_summary(stats: dict, dtype: str):
    """Print conversion summary for the given dtype."""
    already_dtype = f"already_{dtype}"
    print("\n" + "=" * 50)
    print("Conversion Summary")
    print("=" * 50)
    print(f"Total tensors: {stats['total_tensors']}")
    print(f"Converted to {dtype}: {stats['converted']}")
    print(f"Already {dtype}: {stats.get(already_dtype, 0)}")
    print(f"Data types seen: {', '.join(sorted(stats['types_seen']))}")
    print("=" * 50)


def main():
    """Main entry point for the CLI."""
    # Parse command line arguments
    args = sys.argv[1:]
    
    # Check for --float16 flag
    use_float16 = False
    if "--float16" in args:
        use_float16 = True
        args.remove("--float16")
    
    # Validate arguments
    if len(args) != 2:
        print(__doc__)
        sys.exit(1)
    
    input_path = args[0]
    output_path = args[1]
    
    try:
        if use_float16:
            stats = convert_safetensors_to_float16(input_path, output_path)
            print_summary(stats, "float16")
        else:
            stats = convert_safetensors_to_bf16(input_path, output_path)
            print_summary(stats, "bf16")
        
        dtype_name = "float16" if use_float16 else "bf16"
        print(f"\nSuccess! Converted model saved to: {output_path}")
        
    except FileNotFoundError as e:
        print(f"Error: Input file not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

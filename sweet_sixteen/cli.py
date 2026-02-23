"""
Command-line interface for Sweet Sixteen - Brain Float 16 Model Converter
"""

import sys
from sweet_sixteen.converter import convert_safetensors_to_bf16


def main():
    """Main entry point for the CLI."""
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    try:
        stats = convert_safetensors_to_bf16(input_path, output_path)
        
        print("\n" + "=" * 50)
        print("Conversion Summary")
        print("=" * 50)
        print(f"Total tensors: {stats['total_tensors']}")
        print(f"Converted to bf16: {stats['converted']}")
        print(f"Already bf16: {stats['already_bf16']}")
        print(f"Data types seen: {', '.join(sorted(stats['types_seen']))}")
        print("=" * 50)
        print(f"\nSuccess! Converted model saved to: {output_path}")
        
    except FileNotFoundError as e:
        print(f"Error: Input file not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""TensorBus CLI - Debug and inspect TensorBus instances."""
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

try:
    import numpy as np
    import onnx
except ImportError:
    print("Error: tensorbus CLI requires onnx and numpy", file=sys.stderr)
    sys.exit(1)


def list_tensors(bus_dir: Path) -> None:
    """List all tensor snapshots in a POSIX bus directory."""
    if not bus_dir.exists():
        print(f"Error: Directory {bus_dir} does not exist", file=sys.stderr)
        sys.exit(1)
    
    onnx_files = list(bus_dir.glob("*.onnx"))
    
    if not onnx_files:
        print(f"No tensor snapshots found in {bus_dir}")
        return
    
    print(f"Found {len(onnx_files)} tensor snapshot(s) in {bus_dir}:\n")
    print(f"{'Name':<40} {'Size':<15} {'Modified':<20}")
    print("-" * 80)
    
    for path in sorted(onnx_files):
        size = path.stat().st_size
        mtime = path.stat().st_mtime
        size_str = f"{size:,} bytes" if size < 1024 else f"{size/1024:.1f} KB"
        
        from datetime import datetime
        mtime_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"{path.stem:<40} {size_str:<15} {mtime_str:<20}")


def info_tensor(bus_dir: Path, tensor_name: str) -> None:
    """Show detailed information about a tensor."""
    tensor_path = bus_dir / f"{tensor_name}.onnx"
    
    if not tensor_path.exists():
        print(f"Error: Tensor '{tensor_name}' not found in {bus_dir}", file=sys.stderr)
        sys.exit(1)
    
    try:
        model = onnx.load(str(tensor_path))
        
        print(f"Tensor: {tensor_name}")
        print(f"Path: {tensor_path}")
        print(f"Size: {tensor_path.stat().st_size:,} bytes")
        print(f"\nONNX Model Info:")
        print(f"  IR Version: {model.ir_version}")
        print(f"  Producer: {model.producer_name} {model.producer_version}")
        print(f"  Domain: {model.domain}")
        
        if model.graph.input:
            print(f"\n  Inputs:")
            for inp in model.graph.input:
                dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
                dtype = onnx.TensorProto.DataType.Name(inp.type.tensor_type.elem_type)
                print(f"    {inp.name}: {dtype} {dims}")
        
        if model.graph.output:
            print(f"\n  Outputs:")
            for out in model.graph.output:
                dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
                dtype = onnx.TensorProto.DataType.Name(out.type.tensor_type.elem_type)
                print(f"    {out.name}: {dtype} {dims}")
        
        if model.graph.initializer:
            print(f"\n  Initializers: {len(model.graph.initializer)}")
            for init in model.graph.initializer[:5]:  # Show first 5
                dims = list(init.dims)
                dtype = onnx.TensorProto.DataType.Name(init.data_type)
                print(f"    {init.name}: {dtype} {dims}")
            if len(model.graph.initializer) > 5:
                print(f"    ... and {len(model.graph.initializer) - 5} more")
        
        if model.metadata_props:
            print(f"\n  Metadata:")
            for prop in model.metadata_props:
                print(f"    {prop.key}: {prop.value}")
        
    except Exception as e:
        print(f"Error loading tensor: {e}", file=sys.stderr)
        sys.exit(1)


def get_tensor_values(bus_dir: Path, tensor_name: str, output_format: str = "text") -> None:
    """Get tensor values."""
    tensor_path = bus_dir / f"{tensor_name}.onnx"
    
    if not tensor_path.exists():
        print(f"Error: Tensor '{tensor_name}' not found in {bus_dir}", file=sys.stderr)
        sys.exit(1)
    
    try:
        import onnxruntime as ort
        
        # Load and run identity model to extract values
        sess = ort.InferenceSession(str(tensor_path), providers=["CPUExecutionProvider"])
        
        # Get first initializer as the tensor data
        model = onnx.load(str(tensor_path))
        
        if model.graph.initializer:
            init = model.graph.initializer[0]
            tensor = onnx.numpy_helper.to_array(init)
            
            if output_format == "json":
                output = {
                    "name": tensor_name,
                    "dtype": str(tensor.dtype),
                    "shape": list(tensor.shape),
                    "values": tensor.tolist()
                }
                print(json.dumps(output, indent=2))
            else:
                print(f"Tensor: {tensor_name}")
                print(f"Shape: {tensor.shape}")
                print(f"Dtype: {tensor.dtype}")
                print(f"\nValues:")
                
                # Show subset if large
                if tensor.size > 100:
                    flat = tensor.flatten()
                    print(f"  [first 10] {flat[:10]}")
                    print(f"  ...")
                    print(f"  [last 10] {flat[-10:]}")
                    print(f"\n  (showing 20 of {tensor.size} values)")
                else:
                    print(tensor)
        else:
            print(f"No tensor data found in {tensor_name}", file=sys.stderr)
            
    except Exception as e:
        print(f"Error extracting values: {e}", file=sys.stderr)
        sys.exit(1)


def watch_directory(bus_dir: Path, interval: float = 1.0) -> None:
    """Watch directory for changes."""
    import time
    
    print(f"Watching {bus_dir} for changes (Ctrl+C to stop)...")
    print()
    
    last_files = set()
    
    try:
        while True:
            current_files = set(bus_dir.glob("*.onnx"))
            
            # New files
            new_files = current_files - last_files
            for path in new_files:
                print(f"[+] {path.stem} (created)")
            
            # Deleted files
            deleted_files = last_files - current_files
            for path in deleted_files:
                print(f"[-] {path.stem} (deleted)")
            
            last_files = current_files
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nStopped watching")


def main():
    parser = argparse.ArgumentParser(
        description="TensorBus CLI - Debug and inspect TensorBus instances",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all tensors
  tensorbus list /tmp/tensorbus
  
  # Show tensor info
  tensorbus info /tmp/tensorbus my.input
  
  # Get tensor values
  tensorbus get /tmp/tensorbus my.input
  tensorbus get /tmp/tensorbus my.input --format json
  
  # Watch for changes
  tensorbus watch /tmp/tensorbus
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all tensors")
    list_parser.add_argument("directory", type=Path, help="Bus directory path")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show tensor information")
    info_parser.add_argument("directory", type=Path, help="Bus directory path")
    info_parser.add_argument("tensor", help="Tensor name")
    
    # Get command
    get_parser = subparsers.add_parser("get", help="Get tensor values")
    get_parser.add_argument("directory", type=Path, help="Bus directory path")
    get_parser.add_argument("tensor", help="Tensor name")
    get_parser.add_argument("--format", choices=["text", "json"], default="text",
                           help="Output format")
    
    # Watch command
    watch_parser = subparsers.add_parser("watch", help="Watch directory for changes")
    watch_parser.add_argument("directory", type=Path, help="Bus directory path")
    watch_parser.add_argument("--interval", type=float, default=1.0,
                            help="Poll interval in seconds")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == "list":
        list_tensors(args.directory)
    elif args.command == "info":
        info_tensor(args.directory, args.tensor)
    elif args.command == "get":
        get_tensor_values(args.directory, args.tensor, args.format)
    elif args.command == "watch":
        watch_directory(args.directory, args.interval)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import json
import sys
import argparse
import statistics
import os
import glob
from datetime import datetime

# Accelerator and model definitions for MFU calculation
ACCELERATORS = ["b200", "h100", "h200", "a100", "v5e", "v5p"]

MAX_TFLOPS = {
    ("h100", "bf16"): 989,  # https://resources.nvidia.com/en-us-tensor-core page39 - bf16
    ("h100", "fp8"): 1978,  # https://resources.nvidia.com/en-us-tensor-core page39 - fp8
    ("v5e", "bf16"): 197,  # https://cloud.google.com/tpu/docs/v5e
    ("v5p", "bf16"): 459,  # https://cloud.google.com/tpu/docs/v5p
    ("a100", "bf16"): 312,  # https://resources.nvidia.com/en-us-tensor-core page39 - bf16
    ("h200", "bf16"): 989,
    ("h200", "fp8"): 1978,
    ("b200", "bf16"): 2250,  # https://www.nvidia.com/en-us/data-center/hgx/
    ("b200", "fp8"): 4500,
}

MODEL_FLOPS_PER_SAMPLE = {
    "gpt3-5b": 6.69e13,
    "gpt3-175b": 2.2e15,
    "llama2-7b": 1.89e14,
    "llama2-13b": 3.19e14,
    "llama2-70b": 1.82e15,
    "llama3-70b": 3.94e15,
    "llama3.1-70b": 3.9129e15,
    "llama3.1-405b": 2.16533e16,
    "mixtral-7b": 3.4e14,
}

def compute_mfu(
    step_time: float,
    max_tflops: float,
    num_accelerators: int,
    model_flops_per_sample: float,
    batch_size: int,
) -> float:
    """Computes the MFU

    Args:
        step_time (float): forward + backward step time in seconds
        max_tflops (float): Max theoretical TFLOPs supported by the accelerator used
        num_accelerators (int): Number of accelerators used during the training process
        model_flops_per_sample (float): Number of FLOPS for a single sample training step
        batch_size (int): Global batch size used during training

    Returns:
        float: Returns the Model FLOPS Utilization MFU
    """
    tflops_per_accelerator = (
        model_flops_per_sample * batch_size / step_time / num_accelerators / 1e12
    )
    mfu = tflops_per_accelerator / max_tflops

    return mfu, tflops_per_accelerator

def process_dllogger(file_path, num_warm_up, num_gpus, accelerator=None, accelerator_precision=None, model_name=None):
    """
    Process the DLLogger JSON file to calculate tokens/sec/GPU and MFU
    
    Args:
        file_path (str): Path to the dllogger.json file
        num_warm_up (int): Number of initial steps to skip
        num_gpus (int): Number of GPUs used for training
        accelerator (str): Accelerator type used (e.g., "h100", "a100")
        accelerator_precision (str): Precision used (e.g., "bf16", "fp8")
        model_name (str): Model name for FLOPS calculation (e.g., "llama2-70b")
    
    Returns:
        dict: A dictionary with processed metrics
    """
    results = {
        'file_path': file_path,
        'num_gpus': num_gpus,
        'accelerator': accelerator,
        'precision': accelerator_precision,
        'model_name': model_name
    }
    step_times = []
    step_losses = []
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        return results
    except Exception as e:
        print(f"Error reading file '{file_path}': {e}")
        return results
    
    for i, line in enumerate(lines):
        # Skip any lines that don't start with DLLL
        if not line.startswith('DLLL '):
            continue
            
        # Extract the JSON part of the line
        json_str = line[5:]  # Skip the "DLLL " prefix
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            print(f"Warning: JSON decode error on line {i+1}, skipping")
            continue
        
        if i == 0:  # First line contains configuration
            # Extract configuration data
            try:
                results['global_batch_size'] = data['data']['cfg/global_batch_size']
                results['encoder_seq_length'] = data['data']['cfg/encoder_seq_length']
                results['hidden_size'] = data['data'].get('cfg/hidden_size', 'N/A')
                results['num_layers'] = data['data'].get('cfg/num_layers', 'N/A')
                results['tp_size'] = data['data'].get('cfg/tensor_model_parallel_size', 'N/A')
                results['pp_size'] = data['data'].get('cfg/pipeline_model_parallel_size', 'N/A')
                results['micro_batch_size'] = data['data'].get('cfg/micro_batch_size', 'N/A')
                
                # If precision is not provided as an argument, try to get it from the file
                if not accelerator_precision:
                    file_precision = data['data'].get('cfg/precision', '')
                    if 'fp8' in file_precision:
                        results['precision'] = 'fp8'
                    elif 'bf16' in file_precision:
                        results['precision'] = 'bf16'
                    else:
                        results['precision'] = file_precision
                else:
                    results['precision'] = accelerator_precision
                
            except KeyError as e:
                print(f"Warning: Could not find key {e} in configuration data")
        else:  # Subsequent lines contain step data
            # Extract timing and loss data from each step
            if 'train_step_timing in s' in data['data']:
                step_times.append(data['data']['train_step_timing in s'])
            if 'reduced_train_loss' in data['data']:
                step_losses.append(data['data']['reduced_train_loss'])
    
    # Skip warm-up steps and calculate mean step time
    if len(step_times) <= num_warm_up:
        print(f"Warning: number of steps ({len(step_times)}) is less than or equal to warm-up steps ({num_warm_up})")
        valid_step_times = step_times
        valid_step_losses = step_losses
    else:
        valid_step_times = step_times[num_warm_up:]
        valid_step_losses = step_losses[num_warm_up:] if len(step_losses) > num_warm_up else step_losses
    
    if not valid_step_times:
        print("Error: No valid step times found after warm-up")
        return results
    
    # Calculate statistics
    mean_step_time = statistics.mean(valid_step_times)
    results['mean_step_time'] = mean_step_time
    results['min_step_time'] = min(valid_step_times)
    results['max_step_time'] = max(valid_step_times)
    results['stdev_step_time'] = statistics.stdev(valid_step_times) if len(valid_step_times) > 1 else 0
    
    # Calculate iteration time in ms
    results['iteration_time_ms'] = mean_step_time * 1000
    
    # Calculate tokens/sec/GPU
    tokens_per_batch = results.get('global_batch_size', 0) * results.get('encoder_seq_length', 0)
    tokens_per_sec_per_gpu = tokens_per_batch / mean_step_time / num_gpus
    results['tokens/sec/GPU'] = tokens_per_sec_per_gpu
    results['throughput'] = tokens_per_batch / mean_step_time  # Total throughput
    
    # Calculate MFU if all necessary information is provided
    if accelerator and results['precision'] and model_name and model_name in MODEL_FLOPS_PER_SAMPLE:
        accel_key = (accelerator, results['precision'])
        if accel_key in MAX_TFLOPS:
            max_tflops = MAX_TFLOPS[accel_key]
            model_flops = MODEL_FLOPS_PER_SAMPLE[model_name]
            batch_size = results['global_batch_size']
            
            mfu, tflops_per_gpu = compute_mfu(
                step_time=mean_step_time,
                max_tflops=max_tflops,
                num_accelerators=num_gpus,
                model_flops_per_sample=model_flops,
                batch_size=batch_size
            )
            
            results['mfu'] = mfu
            results['tflops_per_gpu'] = tflops_per_gpu
    
    # Calculate loss statistics
    if valid_step_losses:
        results['final_loss'] = valid_step_losses[-1]
        results['mean_loss'] = statistics.mean(valid_step_losses)
        results['min_loss'] = min(valid_step_losses)
        results['loss_improvement'] = valid_step_losses[0] - valid_step_losses[-1] if len(valid_step_losses) > 1 else 0
    
    # Get number of steps
    results['num_steps'] = len(step_times)
    results['num_valid_steps'] = len(valid_step_times)
    
    return results

def create_simple_table(headers, rows):
    """Create a simple ASCII table without using the tabulate library"""
    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    
    # Create the header
    header_line = ' | '.join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    separator = '-+-'.join('-' * w for w in col_widths)
    
    # Create the rows
    table_rows = []
    for row in rows:
        table_rows.append(' | '.join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)))
    
    # Combine all parts
    return '\n'.join([header_line, separator] + table_rows)

def print_results(results, verbose=False, output_file=None, warm_up=5):
    """
    Print the results in a formatted way
    
    Args:
        results: List of result dictionaries
        verbose: Whether to print detailed information
        output_file: Path to file for saving results
        warm_up: Number of warm-up steps skipped
    """
    if not results:
        print("No results to display")
        return
    
    # Prepare output
    output = []
    output.append("\n======= DLLogger Statistics =======\n")
    
    # Create a summary table
    table_data = []
    # Add MFU and iteration time to the headers if available in any result
    has_mfu = any('mfu' in result for result in results)
    
    headers = ["Run", "GPUs", "Batch", "Seq Len"]
    highlight_headers = ["Iter Time (ms)", "Tokens/sec/GPU"]
    if has_mfu:
        highlight_headers.insert(1, "MFU")
    
    headers.extend(highlight_headers)
    headers.extend(["Mean Step (s)", "Final Loss"])
    
    if verbose:
        headers.extend(["Min Step (s)", "Max Step (s)", "StdDev", "TP Size", "PP Size"])
    
    for i, result in enumerate(results, 1):
        # Extract run name from path - more intelligently
        file_path = result['file_path']
        # Try to extract model name from path
        if '/' in file_path:
            path_parts = file_path.split('/')
            if len(path_parts) >= 2:
                # Look for directories that might contain model names
                for part in path_parts:
                    if 'llama' in part.lower() or 'gpt' in part.lower() or 'nemo' in part.lower():
                        run_name = part
                        break
                else:
                    # If no model name found, use the parent directory
                    run_name = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
            else:
                run_name = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
        else:
            run_name = "unknown"
        
        # Prepare row data
        row = [
            run_name,
            result.get('num_gpus', 'N/A'),
            result.get('global_batch_size', 'N/A'),
            result.get('encoder_seq_length', 'N/A')
        ]
        
        # Add highlighted metrics
        iter_time_ms = f"{result.get('iteration_time_ms', 0):.2f}"
        tokens_per_sec = f"{result.get('tokens/sec/GPU', 0):.2f}"
        
        # Add MFU if available
        if has_mfu:
            if 'mfu' in result:
                mfu_value = f"{result.get('mfu', 0):.4f}"
                highlight_values = [f"✦ {iter_time_ms} ✦", f"✦ {mfu_value} ✦", f"✦ {tokens_per_sec} ✦"]
            else:
                highlight_values = [f"✦ {iter_time_ms} ✦", "N/A", f"✦ {tokens_per_sec} ✦"]
        else:
            highlight_values = [f"✦ {iter_time_ms} ✦", f"✦ {tokens_per_sec} ✦"]
        
        row.extend(highlight_values)
        
        # Add remaining standard fields
        row.extend([
            f"{result.get('mean_step_time', 0):.4f}",
            f"{result.get('final_loss', 'N/A'):.6f}" if result.get('final_loss') is not None else 'N/A'
        ])
        
        if verbose:
            row.extend([
                f"{result.get('min_step_time', 0):.4f}",
                f"{result.get('max_step_time', 0):.4f}",
                f"{result.get('stdev_step_time', 0):.4f}",
                result.get('tp_size', 'N/A'),
                result.get('pp_size', 'N/A')
            ])
        
        table_data.append(row)
    
    # Generate the summary table
    table = create_simple_table(headers, table_data)
    output.append(table)
    
    # Add detailed information if verbose
    if verbose:
        for i, result in enumerate(results, 1):
            # Extract run name (same logic as above)
            file_path = result['file_path']
            if '/' in file_path:
                path_parts = file_path.split('/')
                if len(path_parts) >= 2:
                    for part in path_parts:
                        if 'llama' in part.lower() or 'gpt' in part.lower() or 'nemo' in part.lower():
                            run_name = part
                            break
                    else:
                        run_name = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
                else:
                    run_name = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
            else:
                run_name = "unknown"
            
            output.append(f"\n\n--- Detailed Stats for Run: {run_name} ---")
            output.append(f"File: {result['file_path']}")
            output.append(f"Model Config: {result.get('num_layers', 'N/A')} layers, {result.get('hidden_size', 'N/A')} hidden size")
            output.append(f"Parallelism: TP={result.get('tp_size', 'N/A')}, PP={result.get('pp_size', 'N/A')}")
            output.append(f"Batch Config: Global={result.get('global_batch_size', 'N/A')}, Micro={result.get('micro_batch_size', 'N/A')}")
            output.append(f"Sequence Length: {result.get('encoder_seq_length', 'N/A')}")
            output.append(f"Precision: {result.get('precision', 'N/A')}")
            output.append(f"Number of GPUs: {result.get('num_gpus', 'N/A')}")
            
            # Highlight key performance metrics
            output.append(f"✦ Iteration Time: {result.get('iteration_time_ms', 0):.2f} ms ✦")
            if 'mfu' in result:
                output.append(f"✦ MFU: {result.get('mfu', 0):.4f} ✦")
                output.append(f"  TFLOPS/GPU: {result.get('tflops_per_gpu', 0):.4f}")
            output.append(f"✦ Tokens/sec/GPU: {result.get('tokens/sec/GPU', 0):.2f} ✦")
            
            output.append(f"Step Times: Mean={result.get('mean_step_time', 0):.4f}s, Min={result.get('min_step_time', 0):.4f}s, Max={result.get('max_step_time', 0):.4f}s, StdDev={result.get('stdev_step_time', 0):.4f}s")
            output.append(f"Total Throughput: {result.get('throughput', 0):.2f} tokens/sec")
            
            if 'mean_loss' in result:
                output.append(f"Loss: Final={result.get('final_loss', 'N/A'):.6f}, Mean={result.get('mean_loss', 0):.6f}, Min={result.get('min_loss', 0):.6f}")
                output.append(f"Loss Improvement: {result.get('loss_improvement', 0):.6f}")
            
            output.append(f"Steps: Total={result.get('num_steps', 0)}, Analyzed={result.get('num_valid_steps', 0)} (after {warm_up} warm-up steps)")
            
            # Add model and accelerator info if available for MFU calculation
            if result.get('model_name') or result.get('accelerator'):
                output.append(f"MFU Calculation: Model={result.get('model_name', 'N/A')}, Accelerator={result.get('accelerator', 'N/A')}, Precision={result.get('precision', 'N/A')}")
    
    # Print the output
    final_output = "\n".join(output)
    print(final_output)
    
    # Write to file if specified
    if output_file:
        try:
            with open(output_file, 'w') as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"# DLLogger Analysis - Generated on {timestamp}\n\n")
                f.write(final_output)
            print(f"\nResults saved to {output_file}")
        except Exception as e:
            print(f"Error writing to output file: {e}")

def main():
    parser = argparse.ArgumentParser(description='Process DLLogger data and calculate performance metrics including tokens/sec/GPU and MFU')
    parser.add_argument('--file', type=str, default='dllogger.json', 
                        help='Path to the dllogger.json file (can use wildcards for multiple files)')
    parser.add_argument('--warm_up', type=int, default=5, 
                        help='Number of warm-up steps to skip')
    parser.add_argument('--num_gpus', type=int, required=True, 
                        help='Number of GPUs used for training')
    parser.add_argument('--verbose', '-v', action='store_true', 
                        help='Print detailed information')
    parser.add_argument('--output', '-o', type=str, 
                        help='Save results to specified file')
    parser.add_argument('--accelerator', type=str, choices=ACCELERATORS, 
                        help='Accelerator type for MFU calculation')
    parser.add_argument('--precision', type=str, choices=['bf16', 'fp8'], 
                        help='Precision used for training (needed for MFU calculation)')
    parser.add_argument('--model', type=str, choices=list(MODEL_FLOPS_PER_SAMPLE.keys()), 
                        help='Model name for MFU calculation')
    
    args = parser.parse_args()
    
    # Expand wildcards in file path
    file_paths = glob.glob(args.file)
    
    if not file_paths:
        print(f"Error: No files found matching '{args.file}'")
        return 1
    
    all_results = []
    for file_path in file_paths:
        results = process_dllogger(
            file_path, 
            args.warm_up, 
            args.num_gpus,
            args.accelerator,
            args.precision,
            args.model
        )
        all_results.append(results)
    
    print_results(all_results, args.verbose, args.output, args.warm_up)
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

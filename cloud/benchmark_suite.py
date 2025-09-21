#!/usr/bin/env python3
"""
Comprehensive benchmarking suite for batch-invariant operations.

This script benchmarks the performance of batch-invariant operations compared to
standard PyTorch operations across different scenarios, batch sizes, and hardware.
"""

import time
import json
import csv
import argparse
import platform
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

try:
    from batch_invariant_ops import set_batch_invariant_mode, __version__
    LIBRARY_AVAILABLE = True
except ImportError:
    LIBRARY_AVAILABLE = False
    print("⚠️ batch-invariant-ops not installed. Install with: pip install -e .")


class BenchmarkSuite:
    """Comprehensive benchmarking suite for batch-invariant operations."""

    def __init__(self, device='cuda', warmup_iterations=5, benchmark_iterations=50):
        self.device = device
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        self.results = []

        if not LIBRARY_AVAILABLE:
            raise ImportError("batch-invariant-ops library not available")

        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")

        torch.set_default_device(device)

    def benchmark_operation(self, name: str, operation, *args, **kwargs) -> Dict[str, Any]:
        """Benchmark an operation with both standard and batch-invariant modes."""

        def time_operation(mode_enabled: bool) -> float:
            """Time an operation with specified mode."""
            with set_batch_invariant_mode(mode_enabled):
                # Warmup
                for _ in range(self.warmup_iterations):
                    try:
                        _ = operation(*args, **kwargs)
                    except Exception as e:
                        return float('inf')  # Mark as failed

                if self.device == 'cuda':
                    torch.cuda.synchronize()

                # Timing
                start = time.time()
                for _ in range(self.benchmark_iterations):
                    _ = operation(*args, **kwargs)

                if self.device == 'cuda':
                    torch.cuda.synchronize()
                end = time.time()

                return (end - start) / self.benchmark_iterations

        # Measure memory before operation
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        time_standard = time_operation(False)
        memory_standard = torch.cuda.max_memory_allocated() if self.device == 'cuda' else 0

        if self.device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        time_invariant = time_operation(True)
        memory_invariant = torch.cuda.max_memory_allocated() if self.device == 'cuda' else 0

        # Calculate metrics
        speedup = time_standard / time_invariant if time_invariant > 0 and time_invariant != float('inf') else 0
        memory_diff = memory_invariant - memory_standard

        result = {
            'name': name,
            'time_standard_ms': time_standard * 1000,
            'time_invariant_ms': time_invariant * 1000,
            'speedup': speedup,
            'memory_standard_mb': memory_standard / 1e6,
            'memory_invariant_mb': memory_invariant / 1e6,
            'memory_diff_mb': memory_diff / 1e6,
            'successful': time_invariant != float('inf')
        }

        self.results.append(result)
        return result

    def benchmark_matrix_operations(self):
        """Benchmark matrix operations with different sizes."""
        print("🧮 Benchmarking Matrix Operations...")

        sizes = [
            (128, 256, 512),    # Small
            (512, 1024, 512),   # Medium
            (1024, 2048, 1024), # Large
            (2048, 4096, 2048), # Very large
        ]

        for m, k, n in sizes:
            a = torch.randn(m, k, device=self.device)
            b = torch.randn(k, n, device=self.device)
            bias = torch.randn(n, device=self.device)

            # Matrix multiplication
            result = self.benchmark_operation(f"mm_{m}x{k}x{n}", torch.mm, a, b)
            print(f"  {result['name']}: {result['speedup']:.2f}x speedup")

            # Matrix multiplication with bias
            result = self.benchmark_operation(f"addmm_{m}x{k}x{n}", torch.addmm, bias, a, b)
            print(f"  {result['name']}: {result['speedup']:.2f}x speedup")

    def benchmark_activation_functions(self):
        """Benchmark activation functions."""
        print("📊 Benchmarking Activation Functions...")

        sizes = [
            (64, 1000),     # Small vocab
            (256, 10000),   # Medium vocab
            (512, 32000),   # Large vocab (LLM-like)
            (1024, 50000),  # Very large vocab
        ]

        for batch_size, vocab_size in sizes:
            x = torch.randn(batch_size, vocab_size, device=self.device)

            result = self.benchmark_operation(f"log_softmax_{batch_size}x{vocab_size}",
                                            torch.log_softmax, x, -1)
            print(f"  {result['name']}: {result['speedup']:.2f}x speedup")

    def benchmark_reduction_operations(self):
        """Benchmark reduction operations."""
        print("📈 Benchmarking Reduction Operations...")

        test_cases = [
            # (shape, reduction_dims, name)
            ((128, 256), [1], "mean_2d_last"),
            ((32, 64, 128), [1], "mean_3d_middle"),
            ((32, 64, 128), [1, 2], "mean_3d_multiple"),
            ((16, 32, 64, 128), [2], "mean_4d_single"),
            ((16, 32, 64, 128), [1, 3], "mean_4d_multiple"),
        ]

        for shape, dims, name in test_cases:
            x = torch.randn(shape, device=self.device)
            result = self.benchmark_operation(f"mean_{name}_{shape}", torch.mean, x, dims)
            print(f"  {result['name']}: {result['speedup']:.2f}x speedup")

    def benchmark_batch_invariance_test(self):
        """Benchmark the specific batch invariance test scenario."""
        print("🔬 Benchmarking Batch Invariance Scenarios...")

        batch_sizes = [1, 8, 32, 128, 512, 1024, 2048]
        D = 4096

        for B in batch_sizes:
            if B > 2048:  # Skip very large batches for memory
                continue

            a = torch.randn(B, D, device=self.device)
            b = torch.randn(D, D, device=self.device)

            # Single row computation
            result1 = self.benchmark_operation(f"single_row_B{B}", torch.mm, a[:1], b)

            # Full batch then slice
            def full_batch_slice():
                return torch.mm(a, b)[:1]

            result2 = self.benchmark_operation(f"full_batch_slice_B{B}", full_batch_slice)

            print(f"  Batch {B}: Single={result1['speedup']:.2f}x, Slice={result2['speedup']:.2f}x")

    def benchmark_memory_scaling(self):
        """Benchmark memory usage scaling."""
        print("💾 Benchmarking Memory Scaling...")

        sizes = [512, 1024, 2048, 4096]

        for size in sizes:
            try:
                a = torch.randn(size, size, device=self.device)
                b = torch.randn(size, size, device=self.device)

                result = self.benchmark_operation(f"memory_scale_{size}x{size}", torch.mm, a, b)
                print(f"  {size}x{size}: {result['memory_diff_mb']:+.1f} MB difference")

                # Clean up large tensors
                del a, b
                if self.device == 'cuda':
                    torch.cuda.empty_cache()

            except RuntimeError as e:
                print(f"  {size}x{size}: Failed - {e}")

    def run_comprehensive_benchmark(self):
        """Run the complete benchmark suite."""
        print("🏎️ Starting Comprehensive Benchmark Suite")
        print("=" * 50)

        # System info
        self.print_system_info()

        # Run benchmarks
        self.benchmark_matrix_operations()
        self.benchmark_activation_functions()
        self.benchmark_reduction_operations()
        self.benchmark_batch_invariance_test()
        self.benchmark_memory_scaling()

        print("\n✅ Benchmark complete!")

    def print_system_info(self):
        """Print system information."""
        print("💻 System Information:")
        print(f"  Platform: {platform.platform()}")
        print(f"  Python: {platform.python_version()}")
        print(f"  PyTorch: {torch.__version__}")

        if self.device == 'cuda':
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA: {torch.version.cuda}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        print(f"  Library: {__version__}")
        print()

    def generate_report(self, filename: str = None):
        """Generate detailed benchmark report."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_report_{timestamp}"

        # Create summary statistics
        df = pd.DataFrame(self.results)

        if len(df) == 0:
            print("⚠️ No benchmark results to report")
            return

        # Filter successful results for statistics
        successful = df[df['successful']]

        summary = {
            'timestamp': datetime.now().isoformat(),
            'system_info': {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'pytorch_version': torch.__version__,
                'library_version': __version__,
                'device': self.device
            },
            'benchmark_config': {
                'warmup_iterations': self.warmup_iterations,
                'benchmark_iterations': self.benchmark_iterations,
            },
            'results_summary': {
                'total_tests': len(df),
                'successful_tests': len(successful),
                'avg_speedup': successful['speedup'].mean() if len(successful) > 0 else 0,
                'median_speedup': successful['speedup'].median() if len(successful) > 0 else 0,
                'min_speedup': successful['speedup'].min() if len(successful) > 0 else 0,
                'max_speedup': successful['speedup'].max() if len(successful) > 0 else 0,
                'avg_memory_diff_mb': successful['memory_diff_mb'].mean() if len(successful) > 0 else 0,
            },
            'detailed_results': self.results
        }

        if self.device == 'cuda':
            summary['system_info'].update({
                'gpu_name': torch.cuda.get_device_name(0),
                'cuda_version': torch.version.cuda,
                'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9
            })

        # Save JSON report
        with open(f"{filename}.json", 'w') as f:
            json.dump(summary, f, indent=2)

        # Save CSV data
        df.to_csv(f"{filename}.csv", index=False)

        print(f"📊 Report saved:")
        print(f"  • {filename}.json (summary)")
        print(f"  • {filename}.csv (detailed data)")

        return summary

    def plot_results(self, filename: str = None):
        """Generate visualization plots."""
        if not self.results:
            print("⚠️ No results to plot")
            return

        df = pd.DataFrame(self.results)
        successful = df[df['successful']]

        if len(successful) == 0:
            print("⚠️ No successful results to plot")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Speedup distribution
        ax1.hist(successful['speedup'], bins=20, alpha=0.7, color='green', edgecolor='black')
        ax1.set_xlabel('Speedup (x)')
        ax1.set_ylabel('Number of Operations')
        ax1.set_title('Speedup Distribution')
        ax1.axvline(successful['speedup'].mean(), color='red', linestyle='--',
                   label=f'Mean: {successful["speedup"].mean():.2f}x')
        ax1.legend()

        # Performance by operation type
        operation_types = successful['name'].str.extract('([a-z_]+)')[0]
        speedup_by_type = successful.groupby(operation_types)['speedup'].mean()

        bars = ax2.bar(speedup_by_type.index, speedup_by_type.values, color='skyblue', edgecolor='black')
        ax2.set_xlabel('Operation Type')
        ax2.set_ylabel('Average Speedup (x)')
        ax2.set_title('Average Speedup by Operation Type')
        ax2.tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar, value in zip(bars, speedup_by_type.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{value:.2f}x', ha='center', va='bottom')

        # Memory usage comparison
        ax3.scatter(successful['memory_standard_mb'], successful['memory_invariant_mb'], alpha=0.6)
        max_mem = max(successful['memory_standard_mb'].max(), successful['memory_invariant_mb'].max())
        ax3.plot([0, max_mem], [0, max_mem], 'r--', alpha=0.7, label='Equal memory')
        ax3.set_xlabel('Standard PyTorch Memory (MB)')
        ax3.set_ylabel('Batch-Invariant Memory (MB)')
        ax3.set_title('Memory Usage Comparison')
        ax3.legend()

        # Execution time comparison
        ax4.scatter(successful['time_standard_ms'], successful['time_invariant_ms'], alpha=0.6)
        max_time = max(successful['time_standard_ms'].max(), successful['time_invariant_ms'].max())
        ax4.plot([0, max_time], [0, max_time], 'r--', alpha=0.7, label='Equal time')
        ax4.set_xlabel('Standard PyTorch Time (ms)')
        ax4.set_ylabel('Batch-Invariant Time (ms)')
        ax4.set_title('Execution Time Comparison')
        ax4.legend()

        plt.tight_layout()

        if filename:
            plt.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')
            print(f"📈 Plots saved to {filename}.png")

        plt.show()


def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(description="Batch Invariant Operations Benchmark Suite")
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'],
                       help='Device to run benchmarks on')
    parser.add_argument('--warmup', type=int, default=5,
                       help='Number of warmup iterations')
    parser.add_argument('--iterations', type=int, default=50,
                       help='Number of benchmark iterations')
    parser.add_argument('--output', type=str, default=None,
                       help='Output filename prefix (default: timestamp)')
    parser.add_argument('--plot', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--quick', action='store_true',
                       help='Run a quick benchmark with fewer tests')

    args = parser.parse_args()

    if not LIBRARY_AVAILABLE:
        print("❌ batch-invariant-ops library not available")
        print("Install with: pip install -e .")
        return 1

    try:
        # Initialize benchmark suite
        suite = BenchmarkSuite(
            device=args.device,
            warmup_iterations=args.warmup,
            benchmark_iterations=args.iterations
        )

        # Run benchmarks
        if args.quick:
            print("⚡ Running quick benchmark...")
            suite.benchmark_matrix_operations()
        else:
            suite.run_comprehensive_benchmark()

        # Generate report
        summary = suite.generate_report(args.output)

        # Print summary
        print(f"\n📊 Benchmark Summary:")
        print(f"  Tests: {summary['results_summary']['successful_tests']}/{summary['results_summary']['total_tests']} successful")
        print(f"  Average speedup: {summary['results_summary']['avg_speedup']:.2f}x")
        print(f"  Median speedup: {summary['results_summary']['median_speedup']:.2f}x")
        print(f"  Best speedup: {summary['results_summary']['max_speedup']:.2f}x")

        # Generate plots
        if args.plot:
            suite.plot_results(args.output)

        return 0

    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
#!/usr/bin/env python3
"""
Comprehensive GPU and CUDA verification for batch-invariant-ops.
"""

import sys
import subprocess
import json
from datetime import datetime


def check_nvidia_driver():
    """Check NVIDIA driver availability."""
    print("🔌 Checking NVIDIA Driver...")
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                               capture_output=True, text=True, check=True)
        driver_version = result.stdout.strip()
        print(f"  ✅ NVIDIA Driver: {driver_version}")
        return True, driver_version
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  ❌ nvidia-smi not found or failed")
        return False, None


def check_gpu_specs():
    """Get detailed GPU specifications."""
    print("🎮 Checking GPU Specifications...")
    try:
        # Get GPU info
        result = subprocess.run([
            'nvidia-smi',
            '--query-gpu=name,memory.total,memory.free,compute_cap,power.limit',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, check=True)

        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 5:
                    gpu_info = {
                        'name': parts[0],
                        'memory_total_mb': int(parts[1]),
                        'memory_free_mb': int(parts[2]),
                        'compute_capability': parts[3],
                        'power_limit_w': float(parts[4])
                    }
                    gpus.append(gpu_info)
                    print(f"  ✅ {gpu_info['name']}")
                    print(f"    Memory: {gpu_info['memory_total_mb']/1024:.1f} GB total, {gpu_info['memory_free_mb']/1024:.1f} GB free")
                    print(f"    Compute Capability: {gpu_info['compute_capability']}")
                    print(f"    Power Limit: {gpu_info['power_limit_w']} W")

        return True, gpus
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  ❌ Failed to get GPU specifications")
        return False, []


def check_cuda_version():
    """Check CUDA version from nvidia-smi."""
    print("🚀 Checking CUDA Version...")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True)
        # Extract CUDA version from nvidia-smi output
        for line in result.stdout.split('\n'):
            if 'CUDA Version:' in line:
                cuda_version = line.split('CUDA Version:')[1].strip().split()[0]
                print(f"  ✅ CUDA Version: {cuda_version}")
                return True, cuda_version
        print("  ⚠️  CUDA version not found in nvidia-smi output")
        return False, None
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  ❌ Failed to check CUDA version")
        return False, None


def check_pytorch():
    """Check PyTorch installation and CUDA support."""
    print("🔥 Checking PyTorch...")
    try:
        import torch
        print(f"  ✅ PyTorch version: {torch.__version__}")
        print(f"  ✅ CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"  ✅ CUDA version (PyTorch): {torch.version.cuda}")
            print(f"  ✅ CuDNN version: {torch.backends.cudnn.version()}")

            device_count = torch.cuda.device_count()
            print(f"  ✅ GPU devices detected: {device_count}")

            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                print(f"    Device {i}: {props.name}")
                print(f"      Memory: {props.total_memory / 1e9:.1f} GB")
                print(f"      Compute Capability: {props.major}.{props.minor}")
                print(f"      Multiprocessors: {props.multi_processor_count}")

            return True, {
                'version': torch.__version__,
                'cuda_available': True,
                'cuda_version': torch.version.cuda,
                'cudnn_version': torch.backends.cudnn.version(),
                'device_count': device_count
            }
        else:
            print("  ❌ CUDA not available in PyTorch")
            return False, {
                'version': torch.__version__,
                'cuda_available': False
            }
    except ImportError:
        print("  ❌ PyTorch not installed")
        return False, None


def check_triton():
    """Check Triton installation."""
    print("⚡ Checking Triton...")
    try:
        import triton
        print(f"  ✅ Triton version: {triton.__version__}")
        return True, triton.__version__
    except ImportError:
        print("  ❌ Triton not installed")
        return False, None


def check_batch_invariant_ops():
    """Check batch-invariant-ops installation."""
    print("🔧 Checking batch-invariant-ops...")
    try:
        import batch_invariant_ops
        from batch_invariant_ops import set_batch_invariant_mode

        print(f"  ✅ Library version: {batch_invariant_ops.__version__}")

        # Test basic functionality
        print("  🧪 Testing basic functionality...")
        import torch

        if torch.cuda.is_available():
            torch.set_default_device('cuda')

            # Test batch-invariant mode
            with set_batch_invariant_mode(True):
                a = torch.randn(4, 4)
                b = torch.randn(4, 4)
                c = torch.mm(a, b)
                print(f"    ✅ Matrix multiplication test passed")

            print(f"    ✅ Basic functionality verified")
            return True, batch_invariant_ops.__version__
        else:
            print("    ⚠️  Skipping functionality test (no CUDA)")
            return True, batch_invariant_ops.__version__

    except ImportError as e:
        print(f"  ❌ batch-invariant-ops not installed: {e}")
        return False, None
    except Exception as e:
        print(f"  ❌ Error testing functionality: {e}")
        return False, None


def test_memory_allocation():
    """Test GPU memory allocation."""
    print("💾 Testing Memory Allocation...")
    try:
        import torch
        if not torch.cuda.is_available():
            print("  ⚠️  Skipping (no CUDA)")
            return True, None

        # Test allocating tensors of different sizes
        sizes = [
            (100, 100),
            (1000, 1000),
            (2000, 2000)
        ]

        for size in sizes:
            try:
                tensor = torch.randn(size, device='cuda')
                memory_used = torch.cuda.memory_allocated() / 1e6  # MB
                print(f"  ✅ Allocated {size[0]}x{size[1]} tensor ({memory_used:.1f} MB used)")
                del tensor
                torch.cuda.empty_cache()
            except RuntimeError as e:
                print(f"  ⚠️  Failed to allocate {size[0]}x{size[1]} tensor: {e}")

        return True, "Memory allocation tests completed"

    except Exception as e:
        print(f"  ❌ Memory allocation test failed: {e}")
        return False, str(e)


def benchmark_operations():
    """Quick benchmark of operations."""
    print("🏎️ Quick Performance Benchmark...")
    try:
        import torch
        import time
        from batch_invariant_ops import set_batch_invariant_mode

        if not torch.cuda.is_available():
            print("  ⚠️  Skipping (no CUDA)")
            return True, None

        torch.set_default_device('cuda')

        # Small benchmark
        a = torch.randn(1024, 1024)
        b = torch.randn(1024, 1024)

        # Warmup
        for _ in range(5):
            _ = torch.mm(a, b)
        torch.cuda.synchronize()

        # Benchmark standard PyTorch
        start = time.time()
        for _ in range(10):
            _ = torch.mm(a, b)
        torch.cuda.synchronize()
        time_standard = (time.time() - start) / 10

        # Benchmark batch-invariant
        with set_batch_invariant_mode(True):
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(10):
                _ = torch.mm(a, b)
            torch.cuda.synchronize()
            time_invariant = (time.time() - start) / 10

        print(f"  ✅ Standard PyTorch: {time_standard*1000:.2f} ms")
        print(f"  ✅ Batch-invariant:  {time_invariant*1000:.2f} ms")
        print(f"  ✅ Speedup: {time_standard/time_invariant:.2f}x")

        return True, {
            'standard_ms': time_standard * 1000,
            'invariant_ms': time_invariant * 1000,
            'speedup': time_standard / time_invariant
        }

    except Exception as e:
        print(f"  ❌ Benchmark failed: {e}")
        return False, str(e)


def generate_report():
    """Generate comprehensive verification report."""
    print("\n📋 Generating Verification Report...")

    report = {
        'timestamp': datetime.now().isoformat(),
        'checks': {}
    }

    # Run all checks
    checks = [
        ('nvidia_driver', check_nvidia_driver),
        ('gpu_specs', check_gpu_specs),
        ('cuda_version', check_cuda_version),
        ('pytorch', check_pytorch),
        ('triton', check_triton),
        ('batch_invariant_ops', check_batch_invariant_ops),
        ('memory_allocation', test_memory_allocation),
        ('benchmark', benchmark_operations)
    ]

    for check_name, check_func in checks:
        try:
            success, data = check_func()
            report['checks'][check_name] = {
                'success': success,
                'data': data
            }
        except Exception as e:
            report['checks'][check_name] = {
                'success': False,
                'error': str(e)
            }

    # Save report
    with open('gpu_verification_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print("  ✅ Report saved to gpu_verification_report.json")
    return report


def main():
    """Main verification function."""
    print("🔍 GPU and CUDA Verification for batch-invariant-ops")
    print("=" * 60)
    print()

    report = generate_report()

    # Summary
    print("\n📊 VERIFICATION SUMMARY")
    print("=" * 30)

    total_checks = len(report['checks'])
    passed_checks = sum(1 for check in report['checks'].values() if check['success'])

    print(f"Checks passed: {passed_checks}/{total_checks}")

    if passed_checks == total_checks:
        print("🎉 All checks passed! Your environment is ready for batch-invariant-ops.")
        return 0
    else:
        print("⚠️  Some checks failed. Please review the output above.")
        failed_checks = [name for name, check in report['checks'].items() if not check['success']]
        print(f"Failed checks: {', '.join(failed_checks)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
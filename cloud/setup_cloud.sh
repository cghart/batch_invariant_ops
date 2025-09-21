#!/bin/bash
set -e

# Batch Invariant Ops - Universal Cloud Setup Script
# Works on: Colab, Paperspace, Vast.ai, AWS, GCP, Azure, Lambda Labs, RunPod

echo "🚀 Batch Invariant Ops - Cloud Setup"
echo "====================================="

# Detect platform
detect_platform() {
    if [ -n "$COLAB_GPU" ] || [ -n "$COLAB_TPU_ADDR" ]; then
        echo "🔍 Detected: Google Colab"
        export PLATFORM="colab"
    elif [ -n "$PAPERSPACE_METRIC_HOST" ]; then
        echo "🔍 Detected: Paperspace"
        export PLATFORM="paperspace"
    elif [ -n "$VAST_CONTAINERD" ]; then
        echo "🔍 Detected: Vast.ai"
        export PLATFORM="vast"
    elif [ -n "$RUNPOD_POD_ID" ]; then
        echo "🔍 Detected: RunPod"
        export PLATFORM="runpod"
    elif grep -q "Amazon" /sys/hypervisor/uuid 2>/dev/null; then
        echo "🔍 Detected: AWS"
        export PLATFORM="aws"
    elif dmidecode -s system-manufacturer 2>/dev/null | grep -q "Google"; then
        echo "🔍 Detected: Google Cloud Platform"
        export PLATFORM="gcp"
    elif dmidecode -s system-manufacturer 2>/dev/null | grep -q "Microsoft"; then
        echo "🔍 Detected: Microsoft Azure"
        export PLATFORM="azure"
    else
        echo "🔍 Detected: Generic Linux"
        export PLATFORM="generic"
    fi
}

# Check for GPU
check_gpu() {
    echo "🎮 Checking GPU availability..."
    if command -v nvidia-smi > /dev/null 2>&1; then
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | while read -r gpu_info; do
            gpu_name=$(echo "$gpu_info" | cut -d',' -f1 | xargs)
            gpu_memory=$(echo "$gpu_info" | cut -d',' -f2 | xargs)
            echo "  ✅ $gpu_name ($(echo "scale=1; $gpu_memory/1024" | bc)GB)"
        done
    else
        echo "  ❌ No NVIDIA GPU detected!"
        echo "  Please ensure you're using a GPU-enabled instance."
        exit 1
    fi
}

# Check Python version
check_python() {
    echo "🐍 Checking Python..."
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    echo "  ✅ Python $python_version"

    if ! command -v pip3 > /dev/null 2>&1; then
        echo "  📦 Installing pip..."
        curl https://bootstrap.pypa.io/get-pip.py | python3
    fi
}

# Install PyTorch with CUDA support
install_pytorch() {
    echo "🔥 Installing PyTorch with CUDA..."

    # Detect CUDA version
    if command -v nvidia-smi > /dev/null 2>&1; then
        cuda_version=$(nvidia-smi | grep "CUDA Version" | grep -oP '\d+\.\d+' | head -1)
        echo "  🔧 Detected CUDA $cuda_version"

        # Map CUDA version to PyTorch index
        case "$cuda_version" in
            "11.8") torch_index="cu118" ;;
            "12.1") torch_index="cu121" ;;
            "12.4") torch_index="cu124" ;;
            *) torch_index="cu118" ;; # Default fallback
        esac
    else
        torch_index="cu118" # Default
    fi

    # Check if PyTorch is already installed with CUDA
    if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        pytorch_version=$(python3 -c "import torch; print(torch.__version__)")
        echo "  ✅ PyTorch $pytorch_version already installed with CUDA"
    else
        echo "  📦 Installing PyTorch with CUDA support..."
        pip3 install torch>=2.1.0 --index-url https://download.pytorch.org/whl/$torch_index
    fi
}

# Install Triton
install_triton() {
    echo "⚡ Installing Triton..."
    if python3 -c "import triton" 2>/dev/null; then
        triton_version=$(python3 -c "import triton; print(triton.__version__)")
        echo "  ✅ Triton $triton_version already installed"
    else
        echo "  📦 Installing Triton..."
        pip3 install triton
    fi
}

# Install the library
install_library() {
    echo "🔧 Installing batch-invariant-ops..."

    if [ ! -d "batch_invariant_ops" ]; then
        echo "  📥 Cloning repository..."
        git clone https://github.com/cghart/batch_invariant_ops.git
        cd batch_invariant_ops
    fi

    echo "  📦 Installing in development mode..."
    pip3 install -e .
}

# Verify installation
verify_installation() {
    echo "🧪 Verifying installation..."

    python3 -c "
import sys
import torch
from batch_invariant_ops import set_batch_invariant_mode, __version__

print(f'  ✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')
print(f'  ✅ PyTorch {torch.__version__}')
print(f'  ✅ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  ✅ GPU: {torch.cuda.get_device_name(0)}')
print(f'  ✅ batch-invariant-ops {__version__}')

# Quick functionality test
torch.set_default_device('cuda')
with set_batch_invariant_mode(True):
    a = torch.randn(2, 2)
    b = torch.randn(2, 2)
    c = torch.mm(a, b)
    print(f'  ✅ Batch-invariant mode functional')
"
}

# Run tests
run_tests() {
    echo "🧪 Running tests..."
    if [ -f "test_batch_invariance.py" ]; then
        python3 test_batch_invariance.py
    else
        echo "  ⚠️  Test file not found, skipping tests"
    fi
}

# Platform-specific optimizations
platform_optimizations() {
    case "$PLATFORM" in
        "colab")
            echo "🎯 Applying Colab optimizations..."
            # Mount Google Drive if needed
            if [ ! -d "/content/drive" ]; then
                echo "  💾 Consider mounting Google Drive for persistence"
            fi
            ;;
        "paperspace")
            echo "🎯 Applying Paperspace optimizations..."
            # Set up storage optimizations
            export TRITON_CACHE_DIR="/storage/.triton_cache"
            mkdir -p "$TRITON_CACHE_DIR"
            ;;
        "vast")
            echo "🎯 Applying Vast.ai optimizations..."
            # Optimize for container environment
            export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
            ;;
        *)
            echo "🎯 Applying generic optimizations..."
            ;;
    esac
}

# Create environment info file
create_env_info() {
    echo "📋 Creating environment info..."
    python3 -c "
import json
import torch
import platform
from datetime import datetime
from batch_invariant_ops import __version__

info = {
    'timestamp': datetime.now().isoformat(),
    'platform': '$PLATFORM',
    'system': {
        'os': platform.system(),
        'arch': platform.machine(),
        'python_version': platform.python_version(),
    },
    'gpu': {
        'available': torch.cuda.is_available(),
        'name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else None,
    },
    'versions': {
        'pytorch': torch.__version__,
        'cuda': torch.version.cuda,
        'batch_invariant_ops': __version__,
    }
}

with open('cloud_env_info.json', 'w') as f:
    json.dump(info, f, indent=2)

print('  ✅ Environment info saved to cloud_env_info.json')
"
}

# Main execution
main() {
    detect_platform
    check_gpu
    check_python
    install_pytorch
    install_triton
    install_library
    platform_optimizations
    verify_installation
    run_tests
    create_env_info

    echo ""
    echo "🎉 Setup complete!"
    echo "========================"
    echo ""
    echo "📖 Next steps:"
    echo "  • Import with: from batch_invariant_ops import set_batch_invariant_mode"
    echo "  • Use with: with set_batch_invariant_mode(True): ..."
    echo "  • Check notebooks/ for examples"
    echo "  • Run python3 test_batch_invariance.py for verification"
    echo ""
    echo "🚀 Happy deterministic computing!"
}

# Run if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
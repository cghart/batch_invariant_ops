# Batch Invariant Ops

A companion library release to https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/. This library contains some batch-invariant kernels as well as an example of achieving deterministic vLLM inference.

## Overview

This library primarily leverages torch.Library to sub out existing PyTorch kernels with "batch-invariant" ones. This allows many existing PyTorch models to use the batch-invariant ops with low overhead and non-intrusive code changes.

## Installation

### 🚀 Quick Start with Google Colab (Recommended)

The easiest way to get started is with Google Colab, which provides free GPU access:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cghart/batch_invariant_ops/blob/main/notebooks/batch_invariant_ops_colab.ipynb)

**Quick setup in Colab:**
1. Click the badge above (or create a new Colab notebook)
2. Change runtime to GPU: `Runtime → Change runtime type → Hardware accelerator: GPU`
3. Run this cell:
```python
!git clone https://github.com/cghart/batch_invariant_ops
%cd batch_invariant_ops
!bash cloud/setup_cloud.sh
```

### 💻 Local Installation (Requires NVIDIA GPU)

**Prerequisites:**
- NVIDIA GPU with CUDA capability
- CUDA Toolkit 11.8+ or 12.x
- Python 3.8-3.13

**Installation steps:**
```bash
# 1. Install PyTorch with CUDA support
pip install torch>=2.1.0 --index-url https://download.pytorch.org/whl/cu118

# 2. Install Triton
pip install triton

# 3. Clone and install this library
git clone https://github.com/cghart/batch_invariant_ops.git
cd batch_invariant_ops
pip install -e .

# 4. Verify installation
python cloud/verify_gpu.py
```

### ☁️ Cloud GPU Platforms

For longer experiments or better GPUs:

| Platform | Cost | Best For | Setup |
|----------|------|----------|--------|
| **Google Colab** | Free - $10/mo | Quick testing, demos | One-click notebooks |
| **Paperspace** | $0.45-3/hr | Development, training | `bash cloud/setup_cloud.sh` |
| **Vast.ai** | $0.20-2/hr | Cost optimization | Docker deployment |
| **Lambda Labs** | $1.10-2/hr | Production workloads | SSH access |

See [cloud/alternative_platforms.md](cloud/alternative_platforms.md) for detailed setup instructions.

## Quick Start

```python
import torch
from batch_invariant_ops import set_batch_invariant_mode

# Enable batch-invariant mode
with set_batch_invariant_mode():
    # Your inference code here
    model = YourModel()
    output = model(input_tensor)
```

## Testing Batch-Invariance

The following example shows how batch size can affect results in standard PyTorch:

```python
import torch
from batch_invariant_ops import set_batch_invariant_mode
torch.set_default_device('cuda')

# Just to get the logging out of the way haha
with set_batch_invariant_mode(True):
    pass

def test_batch_invariance():
    B, D = 2048, 4096
    a = torch.linspace(-100, 100, B*D).reshape(B, D)
    b = torch.linspace(-100, 100, D*D).reshape(D, D)
    
    # Method 1: Matrix-vector multiplication (batch size 1)
    out1 = torch.mm(a[:1], b)
    
    # Method 2: Matrix-matrix multiplication, then slice (full batch)
    out2 = torch.mm(a, b)[:1]
    
    # Check if results are identical
    diff = (out1 - out2).abs().max()
    print(f"Difference: {diff.item()}")
    return diff.item() == 0

# Test with standard PyTorch (likely to show differences)
print("Standard PyTorch:")
with set_batch_invariant_mode(False):
    is_deterministic = test_batch_invariance()
    print(f"Deterministic: {is_deterministic}")

# Test with batch-invariant operations
print("\nBatch-Invariant Mode:")
with set_batch_invariant_mode(True):
    is_deterministic = test_batch_invariance()
    print(f"Deterministic: {is_deterministic}")

```

## Deterministic Inference in vLLM
`deterministic_vllm_inference.py` shows an proof of concept of validating that vLLM can be made deterministic with a minor upstream PR to use this library. Without the upstream PR, we see that out of 1000 random length 100 completions we see 18 unique samples. After the upstream PR, there is only one unique sample.

## Supported Operations

### Matrix Operations
- `torch.mm()` - Matrix multiplication
- `torch.addmm()` - Matrix multiplication with bias addition

### Activation Functions
- `torch.log_softmax()` - Log-softmax activation

### Reduction Operations
- `torch.mean()` - Mean computation along specified dimensions

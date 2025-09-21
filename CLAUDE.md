# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development
- `pip install -e .` - Install package in development mode
- `python test_batch_invariance.py` - Run batch invariance test comparing standard PyTorch vs batch-invariant operations
- `python deterministic_vllm_inference.py` - Test deterministic inference with vLLM (requires vLLM server running)

### Code Quality (Development Dependencies)
- `black .` - Format code with Black (line length 100)
- `isort .` - Sort imports
- `flake8` - Run linting
- `pytest` - Run tests (when available)

## Architecture

### Core Components

**batch_invariant_ops.py** - Main implementation containing:
- Triton kernels for batch-invariant operations (matmul, log_softmax, mean)
- PyTorch library overrides using `torch.library.Library`
- Context manager `set_batch_invariant_mode()` for enabling/disabling batch-invariant behavior

### Key Design Patterns

**Kernel Replacement Strategy**: Uses `torch.library.Library` to override CUDA implementations of specific PyTorch operations (`mm`, `addmm`, `_log_softmax`, `mean.dim`) with Triton-based batch-invariant versions.

**Context Management**: The library provides both context manager (`set_batch_invariant_mode()`) and explicit enable/disable functions for controlling when batch-invariant operations are active.

**Triton Implementation**: All custom kernels use Triton for GPU acceleration with deterministic execution patterns that ensure batch size doesn't affect individual element computations.

### Supported Operations
- Matrix multiplication (`torch.mm`, `torch.addmm`) - Uses persistent matmul kernel
- Log softmax (`torch.log_softmax`) - Custom implementation along last dimension
- Mean reduction (`torch.mean`) - Supports single and multiple dimension reduction

### Testing Approach
The library validates batch invariance by comparing:
1. Small batch computation (e.g., single row matrix multiplication)
2. Large batch computation with slicing (e.g., full matrix multiplication, then slice)

These should produce identical results with batch-invariant mode enabled, but may differ with standard PyTorch due to floating-point precision and execution order differences.
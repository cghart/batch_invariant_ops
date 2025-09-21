# Alternative Cloud GPU Platforms

This guide covers cloud GPU platforms beyond Google Colab for running batch-invariant-ops when you need more resources, longer session times, or better GPUs.

## 📊 Platform Comparison

| Platform | Cost/Hour | Free Tier | Session Limit | Best GPU | Setup Difficulty | Best For |
|----------|-----------|-----------|---------------|----------|------------------|----------|
| **Google Colab** | Free - $10/mo | Yes (T4) | ~12h | A100 (Pro+) | ⭐ Easy | Quick testing |
| **Paperspace** | $0.45 - $3.09 | $10 credit | Unlimited | A100-80GB | ⭐⭐ Easy | Development |
| **Vast.ai** | $0.20 - $2.00 | No | Unlimited | RTX 4090 | ⭐⭐⭐ Medium | Cost optimization |
| **Lambda Labs** | $1.10 - $2.00 | No | Unlimited | A100-80GB | ⭐⭐ Easy | Production |
| **RunPod** | $0.39 - $2.89 | $10 credit | Unlimited | H100 | ⭐⭐ Easy | Serverless |
| **AWS EC2** | $3.06 - $32.77 | $300 credit | Unlimited | H100 | ⭐⭐⭐⭐ Hard | Enterprise |
| **GCP** | $2.48 - $33.25 | $300 credit | Unlimited | A100-80GB | ⭐⭐⭐⭐ Hard | Enterprise |

## 🚀 Quick Setup Guide

### 1. Paperspace Gradient

**Best for:** Development, training, persistent storage

**Setup:**
```bash
# Option 1: Create new notebook
# 1. Go to https://gradient.paperspace.com
# 2. Create account (get $10 free credit)
# 3. Create new notebook → PyTorch template
# 4. Select GPU (P5000, V100, A100)
# 5. Run setup in first cell:

!git clone https://github.com/cghart/batch_invariant_ops
%cd batch_invariant_ops
!bash cloud/setup_cloud.sh

# Option 2: Use our pre-configured environment
# Upload this repository and run the setup script
```

**Pros:**
- Persistent storage (notebooks and data saved)
- Good GPU selection
- Jupyter Lab interface
- Team collaboration features

**Cons:**
- Costs money (but reasonable)
- Can be slow to start instances

**Cost optimization:**
- Use auto-shutdown features
- Start with P5000 GPU ($0.45/hr) for development
- Upgrade to A100 only for heavy workloads

---

### 2. Vast.ai

**Best for:** Cost optimization, variety of GPU options

**Setup:**
```bash
# 1. Create account at https://vast.ai
# 2. Add credit to account
# 3. Search for instances with:
#    - CUDA 11.8 or 12.x
#    - PyTorch image (pytorch/pytorch:latest)
#    - RTX 3090/4090 or better

# 4. Once connected via SSH or Jupyter:
git clone https://github.com/cghart/batch_invariant_ops
cd batch_invariant_ops
bash cloud/setup_cloud.sh
```

**Docker Template:**
```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
WORKDIR /workspace
RUN apt-get update && apt-get install -y git
RUN git clone https://github.com/cghart/batch_invariant_ops
WORKDIR /workspace/batch_invariant_ops
RUN pip install triton && pip install -e .
CMD ["bash", "cloud/setup_cloud.sh"]
```

**Pros:**
- Very competitive pricing
- Wide variety of GPUs
- Market-based pricing
- Good for experiments

**Cons:**
- Variable reliability (community hosts)
- Setup can be more complex
- Instance availability varies

---

### 3. Lambda Labs

**Best for:** Professional ML infrastructure, production workloads

**Setup:**
```bash
# 1. Create account at https://lambdalabs.com
# 2. Launch instance (1x A10, 1x A100, etc.)
# 3. SSH into instance (comes with PyTorch pre-installed)

# 4. Run setup:
git clone https://github.com/cghart/batch_invariant_ops
cd batch_invariant_ops
bash cloud/setup_cloud.sh

# 5. Optional: Set up Jupyter
pip install jupyterlab
jupyter lab --ip=0.0.0.0 --port=8888 --allow-root
# Then access via SSH tunnel or Lambda's web interface
```

**Pros:**
- Excellent hardware (latest GPUs)
- Very fast instances
- ML-optimized software stack
- Professional support

**Cons:**
- Higher cost than marketplace options
- Limited instance availability during peak times

---

### 4. RunPod

**Best for:** Serverless workloads, auto-scaling

**Setup:**

**Option 1: Pods (persistent)**
```bash
# 1. Create account at https://runpod.io
# 2. Launch pod with PyTorch template
# 3. Connect via web terminal or Jupyter

git clone https://github.com/cghart/batch_invariant_ops
cd batch_invariant_ops
bash cloud/setup_cloud.sh
```

**Option 2: Serverless (auto-scaling)**
```python
# Create serverless endpoint with our code
# Upload batch_invariant_ops as a zip file
# Use their API to run inference
```

**Pros:**
- Serverless option for auto-scaling
- Good GPU variety including H100
- Web-based access
- Reasonable pricing

**Cons:**
- Interface can be less intuitive
- Serverless has cold start times

---

### 5. AWS EC2

**Best for:** Enterprise deployment, custom infrastructure

**Setup:**
```bash
# 1. Launch Deep Learning AMI (Ubuntu 20.04)
# 2. Choose p3.2xlarge or better
# 3. SSH into instance

# The Deep Learning AMI comes with PyTorch pre-installed
git clone https://github.com/cghart/batch_invariant_ops
cd batch_invariant_ops
bash cloud/setup_cloud.sh

# Optional: Set up secure Jupyter access
jupyter lab --generate-config
# Configure password and SSL certificate
# Set up security groups for port 8888
```

**Instance recommendations:**
- **p3.2xlarge** (V100, 16GB) - $3.06/hr - Good for development
- **p4d.24xlarge** (8x A100, 320GB) - $32.77/hr - Production workloads
- **g4dn.xlarge** (T4, 16GB) - $0.526/hr - Budget option

**Pros:**
- Massive scale available
- Integration with AWS services
- Spot instances for cost savings
- Enterprise features

**Cons:**
- Complex setup and pricing
- Requires AWS knowledge
- Can be expensive

---

### 6. Google Cloud Platform (GCP)

**Best for:** Integration with Google services, research credits

**Setup:**
```bash
# 1. Create VM with Deep Learning VM image
gcloud compute instances create batch-invariant-test \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-v100,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --maintenance-policy=TERMINATE

# 2. SSH into instance
gcloud compute ssh batch-invariant-test --zone=us-central1-a

# 3. Run setup
git clone https://github.com/cghart/batch_invariant_ops
cd batch_invariant_ops
bash cloud/setup_cloud.sh
```

**Pros:**
- Academic research credits available
- Deep Learning VM images
- Good integration with AI/ML services
- Preemptible instances for cost savings

**Cons:**
- Complex pricing and setup
- GPU availability can be limited
- Requires GCP knowledge

---

## 💡 Tips for All Platforms

### Cost Optimization
1. **Use smaller GPUs for development**: Start with T4/P5000, scale up for production
2. **Monitor usage**: Set up billing alerts and auto-shutdown
3. **Use spot/preemptible instances**: 60-90% cost savings
4. **Storage optimization**: Use cloud storage for data, local SSD for active work

### Performance Optimization
1. **Data loading**: Use cloud storage close to compute
2. **Batch size**: Optimize for your specific GPU memory
3. **Mixed precision**: Use torch.cuda.amp for better performance
4. **Persistent storage**: Keep compiled kernels cached

### Security Best Practices
1. **SSH keys**: Use key-based authentication
2. **VPN access**: For sensitive workloads
3. **Data encryption**: Encrypt data at rest and in transit
4. **Access logs**: Monitor who accesses your instances

## 🔄 Migration Between Platforms

### From Colab to Other Platforms
```python
# Save your work to Google Drive first
from google.colab import drive
drive.mount('/content/drive')

# Export notebooks and data
!cp -r /content/batch_invariant_ops /content/drive/MyDrive/
!cp *.json *.csv /content/drive/MyDrive/results/
```

### Universal Setup Script
Our `cloud/setup_cloud.sh` script automatically detects the platform and applies appropriate optimizations:

```bash
# Works on all platforms
curl -sSL https://raw.githubusercontent.com/cghart/batch_invariant_ops/main/cloud/setup_cloud.sh | bash
```

## 📞 Support and Troubleshooting

### Common Issues

1. **CUDA version mismatch**
   ```bash
   # Check versions
   nvidia-smi
   python -c "import torch; print(torch.version.cuda)"

   # Reinstall PyTorch with correct CUDA version
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Out of memory errors**
   ```python
   # Reduce batch size or use gradient checkpointing
   torch.cuda.empty_cache()

   # Monitor memory usage
   python cloud/verify_gpu.py
   ```

3. **Slow training**
   ```bash
   # Check GPU utilization
   nvidia-smi -l 1

   # Run benchmarks
   python cloud/benchmark_suite.py --quick
   ```

### Getting Help
- **GitHub Issues**: Report problems and get help
- **Platform Support**: Each platform has their own support channels
- **Community**: Join ML/AI communities for advice

## 🎯 Choosing the Right Platform

### For Beginners
**Start with**: Google Colab → Paperspace
- Free tier to learn
- Easy setup
- Good documentation

### For Cost-Conscious Users
**Recommended**: Vast.ai → RunPod
- Market pricing
- Variety of options
- Good performance per dollar

### For Production
**Recommended**: Lambda Labs → AWS/GCP
- Reliable infrastructure
- Professional support
- Enterprise features

### For Research
**Recommended**: GCP (research credits) → Lambda Labs
- Academic discounts
- Latest hardware
- Research-friendly policies

Remember: Start small, measure performance, then scale up based on your specific needs!
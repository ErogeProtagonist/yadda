# Yadda - Efficient Attention Research

Comparative study of 500M parameter transformers:
- **Hybrid SWA** (Gemma-style): Sliding Window + Global Attention
- **MLA** (DeepSeek-style): Multi-Head Latent Attention

## Quick Start

### Training on Lambda H100 Cluster
```bash
# MLA Model
torchrun --standalone --nproc_per_node=8 train_edge.py --model_type mla --compile

# Hybrid Model  
torchrun --standalone --nproc_per_node=8 train_edge.py --model_type hybrid --compile
```

### Inference on Local GPU (RTX 3090)
```bash
python inference.py --checkpoint log/mla_05000.pt --interactive
```

## Data Preparation
```bash
python edu_fineweb10B.py --output_dir /lambda/nfs/riddy/edu_fineweb10B
```

## Files
- `models/` - Core model implementations (config, attention, transformer)
- `train_edge.py` - DDP training script
- `inference.py` - Portable inference with benchmarking
- `edu_fineweb10B.py` - FineWeb-Edu tokenization

# Preprocessing Paradox in Vision Transformer-Based Micro-Expression Recognition

[![Paper](https://img.shields.io/badge/Paper-ICICyTA%202025-blue)](link-to-paper)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

Official implementation of "[Preprocessing Paradox in Vision Transformer-Based Micro-Expression Recognition](link)" (ICICyTA 2025).

**Muhammad Taufiq Al Fikri**, Kurniawan Nur Ramadhani  
*Telkom University, Indonesia*

---

## Abstract

> Micro-expression recognition faces significant challenges from subtle facial movements lasting 40-500 milliseconds and severely limited training data. This study systematically compares three Vision Transformer architectures—ViT, Swin Transformer, and PoolFormer—representing distinct token mixing strategies across dual preprocessing methodologies on the CASME II dataset.
> 
> **Key Findings:**
> - Discovered **preprocessing paradox**: systematic face-aware preprocessing degraded performance by 18.4-20.5%
> - **PoolFormer** achieved best overall performance (macro F1=0.4762) with 67.4% improvement through temporal aggregation
> - First comprehensive 7-category CASME II benchmark under 49.5:1 class imbalance

---

## Main Results

| Model | Preprocessing | Temporal Phase | Macro F1 | Δ from Baseline |
|-------|--------------|----------------|----------|-----------------|
| **PoolFormer-m36** | M1 (Raw) | MFS (10.25×) | **0.4762** | +67.4% |
| ViT-patch32 | M1 (Raw) | AF (1×) | 0.4235 | - |
| PoolFormer-m48 | M1 (Raw) | KFS (3×) | 0.3974 | - |

*Full results and statistical analysis available in the paper.*

---

## Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/taufiq-afk/iciciyta25-preprocessing-paradox-mer.git
cd iciciyta25-preprocessing-paradox-mer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dataset Preparation
```bash
# Download CASME II dataset (requires registration)
# Follow instructions at: http://casme.psych.ac.cn/casme/e2

# Preprocess data
python data/download.py --dataset casme2 --output data/raw
python data/preprocess_m1.py --input data/raw --output data/processed/m1
python data/preprocess_m2.py --input data/raw --output data/processed/m2
```

### Training
```bash
# Train PoolFormer-m36 with Multi-Frame Sampling (best config)
python training/train.py --config experiments/poolformer_mfs_m1.yaml

# Reproduce all experiments
bash scripts/run_all_experiments.sh
```

### Evaluation
```bash
# Evaluate trained model
python evaluation/evaluate.py --checkpoint results/checkpoints/poolformer_m36_mfs.pth \
                               --test-data data/processed/m1/test

# Generate paper figures
python scripts/generate_paper_figures.py
```

---

## Repository Structure
```
├── data/                  # Data preprocessing pipelines
├── models/                # Vision Transformer implementations
├── training/              # Training scripts & configs
├── evaluation/            # Evaluation & visualization
├── experiments/           # Experiment configurations
├── results/               # Outputs (figures, tables, checkpoints)
├── notebooks/             # Exploratory analysis
└── paper/                 # Published paper & supplementary
```

---

## Reproducing Paper Results

**System Requirements:**
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)
- 16GB RAM minimum
- NVIDIA GPU with 8GB+ VRAM (tested on L4)

**Reproduce main results (Table III in paper):**
```bash
# Run all configurations from paper
bash scripts/run_all_experiments.sh

# Expected runtime: ~24 hours on single L4 GPU
```

**Pre-trained weights:**  
Download from [Google Drive](link) or [HuggingFace](link)

---

## Citation

If you find this work useful, please cite:
```bibtex
@inproceedings{alfikri2025preprocessing,
  title={Preprocessing Paradox in Vision Transformer-Based Micro-Expression Recognition},
  author={Al Fikri, Muhammad Taufiq and Ramadhani, Kurniawan Nur},
  booktitle={International Conference on Information and Communication Technology and Applications (ICICyTA)},
  year={2025}
}
```

---

## Related Work

- **CASME II Dataset:** [Yan et al., 2014](https://doi.org/10.1371/journal.pone.0086041)
- **PoolFormer:** [Yu et al., CVPR 2022](https://arxiv.org/abs/2111.11418)
- **Vision Transformer:** [Dosovitskiy et al., ICLR 2021](https://arxiv.org/abs/2010.11929)

---

## Contact

Muhammad Taufiq Al Fikri  
Email: taufiqafk@student.telkomuniversity.ac.id  
LinkedIn: [linkedin.com/in/taufiq-afk](https://linkedin.com/in/taufiq-afk)  

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- CASME II dataset creators
- Telkom University for computational resources
- [Any funding/support you received]

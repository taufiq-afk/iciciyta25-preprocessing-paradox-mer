# Preprocessing Paradox in Vision Transformer-Based Micro-Expression Recognition

Official code archive for "Preprocessing Paradox in Vision Transformer-Based Micro-Expression Recognition" published at ICICyTA 2025.

**Authors:** Muhammad Taufiq Al Fikri, Kurniawan Nur Ramadhani  
**Affiliation:** School of Computing, Telkom University, Indonesia  
**Conference:** International Conference on Information and Communication Technology and Applications (ICICyTA) 2025  
**Paper:** [ICICyTA 2025 Proceedings](link-when-available)

---

## Abstract

Micro-expression recognition faces significant challenges from subtle facial movements lasting 40-500 milliseconds and severely limited training data. This study systematically compares three Vision Transformer architectures—ViT, Swin Transformer, and PoolFormer—representing distinct token mixing strategies across dual preprocessing methodologies on the CASME II dataset.

**Key Contributions:**

1. **Preprocessing Paradox Discovery:** Systematic face-aware preprocessing paradoxically degraded temporal performance by 18.4-20.5% compared to minimally processed raw images on small datasets (201 training samples). This challenges conventional computer vision assumptions about preprocessing optimization.

2. **Temporal Robustness Identification:** PoolFormer's attention-free architecture demonstrated unique temporal aggregation capability, achieving best overall performance (macro F1=0.4762) with 67.4% improvement as temporal density increased, contrasting sharply with ViT's 40.5% degradation.

3. **Comprehensive Benchmark:** First rigorous Vision Transformer comparison on full 7-category CASME II under extreme 49.5:1 class imbalance, avoiding common class-merging simplifications.

4. **Multi-Frame Sampling Strategy:** Developed adaptive temporal sampling achieving 10.25x training expansion through metadata-driven windowing around key frames.

---

## Main Results

### Best Performance per Temporal Aggregation Phase

| Temporal Phase | Model | Preprocessing | Macro F1 | 95% CI | Training Samples |
|----------------|-------|---------------|----------|---------|------------------|
| Apex Frame (1x) | ViT-patch32 | M1 (Raw) | 0.4235 | [0.22, 0.68] | 201 |
| Key Frame Sequence (3x) | PoolFormer-m48 | M1 (Raw) | 0.3974 | - | 603 |
| Multi-Frame Sampling (10.25x) | **PoolFormer-m36** | M1 (Raw) | **0.4762** | [0.26, 0.71] | 2,061 |

### Preprocessing Impact Analysis (M1 Raw vs M2 Face-Aware)

| Temporal Phase | M1 (Raw) F1 | M2 (Face) F1 | Performance Gap |
|----------------|-------------|--------------|-----------------|
| Apex Frame (1x) | 0.4235 | 0.4229 | -0.1% |
| Key Frame Seq (3x) | 0.3974 | 0.3241 | **-18.4%** |
| Multi-Frame (10.25x) | 0.4762 | 0.3785 | **-20.5%** |

**Observation:** Preprocessing degradation intensified with temporal expansion, revealing overfitting to preprocessing artifacts rather than robust expression features under severe data scarcity.

### Architecture-Specific Temporal Behaviors

| Model | Apex Frame | Key Frame Seq | Multi-Frame | Trajectory |
|-------|-----------|---------------|-------------|------------|
| ViT-patch32 | 0.4235 | 0.2520 | 0.3285 | Peak → Collapse → Partial Recovery |
| Swin Transformer | 0.4075 | 0.3820 | 0.3282 | Stable Moderate Performance |
| PoolFormer-m36 | 0.2844 | 0.3505 | **0.4762** | Consistent Upward (+67.4%) |

**Finding:** Attention-free PoolFormer uniquely benefited from temporal aggregation, while global attention (ViT) degraded severely with conflicting temporal signals.

### Per-Class Performance (PoolFormer-m36, Best Overall)

| Emotion | Test Samples | M1 F1 | M2 F1 | Class Difficulty |
|---------|-------------|-------|-------|------------------|
| Surprise | 3 | 0.800 | 0.500 | Most Recognizable |
| Repression | 3 | 0.667 | 0.400 | Mid-tier |
| Disgust | 7 | 0.533 | 0.667 | Mid-tier (M2 benefits) |
| Others | 10 | 0.571 | 0.455 | Mid-tier |
| Happiness | 4 | 0.286 | 0.250 | Most Challenging |
| Sadness | 1 | 0.000 | 0.000 | Insufficient Data |
| Fear | 0 | - | - | No Test Samples |

**Macro F1 (excluding Fear):** 0.4762 (M1), 0.3785 (M2)

---

## Repository Contents

This repository contains research artifacts for academic documentation and reference purposes.

### Paper

**paper/ICICIYTA2025_Final.pdf** - Published conference paper (6 pages)

**Citation:**
```bibtex
@inproceedings{alfikri2025preprocessing,
  title={Preprocessing Paradox in Vision Transformer-Based Micro-Expression Recognition},
  author={Al Fikri, Muhammad Taufiq and Ramadhani, Kurniawan Nur},
  booktitle={2025 International Conference on Information and Communication Technology and Applications (ICICyTA)},
  pages={1--6},
  year={2025},
  organization={IEEE}
}
```

### Notebooks

Experimental notebooks developed in Google Colab environment:

**Preprocessing:**
- `01_01_EDA_CASME2-AF.ipynb` - Methodology 1 (minimal preprocessing, 384x384 RGB)
- `01_04_EDA_CASME2-PREP.ipynb` - Methodology 2 (face-aware preprocessing, 224x224 grayscale, Dlib detection)

**Model Training - Apex Frame Phase:**
- `02_01_ViT_CASME2-AF.ipynb` - ViT-patch32 training (baseline, best single-frame: F1=0.4235)
- `02_02_SwinT_CASME2-AF.ipynb` - Swin Transformer-base training
- `02_05_PoolFormer_CASME2-AF.ipynb` - PoolFormer-m36/m48 training

**Temporal Aggregation:**
- `03_03_PoolFormer_CASME2-KFS.ipynb` - Key Frame Sequence (onset-apex-offset, 3x expansion)
- `04_03_PoolFormer_CASME2-MFS.ipynb` - Multi-Frame Sampling (adaptive windowing, 10.25x expansion)

### Figures

High-resolution figures from the published paper:
- Experimental framework illustration
- Preprocessing paradox visualization (quality vs performance)
- Temporal aggregation performance trajectories
- Confusion matrices for best models

---

## Dataset Information

**CASME II Dataset** (Chinese Academy of Sciences Micro-Expression Database II)

**Access:** Licensed dataset requiring formal approval  
**Application:** http://casme.psych.ac.cn/casme/e2  
**Processing Time:** Typically 1-2 weeks after proposal submission

**Dataset Characteristics:**
- 255 video sequences from 26 Chinese participants
- 7 emotion categories with 49.5:1 class imbalance
- 200 fps, 640x480 resolution
- Spontaneous micro-expressions (40-500ms duration)

**Class Distribution:**
- Others: 99 samples (38.8%)
- Disgust: 63 (24.7%)
- Happiness: 32 (12.5%)
- Repression: 27 (10.6%)
- Surprise: 25 (9.8%)
- Sadness: 7 (2.7%)
- Fear: 2 (0.8%)

**Experimental Splits (Stratified):**
- Training: 201 videos (78.8%)
- Validation: 26 videos (10.2%)
- Test: 28 videos (11.0%)

**Citation:**
```bibtex
@article{yan2014casme,
  title={CASME II: An improved spontaneous micro-expression database and the baseline evaluation},
  author={Yan, Wen-Jing and Li, Xiaobai and Wang, Su-Jing and Zhao, Guoying and Liu, Yong-Jin and Chen, Yu-Hsin and Fu, Xiaolan},
  journal={PloS one},
  volume={9},
  number={1},
  pages={e86041},
  year={2014},
  publisher={Public Library of Science}
}
```

---

## Experimental Configuration

### Model Architectures

**ViT-patch32** (Vision Transformer)
- Token mixing: Global self-attention
- Parameters: 88M
- Patch size: 32x32 (49 tokens at 224x224)
- Strength: Peak single-frame intensity capture
- Weakness: Temporal aggregation degradation (-40.5%)

**Swin Transformer-base**
- Token mixing: Hierarchical shifted window attention
- Parameters: 88M
- Strength: Multi-scale feature extraction
- Weakness: No specialized advantage identified

**PoolFormer-m36/m48** (Attention-free MetaFormer)
- Token mixing: Fixed spatial pooling (no learned attention)
- Parameters: 56M (m36), 73M (m48)
- Strength: Temporal robustness (+67.4% with aggregation)
- Finding: Simpler m36 outperformed m48 under data scarcity

### Training Details

**Optimization:**
- Optimizer: AdamW (lr=5e-5, weight_decay=0.001)
- Loss: Focal Loss (ViT, PoolFormer-m36) or Class-weighted CrossEntropy (Swin, PoolFormer-m48)
- Early stopping: Patience 3-5 epochs (validation macro F1)
- Max epochs: 50

**Regularization:**
- Drop path rate: 0.05-0.3 (architecture-specific)
- Dropout: 0.0-0.2
- ImageNet pre-training: All models initialized with transfer learning

**Hardware:**
- Google Colab Pro+ with NVIDIA L4 GPU (22GB VRAM)
- Batch sizes: 16-32 (M1), 4-16 (M2, resolution-constrained)

### Evaluation Metrics

**Primary Metric:** Macro F1-score (equal weight across 7 classes, excluding Fear from test set calculation)

**Statistical Validation:** Bootstrap 95% confidence intervals (1,000 resampling iterations with replacement)

**Secondary Metrics:** Per-class F1, confusion matrices, temporal trajectory analysis

---

## Important Notes

### Research Context

This repository serves as an **academic archive** for the published ICICyTA 2025 paper. The code is provided for:
- Transparency of methodology
- Academic reference and citation
- Understanding experimental implementation details

### Technical Limitations

**Not designed for direct reproduction:**
- Notebooks developed for Google Colab + Google Drive workflow
- Hardcoded paths specific to author's Drive organization
- CASME II dataset requires independent licensing (not included)
- Model checkpoints not provided (storage limitations, >500MB per checkpoint)

**Adaptation requirements for local execution:**
- Replace `/content/drive/MyDrive/Thesis_MER_Project/` paths
- Obtain CASME II dataset access independently
- Setup local CUDA environment matching specifications
- Adjust batch sizes for available VRAM

### Reproducibility Considerations

**Expected variance:** ±0.02-0.05 in macro F1 due to:
- Hardware differences (GPU architecture, CUDA version)
- Randomness in data augmentation despite fixed seeds
- Bootstrap resampling variability

**For reproduction attempts:**
1. Use identical dataset splits (provided in notebook documentation)
2. Match hyperparameter configurations exactly
3. Verify preprocessing methodology (M1 vs M2 specifications)
4. Account for hardware-dependent batch size adjustments

---

## Related Publications and Resources

### Vision Transformer Architectures

**Vision Transformer (ViT):**  
A. Dosovitskiy et al., "An image is worth 16x16 words: Transformers for image recognition at scale," ICLR, 2021.

**Swin Transformer:**  
Z. Liu et al., "Swin transformer: Hierarchical vision transformer using shifted windows," ICCV, 2021.

**PoolFormer (MetaFormer):**  
W. Yu et al., "MetaFormer is actually what you need for vision," CVPR, 2022.

### Micro-Expression Recognition

**Micro-Expression Overview:**  
H. Pan et al., "Review of micro-expression spotting and recognition in video sequences," Virtual Reality & Intelligent Hardware, 2021.

**Transformer-based MER:**  
X.-B. Nguyen et al., "Micron-BERT: BERT-based facial micro-expression recognition," CVPR, 2023.

**Hybrid Approaches:**  
Y. Zheng and E. Blasch, "Facial micro-expression recognition enhanced by score fusion and a hybrid model from convolutional LSTM and vision transformer," Sensors, 2023.

### Class Imbalance Handling

**Focal Loss:**  
X. Li et al., "Generalized focal loss: Towards efficient representation learning for dense object detection," TPAMI, 2022.

---

## Contact

**Muhammad Taufiq Al Fikri**  
Final-Year Undergraduate Student (Informatics Engineering)  
School of Computing, Telkom University  
Bandung, Indonesia

**Email:** taufiqafk@student.telkomuniversity.ac.id  
**LinkedIn:** [linkedin.com/in/taufiq-afk](https://linkedin.com/in/taufiq-afk)  
**GitHub:** [github.com/taufiq-afk](https://github.com/taufiq-afk)

**Research Interests:** Computer Vision, Medical AI, Micro-Expression Recognition, Vision Transformers

**For inquiries about:**
- **Paper content and methodology:** Email primary author
- **Code implementation details:** Open GitHub issue
- **CASME II dataset access:** Contact official maintainers at http://casme.psych.ac.cn
- **Collaboration opportunities:** Email or LinkedIn message

---

## Future Research Directions

Based on findings from this work, promising directions include:

1. **Cross-dataset validation:** Test preprocessing paradox on larger datasets (CAS(ME)³, SMIC, SAMM) to determine dataset-size thresholds where optimization transitions from harmful to beneficial

2. **Sequential modeling enhancement:** Integrate PoolFormer with LSTM/GRU modules to exploit its temporal robustness for frame-sequence modeling

3. **Hybrid architectures:** Combine PoolFormer's temporal aggregation strength with ViT's peak-frame intensity capture

4. **Multi-modal fusion:** Incorporate depth information (available in CAS(ME)³) to investigate whether additional modalities mitigate preprocessing paradox

5. **Extended journal publication:** Develop conference findings into comprehensive journal manuscript for IEEE Transactions on Affective Computing or Pattern Analysis and Machine Intelligence

---

## License

This project is released under the MIT License.

**Note:** CASME II dataset has separate licensing terms governed by the Chinese Academy of Sciences. Users must obtain independent access approval and comply with dataset usage agreements.

---

## Acknowledgments

This research was conducted at the School of Computing, Telkom University, Indonesia.

**Special thanks to:**
- Dr. Kurniawan Nur Ramadhani (research supervisor) for guidance and collaboration
- CASME II dataset creators at the Chinese Academy of Sciences for making the dataset available to research community
- Google Colab for providing computational infrastructure
- ICICyTA 2025 reviewers and editors for constructive feedback that improved the paper quality

**Conference Publication:**  
This work was accepted and presented at the International Conference on Information and Communication Technology and Applications (ICICyTA) 2025.

---

**Repository Status:** Active archive (maintained for academic reference and citation support)  
**Last Updated:** December 2025  
**Publication Status:** Published at ICICyTA 2025

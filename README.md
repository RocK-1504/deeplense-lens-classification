# DeepLense — Common Test I | GSoC 2026 | ML4Sci

## Multi-Class Gravitational Lens Classification

Classifying strong gravitational lensing images into three dark matter 
substructure categories using a fine-tuned ResNet18.

---

## Task

| Label | Class |
|-------|-------|
| 0 | No Substructure |
| 1 | Subhalo Substructure |
| 2 | Vortex Substructure |

---

## Approach

Fine-tuned **ResNet18** (ImageNet pretrained) with:
- Differential learning rates (backbone: 1e-4, head: 1e-3)
- Cosine annealing LR scheduler
- Random horizontal/vertical flip augmentation
- Global mean subtraction + per-sample z-score normalisation

---

## Results

| Metric | Value |
|--------|-------|
| Best Val AUC (macro) | 0.7585 |
| Val Accuracy | ~57.6% |

### ROC Curve

![CNN](results/roc_cnn.png)

---

## Dataset

DeepLense dataset — not included due to size.  
Available via ML4Sci: https://ml4sci.org

---

## Dependencies
```
torch, torchvision, scikit-learn, numpy, matplotlib
```

---

## Author

**Prathik M Nambiar**  
B.Tech. Computer Science and Engineering, PES University, Bangalore  
GSoC 2026 — ML4Sci | Common Test I

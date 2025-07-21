# 🔬 APC-Net: Adaptive Perturbation and Consistency Network for Semi-Supervised Medical Image Segmentation

We provide the official PyTorch implementation of our APC-Net, a semi-supervised medical image segmentation framework integrating:

- **PAPE**: Perception-guided Adaptive Perturbation Enhancement  
- **CRP**: Consistency-aware Region Propagation  
- **PPO**: Progressive Pseudo-label Optimization

> 📄 **The Semi-Supervised Medical Image Segmentation Method Based on Adaptive Perturbation Enhancement and Progressive Consistency Propagation**  
> *Accepted in [your venue, e.g., MICCAI/NeurIPS 2025]*

---

## 📈 Highlights

- ✅ Enhanced structure-aware perturbations (PAPE)  
- 🔁 Robust region consistency modeling (CRP)  
- 📉 Progressive pseudo-label refinement (PPO)  
- 💡 Outperforms state-of-the-art under 10–20% annotation

---

## 📦 Installation

```bash
git clone https://github.com/yourname/APC-Net.git
cd APC-Net
conda create -n apcnet python=3.10
conda activate apcnet
pip install -r requirements.txt
```

---

## 🩺 Datasets

- **BUSI** (Breast Ultrasound)  
- **ISIC** (Skin Lesion)  
- **DDTI** (Dermoscopic)

Please place your datasets as:

```
├── data
│   ├── BUSI
│   ├── ISIC
│   └── DDTI
```

---

## 🚀 Usage

### Semi-Supervised Training (APC-Net)

```bash
bash scripts/train_apc.sh <DATASET> <PERCENT_LABEL> <GPU>
```

Example:

```bash
bash scripts/train_apc.sh ISIC 20 0
```

### Supervised Baseline

```bash
bash scripts/train_baseline.sh ISIC 100 0
```

---

## 📊 Results

| Dataset | Method       | Dice (%) | Jaccard (%) |
|---------|--------------|----------|-------------|
| ISIC    | APC-Net (20%)| **76.37**| **68.09**   |
| BUSI    | APC-Net (20%)| **78.42**| **70.12**   |
| DDTI    | APC-Net (20%)| **73.10**| **64.88**   |

---

## 📌 Key Components

- `models/pape.py`: PG-MSPCNN-based perturbation generator  
- `modules/crp.py`: Region consistency propagation (APCM + RSAP)  
- `utils/ppo.py`: Progressive pseudo-label filtering strategy

---

## 🧪 Evaluation

```bash
bash scripts/eval.sh ISIC best_model.pth
```

---

## 📄 Citation

```bibtex
@article{apcnet2025,
  title={The Semi-Supervised Medical Image Segmentation Method Based on Adaptive Perturbation Enhancement and Progressive Consistency Propagation},
  author={Your Name and Collaborators},
  journal={Under Review/Conference},
  year={2025}
}
```

---

## 🙏 Acknowledgements

We build upon [UniMatch](https://github.com/LiheYoung/UniMatch), [SSL4MIS](https://github.com/HiLab-git/SSL4MIS), and [UA-MT](https://github.com/yulequan/UA-MT).
# ApneaVision 💤
ApneaVision uses deep learning for sleep apnea detection from spectrograms.

## Features
- Conditional GAN for generating underrepresented apnea samples
- Hybrid CNN + BiLSTM + Attention + ViT model
- Explainable AI (SHAP, Grad-CAM, LIME, LRP)
- PyTorch training pipeline

## Folder Structure
- `data/` - raw and processed datasets
- `src/` - core code (models, training, evaluation)
- `notebooks/` - experiments
- `results/` - logs & visualizations
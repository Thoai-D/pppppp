# Adversarial Robustness in Chest X-ray Pneumonia Classification

A research project investigating adversarial robustness of deep learning models for pneumonia detection from chest X-rays. Built for SUTD AI Course (50.021).

## Overview

This project train an existing ResNet-152 model to classify chest X-rays as NORMAL or PNEUMONIA, evaluates their vulnerability to FGSM adversarial attacks, and applies adversarial training to improve robustness. Grad-CAM visualizations provide interpretability into model attention.

## Project Structure

```
├── Phase1_Data_Preprocessing.ipynb          # Dataset loading, balancing, merging, splitting
├── Phase2_Workflow.ipynb                    # Baseline training + evaluation pipeline
├── Phase3_FGSMAttack.ipynb                  # FGSM robustness evaluation
├── Phase4_AdvserialTraining.ipynb           # Adversarial training experiments
├── CNN_PYTORCH_CONVERTED.ipynb             # ResNet-152 training (PyTorch)
├── CNNADVERSARIAL_PYTORCH_COMBINED.ipynb   # Combined adversarial training pipeline
├── fgsm_export.py                           # Generate and export adversarial examples
├── gradcam_resnet152.py                     # Grad-CAM visualization for ResNet-152V2
├── gui.py                                   # Streamlit app for interactive model comparison
└── requirements.txt                         # Python dependencies
```

## Datasets

Three chest X-ray datasets are merged into a balanced master dataset:

| Dataset         | Source                               |
| --------------- | ------------------------------------ |
| Kermany         | Balanced subset (1,341 images/class) |
| NIH Chest X-ray | Sampled Subset                       |
| RSNA Pneumonia  | Sampled Subset                       |

**Final split:** ~14,054 train / 1,757 val / 1,757 test (80/10/10)  
**Preprocessing:** Resized to 224×224, normalized with ImageNet statistics (mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`)

## Models

| Model                                   | Framework        | Type                 |
| --------------------------------------- | ---------------- | -------------------- |
| baseline_resnet152.keras                | Keras/Tensorflow | Baseline Finetune    |
| baseline_resnet152_adversarial_combined | Pytorch          | Adversarial training |

Both models are pre-trained on ImageNet and fine-tuned for binary pneumonia classification.

## Methods

**Adversarial Attack — FGSM:**

$$x' = x + \varepsilon \cdot \text{sign}(\nabla_x \mathcal{L}(\theta, x, y))$$

Evaluated at ε ∈ {0.0, 0.01, 0.03, 0.05, 0.1}.

**Adversarial Training:**  
Mixed training with 50% clean + 50% FGSM-perturbed samples per batch.  
Loss = 0.5 × L_clean + 0.5 × L_adversarial

**Interpretability:**  
Grad-CAM visualizes which image regions drive model predictions.

## Results

??

## Setup

```bash
py -3.13 -m venv .venv
```

```bash
pip install -r requirements.txt
```

## Usage

**Run notebooks in order:**

1. `Phase1_Data_Preprocessing.ipynb` — prepare the dataset
2. `Phase2_Workflow.ipynb` — train and evaluate baseline model
3. `Phase3_FGSMAttack.ipynb` — evaluate FGSM robustness
4. `Phase4_AdvserialTraining.ipynb` — apply adversarial training

**Launch the interactive GUI:**

```bash
streamlit run gui.py
```

Provides side-by-side model comparison with Grad-CAM overlays on clean and adversarial inputs.

**Generate adversarial examples:**

```bash
python fgsm_export.py
```

**Visualize Grad-CAM:**

```bash
python gradcam_resnet152.py
```

## Trained Weights

| File                                          | Description                                |
| --------------------------------------------- | ------------------------------------------ |
| `baseline_resnet152_adversarial_combined.pth` | Adversarially-trained ResNet-152 (PyTorch) |

## Key Findings

- Baseline ResNet-152 achieves ~97% clean accuracy but is vulnerable to FGSM — small perturbations invisible to humans cause significant accuracy drops.
- Adversarial training recovers near-full clean accuracy while achieving near-perfect FGSM robustness.
- Grad-CAM reveals that adversarially-trained models attend more consistently to lung opacity regions rather than bone artifacts.

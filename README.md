# Chest X-ray Pneumonia Detection — Adversarial Robustness Study

## Project Overview

This project investigates the robustness of deep learning models for pneumonia detection in chest X-rays against adversarial attacks (FGSM), and evaluates how adversarial training improves model resilience across multiple datasets.

**Datasets used:**
- `chest_xray/` — Kaggle Chest X-Ray Images (Pneumonia) — adult patients
- `Master_Dataset/` — combined adult dataset used for adversarial training
- `Kermany_Pediatric_Attack/` (Pool B) — pediatric domain-shift evaluation set

---

## Environment Setup

**Python version:** 3.11 (Windows) / 3.13 (Mac)

Install dependencies:

```bash
pip install torch torchvision keras opencv-python matplotlib scikit-learn numpy pandas tqdm
```

> GPU strongly recommended. Running training or evaluation on CPU will be very slow and may crash due to memory.

---

## Repository Structure

| File / Folder | Description |
|---|---|
| `Phase2_CreateBaselineModels.ipynb` | Trains GoogLeNet, AlexNet, ResNet-18 on Master_Dataset — note: contains hardcoded Windows paths, update dataset paths before running |
| `Phase2_EvaluateBaselineModels.ipynb` | Evaluates baseline models — requires GPU |
| `Phase3_FGSMAttack.ipynb` | FGSM adversarial attack on AlexNet and ResNet-18 baselines |
| `Phase4_AdvserialTraining.ipynb` | Adversarial training on AlexNet and ResNet-18 (PyTorch + TensorFlow) |
| `CNNADVERSARIAL_PYTORCH_COMBINED.ipynb` | **Main adversarial training notebook** (PyTorch) |
| `CNN_PYTORCH_CONVERTED.ipynb` | PyTorch baseline ResNet-152 training |
| `baseline.py` | GoogLeNet baseline evaluator script |
| `fgsm_export.py` | Generates FGSM adversarial images from ResNet-152V2 |
| `gradcam_resnet152v2.py` | **Grad-CAM visualisation pipeline** |
| `requirements.txt` | Python dependencies |

---

## How to Run

### 1. Baseline evaluation
```bash
python baseline.py
```
Expects `./model/*.pth` and `./chest_xray/val`, `./chest_xray/test`.

---

### 2. Adversarial training (PyTorch — main path)

Open `CNNADVERSARIAL_PYTORCH_COMBINED.ipynb` and run all cells **except** the training loop cell (`adversarial_train(...)`).

The training loop requires GPU and was pre-run. The saved model `baseline_resnet152_adversarial_combined.pth` is required for the final evaluation cells.

The **Final Evaluation block** at the bottom loads the saved model and reports clean + FGSM accuracy across all 3 datasets.

---

### 3. GradCAM visualisation pipeline

**Step 1 — Generate FGSM adversarial images:**
```bash
python fgsm_export.py
```
Requires `baseline_resnet152v2_adversarial.keras` in the project root. Outputs 10 adversarial PNG images to `fgsm_outputs/`.

**Step 2 — Run Grad-CAM:**
```bash
python gradcam_resnet152v2.py
```
Requires:
- `baseline_resnet152v2_adversarial.keras`
- `fgsm_outputs/` (from Step 1)
- `chest_xray/test/` (clean images)
- `pool_b/Master_Dataset/test/` (pediatric images)

Outputs 3-panel figures (original | heatmap | overlay) to `gradcam_outputs/` across three sets: clean, FGSM-attacked, and pediatric.

---

## Grad-CAM Analysis

Grad-CAM is applied to the adversarially trained ResNet-152V2 model to visualise **which regions the model attends to** when making predictions.

Three image sets are compared:
- **Clean** — standard adult test images (baseline behaviour)
- **FGSM** — adversarially perturbed images (ε = 0.01)
- **Pediatric** — domain-shift images from a different population

The key research question: *Is the model attending to lung opacity (clinically meaningful) or bone density (spurious feature)?*

Target layer: `conv5_block3_out` (final convolutional block of ResNet-152V2)

---

## Model Files

Model weights are not tracked in git (`.pth`, `.keras` files are gitignored). Obtain from the team:

| File | Used by |
|---|---|
| `baseline_resnet152.pth` | `CNN_PYTORCH_CONVERTED.ipynb`, `CNNADVERSARIAL_PYTORCH_COMBINED.ipynb` |
| `baseline_resnet152_adversarial_combined.pth` | Final eval in `CNNADVERSARIAL_PYTORCH_COMBINED.ipynb` |
| `baseline_resnet152v2_adversarial.keras` | `fgsm_export.py`, `gradcam_resnet152v2.py` |

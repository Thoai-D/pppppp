"""
FGSM image exporter for ResNet-152V2 (keras 3, PyTorch backend).

Generates adversarial images by attacking baseline_resnet152v2_adversarial.keras,
saves them to fgsm_outputs/ as PNGs for use in gradcam_resnet152v2.py.

Usage:
    python fgsm_export.py
"""

import os
os.environ["KERAS_BACKEND"] = "torch"

import glob
import numpy as np
import cv2
import torch
import keras

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = "baseline_resnet152v2_adversarial.keras"
SOURCE_DIR = "chest_xray/test"
OUT_DIR    = "fgsm_outputs"
EPSILON    = 0.01
N_SAMPLES  = 10
IMG_SIZE   = (224, 224)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def preprocess(img_path: str) -> np.ndarray:
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE).astype(np.float32) / 255.0
    x = (img_resized - IMAGENET_MEAN) / IMAGENET_STD
    return np.expand_dims(x, 0)   # [1, H, W, C]


def fgsm_attack(model, img_array: np.ndarray, epsilon: float) -> np.ndarray:
    x = torch.tensor(img_array, dtype=torch.float32, requires_grad=True)
    preds = model(x)
    if preds.shape[-1] == 1:
        score = preds[0, 0]
    else:
        score = preds[0, int(preds[0].argmax())]
    model.zero_grad()
    score.backward()
    x_adv = x + epsilon * x.grad.sign()
    return x_adv.detach().numpy()


def save_adv(x_adv: np.ndarray, out_path: str) -> None:
    img = x_adv[0] * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img, 0.0, 1.0)
    img_bgr = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, img_bgr)


def collect_paths(directory: str, n: int) -> list[str]:
    exts = ("*.jpg", "*.jpeg", "*.png")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
        paths.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(set(paths))[:n]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading model from {MODEL_PATH!r} ...")
    model = keras.models.load_model(MODEL_PATH)

    paths = collect_paths(SOURCE_DIR, N_SAMPLES)
    if not paths:
        raise RuntimeError(f"No images found in {SOURCE_DIR!r}")

    print(f"Generating FGSM images (ε={EPSILON}) for {len(paths)} samples ...")
    for i, img_path in enumerate(paths):
        img_array = preprocess(img_path)
        x_adv = fgsm_attack(model, img_array, EPSILON)
        fname = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(OUT_DIR, f"{fname}_fgsm_eps{EPSILON:.3f}.png")
        save_adv(x_adv, out_path)
        print(f"  [{i+1}/{len(paths)}] saved: {out_path}")

    print(f"\nDone. {len(paths)} adversarial images in {OUT_DIR!r}")
    print("Next: python gradcam_resnet152v2.py")


if __name__ == "__main__":
    main()

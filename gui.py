import os
os.environ["KERAS_BACKEND"] = "torch"

import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.cm as cm
import keras

st.set_page_config(
    page_title="SUTD Group 2: X-ray Classifier",
    page_icon="🫁",
    layout="wide",
)

CLASSES = ["NORMAL", "PNEUMONIA"]
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]
IMAGENET_MEAN = np.array(IMG_MEAN, dtype=np.float32)
IMAGENET_STD  = np.array(IMG_STD,  dtype=np.float32)

MODEL_PATHS = {
    "Baseline ResNet-152V2":  "baseline_resnet152v2.keras",
    "Adversarial ResNet-152": "baseline_resnet152_adversarial_combined.pth",
}

_pt_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
])

# ── PyTorch architecture (adversarial model) ──────────────────────────────────
class ResNet152Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet152(weights=None)
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.model(x)

# ── PyTorch GradCAM ───────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, net: ResNet152Classifier):
        self.net = net
        self._feats = None
        self._grads = None
        target = net.model.layer4[-1]
        target.register_forward_hook(self._fwd_hook)
        target.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, _mod, _inp, out):
        self._feats = out.detach()

    def _bwd_hook(self, _mod, _grad_inp, grad_out):
        self._grads = grad_out[0].detach()

    def generate(self, img_tensor: torch.Tensor, device: torch.device) -> np.ndarray:
        self._feats = None
        self._grads = None
        img_tensor = img_tensor.to(device)
        self.net.zero_grad()
        with torch.enable_grad():
            logit = self.net(img_tensor)
            logit[0, 0].backward()
        if self._feats is None or self._grads is None:
            return np.zeros((7, 7))
        weights = self._grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self._feats).sum(dim=1)).squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam

# ── Keras GradCAM (baseline model, torch backend) ────────────────────────────
def _find_layer(model, name: str):
    for layer in model.layers:
        if layer.name == name:
            return layer
        if hasattr(layer, "layers"):
            found = _find_layer(layer, name)
            if found is not None:
                return found
    return None

class KerasGradCAM:
    TARGET_LAYER = "conv5_block3_out"

    def __init__(self, model):
        self.model = model
        self.target_layer = _find_layer(model, self.TARGET_LAYER)

    def generate(self, img_array: np.ndarray, device=None) -> np.ndarray:
        activations = [None]
        gradients   = [None]

        def fwd_hook(_mod, _inp, out):
            activations[0] = out

        def bwd_hook(_mod, _grad_in, grad_out):
            gradients[0] = grad_out[0]

        h_fwd = self.target_layer.register_forward_hook(fwd_hook)
        h_bwd = self.target_layer.register_full_backward_hook(bwd_hook)

        x = torch.tensor(img_array, dtype=torch.float32)
        preds = self.model(x)
        score = preds[0, 0] if preds.shape[-1] == 1 else preds[0, int(preds[0].argmax())]

        self.model.zero_grad()
        score.backward()
        h_fwd.remove()
        h_bwd.remove()

        act  = activations[0].detach().cpu()
        grad = gradients[0].detach().cpu()

        # Keras ResNet152V2 uses NHWC layout
        weights = grad[0].mean(dim=(0, 1))        # [C]
        heatmap = (act[0] * weights).sum(dim=-1)  # [H, W]

        heatmap = torch.relu(heatmap).numpy()
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        return heatmap.astype(np.float32)

# ── Heatmap overlay ───────────────────────────────────────────────────────────
def overlay_heatmap(pil_img: Image.Image, cam: np.ndarray, alpha: float) -> Image.Image:
    w, h = pil_img.size
    cam_pil = Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    cam_np = np.array(cam_pil) / 255.0
    heatmap = (cm.jet(cam_np)[:, :, :3] * 255).astype(np.uint8)
    orig = np.array(pil_img).astype(np.float32)
    blended = (alpha * heatmap.astype(np.float32) + (1 - alpha) * orig).clip(0, 255).astype(np.uint8)
    return Image.fromarray(blended)

# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_cam(path: str):
    if path.endswith(".keras"):
        model = keras.models.load_model(path)
        return model, KerasGradCAM(model), None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ResNet152Classifier()
    state = torch.load(path, map_location=device, weights_only=False)
    net.load_state_dict(state)
    net.to(device)
    net.eval()
    return net, GradCAM(net), device

# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(net, gradcam, device, pil_img: Image.Image, alpha: float):
    if device is None:  # Keras baseline
        img = pil_img.resize((224, 224))
        x = (np.array(img).astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        img_array = np.expand_dims(x, 0)  # [1, 224, 224, 3]
        cam = gradcam.generate(img_array)
        with torch.no_grad():
            preds = net(torch.tensor(img_array, dtype=torch.float32))
        prob = float(preds.detach().cpu().numpy()[0, 0])
    else:  # PyTorch adversarial
        img_t = _pt_transform(pil_img).unsqueeze(0)
        cam = gradcam.generate(img_t, device)
        with torch.no_grad():
            prob = torch.sigmoid(net(img_t.to(device))).item()

    label = CLASSES[int(prob >= 0.5)]
    overlay = overlay_heatmap(pil_img, cam, alpha)
    return label, prob, overlay

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    alpha = st.slider("GradCAM Opacity", 0.0, 1.0, 0.5, 0.05,
                      help="Blend strength of the heatmap over the original image.")
    st.divider()
    st.markdown("**Model Status**")
    for name, path in MODEL_PATHS.items():
        icon = "✅" if os.path.exists(path) else "❌"
        st.caption(f"{icon} {name}")
    st.divider()
    st.caption("Input: 224×224 RGB\nNorm: ImageNet Stats\nBaseline GradCAM: conv5_block3_out\nAdversarial GradCAM: layer4[-1]")

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🫁 Chest X-ray · Baseline vs Adversarial + GradCAM")
st.markdown(
    "Side-by-side comparison of **Baseline ResNet-152V2 (Keras)** vs "
    "**Adversarially-Trained ResNet-152 (PyTorch)** with GradCAM explanations. "
    "Heatmaps highlight regions most influential to the model's decision."
)

uploaded = st.file_uploader(
    "Upload a chest X-ray", type=["png", "jpg", "jpeg"], label_visibility="collapsed"
)
if not uploaded:
    st.info("Upload a chest X-ray image to begin.")
    st.stop()

image = Image.open(uploaded).convert("RGB")

_, col_img, _ = st.columns([1, 2, 1])
with col_img:
    st.image(image, caption="Input X-ray", use_container_width=True)

st.divider()
st.subheader("Model Comparison")

col_base, col_adv = st.columns(2)

for col, (model_name, model_path) in zip([col_base, col_adv], MODEL_PATHS.items()):
    with col:
        st.markdown(f"#### {model_name}")

        if not os.path.exists(model_path):
            st.error(f"Model file not found: `{model_path}`")
            st.caption("Ensure the file is downloaded from OneDrive/cloud storage.")
            continue

        with st.spinner("Running inference & GradCAM..."):
            net, gradcam, device = load_model_and_cam(model_path)
            label, prob, overlay = run_inference(net, gradcam, device, image, alpha)

        pneumonia_pct = prob * 100
        normal_pct   = (1 - prob) * 100

        if label == "PNEUMONIA":
            st.error(f"### {label}")
        else:
            st.success(f"### {label}")

        for cls, pct, color in [
            ("NORMAL",    normal_pct,   "#2ecc71"),
            ("PNEUMONIA", pneumonia_pct, "#e74c3c"),
        ]:
            st.markdown(f"**{cls}**: {pct:.1f}%")
            st.markdown(
                f'<div style="background:#444;height:10px;border-radius:5px">'
                f'<div style="background:{color};width:{pct:.1f}%;height:10px;border-radius:5px"></div>'
                f'</div><br>',
                unsafe_allow_html=True,
            )

        st.divider()
        st.image(overlay, caption="GradCAM Heatmap", use_container_width=True)
        st.caption(
            "Red = high activation (strong pneumonia signal). "
            "Blue = low activation."
        )

st.divider()
st.caption("SUTD AI & Cybersecurity Project · Developed by Group 2")

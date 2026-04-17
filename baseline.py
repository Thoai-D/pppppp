import glob
import torch
import torchvision
from torchvision import transforms, datasets
from torch import nn

# --- Model ---
model = torchvision.models.googlenet(pretrained=False, aux_logits=False)
model.fc = nn.Linear(1024, 2)

# Auto-find weights from ./model/
weight_files = glob.glob("./model/*.pth") + glob.glob("./model/*.pt")
if not weight_files:
    raise FileNotFoundError("No .pth or .pt weight file found in ./model/")
weight_path = weight_files[0]
print(f"Loading weights from: {weight_path}")
model.load_state_dict(torch.load(weight_path, map_location="cpu"))
model.eval()

# --- Transforms ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# --- Evaluation helper ---
def evaluate(split_path):
    dataset = datasets.ImageFolder(split_path, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total * 100

# --- Run ---
val_acc  = evaluate("./chest_xray/val")
test_acc = evaluate("./chest_xray/test")
print(f"Val  accuracy: {val_acc:.2f}%")
print(f"Test accuracy: {test_acc:.2f}%")

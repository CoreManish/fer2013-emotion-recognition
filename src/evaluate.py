import sys
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

from dataset import FER2013, CLASSES
from models import get_model

name = sys.argv[1]                       # baseline | resnet18 | se_resnet18
device = "cuda" if torch.cuda.is_available() else "cpu"

model = get_model(name).to(device)
model.load_state_dict(torch.load(f"../checkpoints/{name}.pt", map_location=device))
model.eval()

dl = DataLoader(FER2013("../data", "test"), 128)
preds, labels = [], []
with torch.no_grad():
    for x, y in dl:
        preds.append(model(x.to(device)).argmax(1).cpu()); labels.append(y)
preds, labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()

print(classification_report(labels, preds, target_names=CLASSES, digits=3))
cm = confusion_matrix(labels, preds, normalize="true")
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt=".2f", xticklabels=CLASSES, yticklabels=CLASSES, cmap="Blues")
plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(name)
plt.tight_layout(); plt.savefig(f"../results/{name}_cm.png", dpi=150)
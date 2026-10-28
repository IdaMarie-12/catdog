import argparse
import json
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

import config
from dataset import get_loaders, load_split
from model import load_model
from utils import get_device


def predict(model, loader, device):
    """Returns class probabilities (N,2) and labels (N,) in loader order"""
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            probs.append(torch.softmax(model(x.to(device)), dim=1).cpu())
            labels.append(y)
    return torch.cat(probs), torch.cat(labels)


def plot_confusion(cm, path):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], config.CLASSES)
    ax.set_yticks([0, 1], config.CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    name = ap.parse_args().name

    device = get_device()
    model, cfg = load_model(config.CKPT_DIR / f"{name}.pth", device)
    _, _, test_loader = get_loaders(augment=False)
    probs, labels = predict(model, test_loader, device)
    preds = probs.argmax(1)

    acc = (preds == labels).float().mean().item()
    print(f"TEST accuracy: {acc:.4f}  ({len(labels)} images)")
    cm = confusion_matrix(labels, preds)
    print("confusion matrix (rows=true, cols=pred):\n", cm)
    print(classification_report(labels, preds, target_names=config.CLASSES, digits=3))
    plot_confusion(cm, config.FIG_DIR / f"confusion_{name}.png")

    items = load_split()["test"]
    wrong = [dict(path=items[i][0], true=config.CLASSES[labels[i]],
                  pred=config.CLASSES[preds[i]], confidence=probs[i].max().item())
             for i in range(len(labels)) if preds[i] != labels[i]]
    with open(config.OUT_DIR / f"misclassified_{name}.json", "w") as f:
        json.dump(wrong, f, indent=2)
    print(f"{len(wrong)} misclassified images saved to misclassified_{name}.json")
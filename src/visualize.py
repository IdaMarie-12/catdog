import argparse
import csv
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image

import config
from dataset import denormalize, get_loaders, get_eval_transform, get_stats, load_split
from evaluate import predict
from model import load_model
from utils import get_device


def save_figure(fig, out):
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def read_results():
    results_file = config.RESULTS_FILE
    if not results_file.exists():
        results_file = config.ROOT / "experiments" / "results.csv"
    with open(results_file) as f:
        return list(csv.DictReader(f))


def plot_ablation_results(out):
    rows = read_results()
    names = [r["name"] for r in rows]
    acc = [float(r["best_val_acc"]) for r in rows]
    gaps = [float(r["gap"]) for r in rows]
    colors = ["C0" if n.startswith("s1_") else "C1" if n.startswith("s2_")
              else "C2" if n.startswith("s3_") else "C4" if n.startswith("fair_")
              else "C3" for n in names]
    fig, (ax_acc, ax_gap) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax_acc.bar(names, acc, color=colors)
    ax_acc.set_ylabel("Best validation accuracy")
    ax_acc.set_ylim(0.5, 1.0)
    ax_acc.grid(axis="y", alpha=0.3)
    fair_seed_rows = [r for r in rows
                      if r["name"] in {"fair_aug", "fair_aug_s1", "fair_aug_s2"}]
    seed_rows = fair_seed_rows or [
        r for r in rows if r["name"] in {"final", "final_s1", "final_s2"}
    ]
    seed_values = [float(r["best_val_acc"]) for r in seed_rows]
    seed_mean = sum(seed_values) / len(seed_values)
    seed_std = (sum((v - seed_mean) ** 2 for v in seed_values) /
                (len(seed_values) - 1)) ** 0.5
    ax_acc.axhspan(seed_mean - seed_std, seed_mean + seed_std, color="C3",
                   alpha=0.12, label=f"fair augmented seeds mean ± std ({seed_std:.3f})")
    ax_gap.bar(names, gaps, color=colors)
    ax_gap.axhline(0, color="black", linewidth=0.8)
    ax_gap.set_ylabel("Train - validation accuracy")
    ax_gap.set_ylim(min(-0.05, min(gaps) - 0.02), max(gaps) + 0.04)
    ax_gap.set_xticks(range(len(names)), names, rotation=60, ha="right", fontsize=8)
    ax_gap.grid(axis="y", alpha=0.3)
    legend = [
        Patch(facecolor="C0", label="Stage 1: regularization"),
        Patch(facecolor="C1", label="Stage 2: architecture"),
        Patch(facecolor="C2", label="Stage 3: optimization"),
        Patch(facecolor="C3", label="Stage 4: final seeds"),
        Patch(facecolor="C4", label="Fair long-budget retest"),
    ]
    ax_acc.legend(handles=legend + [
        Patch(facecolor="C3", alpha=0.12,
              label=f"fair augmented seed band: ±{seed_std:.3f}")
    ], loc="upper left", fontsize=8)
    save_figure(fig, out)


def plot_accuracy_vs_parameters(out):
    rows = read_results()
    selected_names = {"fair_aug_width8", "fair_aug_width16", "fair_aug"}
    rows = [r for r in rows if r["name"] in selected_names]
    fig, ax = plt.subplots(figsize=(7, 5))
    selected = sorted(rows, key=lambda r: int(r["params"]))
    labels = {"fair_aug_width8": "width 8", "fair_aug_width16": "width 16",
              "fair_aug": "width 64"}
    ax.scatter([int(r["params"]) for r in selected],
               [float(r["best_val_acc"]) for r in selected],
               color="C1", s=65, label="Fair augmented runs")
    for r in selected:
        ax.annotate(labels[r["name"]], (int(r["params"]), float(r["best_val_acc"])),
                    xytext=(5, 6), textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (log scale)")
    ax.set_ylabel("Best validation accuracy")
    ax.set_ylim(0.75, 0.83)
    ax.grid(alpha=0.3)
    ax.legend()
    save_figure(fig, out)


def plot_seed_variability(out):
    all_rows = read_results()
    fair_names = ["fair_aug", "fair_aug_s1", "fair_aug_s2"]
    names = fair_names if all(name in {r["name"] for r in all_rows} for name in fair_names) \
        else ["final", "final_s1", "final_s2"]
    rows = {r["name"]: r for r in all_rows}
    values = [float(rows[n]["best_val_acc"]) for n in names if n in rows]
    label_map = {
        "final": "seed 42",
        "final_s1": "seed 1",
        "final_s2": "seed 2",
        "fair_aug": "seed 42",
        "fair_aug_s1": "seed 1",
        "fair_aug_s2": "seed 2",
    }
    labels = [label_map[n] for n in names if n in rows]
    mean = sum(values) / len(values)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(range(len(values)), values, color="C3", s=70, zorder=3)
    ax.axhline(mean, color="C0", label=f"mean={mean:.3f}")
    if len(values) > 1:
        std = (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5
        ax.axhspan(mean - std, mean + std, color="C0", alpha=0.15, label=f"std={std:.3f}")
    ax.set_xticks(range(len(values)), labels)
    ax.set_ylabel("Best validation accuracy")
    ax.set_ylim(0.5, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    save_figure(fig, out)


def _test_predictions(model, device):
    _, _, loader = get_loaders(augment=False)
    probs, labels = predict(model, loader, device)
    return probs, labels, load_split()["test"]


def _gradcam(model, image, target, device):
    activations, gradients = [], []
    layer = [m for m in model.features if isinstance(m, nn.Conv2d)][-1]
    forward_hook = layer.register_forward_hook(lambda _, __, output: activations.append(output))
    backward_hook = layer.register_full_backward_hook(lambda _, grad_input, grad_output:
                                                       gradients.append(grad_output[0]))
    model.zero_grad(set_to_none=True)
    logits = model(image.unsqueeze(0).to(device))
    other = 1 - target
    score = logits[0, target] - logits[0, other]
    score.backward()
    forward_hook.remove()
    backward_hook.remove()
    weights = gradients[0].mean(dim=(2, 3), keepdim=True)
    cam = (weights * activations[0]).sum(dim=1).relu()[0]
    cam_max = cam.max()
    used_magnitude = False
    if cam_max <= 1e-8:
        print("Grad-CAM target produced no positive activation; using magnitude map")
        cam = (weights * activations[0]).sum(dim=1).abs()[0]
        cam_max = cam.max()
        used_magnitude = True
    if cam_max <= 1e-8:
        raise RuntimeError("Grad-CAM produced an all-zero activation map")
    heatmap = cam / cam_max
    return heatmap.detach().cpu(), used_magnitude


def plot_gradcam(model, device, out, max_each=3):
    probs, labels, items = _test_predictions(model, device)
    preds = probs.argmax(1)
    correct = [i for i in range(len(labels)) if preds[i] == labels[i]][:max_each]
    wrong = [i for i in range(len(labels)) if preds[i] != labels[i]][:max_each]
    tf = get_eval_transform()
    fig, axes = plt.subplots(2, max_each, figsize=(3 * max_each, 6), squeeze=False)
    for row, indices, title in ((0, correct, "Correct"), (1, wrong, "Misclassified")):
        for col in range(max_each):
            ax = axes[row, col]
            ax.axis("off")
            if col >= len(indices):
                continue
            i = indices[col]
            path = config.DATA_DIR / items[i][0]
            raw = Image.open(path).convert("RGB")
            heatmap, used_magnitude = _gradcam(model, tf(raw), int(preds[i]), device)
            heatmap = F.interpolate(
                heatmap[None, None],
                size=(raw.height, raw.width),
                mode="bilinear",
                align_corners=False,
            )[0, 0].numpy()
            alpha = ((heatmap - 0.35) / 0.65).clip(0, 1) * 0.7
            ax.imshow(raw)
            ax.imshow(heatmap, cmap="jet", alpha=alpha,
                      extent=(0, raw.width, raw.height, 0))
            fallback = "\n[magnitude CAM]" if used_magnitude else ""
            ax.set_title(f"{title}: true {config.CLASSES[labels[i]]}, "
                         f"pred {config.CLASSES[preds[i]]} ({probs[i].max():.0%})"
                         f"{fallback}", fontsize=8)
        axes[row, 0].set_ylabel(title)
    save_figure(fig, out)


def plot_confidence(out, probs, labels):
    preds = probs.argmax(1)
    confidence = probs.max(1).values
    correct = preds == labels
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(confidence[correct].numpy(), bins=10, alpha=0.7, label="Correct")
    ax.hist(confidence[~correct].numpy(), bins=10, alpha=0.7, label="Incorrect")
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Number of test images")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    save_figure(fig, out)


def plot_correct_examples(model, device, out, max_n=8):
    probs, labels, items = _test_predictions(model, device)
    preds = probs.argmax(1)
    correct = [i for i in range(len(labels)) if preds[i] == labels[i]][:max_n]
    cols = 4
    fig, axes = plt.subplots(math.ceil(len(correct) / cols), cols,
                             figsize=(3 * cols, 3 * math.ceil(len(correct) / cols)),
                             squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), correct):
        ax.imshow(Image.open(config.DATA_DIR / items[i][0]).convert("RGB"))
        ax.set_title(f"{config.CLASSES[preds[i]]} ({probs[i].max():.0%})", fontsize=9)
    save_figure(fig, out)


def plot_architecture(out, cfg):
    channels = [cfg["base"] * 2 ** i for i in range(cfg["n_blocks"])]
    sizes = [config.IMG_SIZE // 2 ** (i + 1) for i in range(cfg["n_blocks"])]
    labels = [f"Input\n3@{config.IMG_SIZE}×{config.IMG_SIZE}"]
    for channel, size in zip(channels, sizes):
        labels.append(f"Conv {channel}\nBN → ReLU → Pool\n{channel}@{size}×{size}")
    labels += [f"Global average\npooling\n{channels[-1]}@1×1", "Linear\n2 logits"]
    fig, ax = plt.subplots(figsize=(13, 2.7))
    ax.axis("off")
    for i, label in enumerate(labels):
        x = i * 1.35
        ax.text(x, 0.12, label, ha="center", va="center", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.6", facecolor="lightsteelblue",
                          edgecolor="C0"))
        if i < len(labels) - 1:
            ax.annotate("", xy=(x + 0.87, 0.12), xytext=(x + 0.48, 0.12),
                        arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(4.05, -0.42, "Feature-map sizes shown after each MaxPool2d stage.",
            ha="center", va="center", fontsize=9)
    ax.set_xlim(-0.5, (len(labels) - 1) * 1.35 + 0.5)
    ax.set_ylim(-0.58, 0.55)
    save_figure(fig, out)


def plot_curves(names, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name in names:
        with open(config.HIST_DIR / f"{name}.csv") as f:
            rows = list(csv.DictReader(f))
        ep = [int(r["epoch"]) for r in rows]
        best_row = min(rows, key=lambda r: float(r["val_loss"]))
        best_epoch = int(best_row["epoch"])
        for ax, key in zip(axes, ["loss", "acc"]):
            l1, = ax.plot(ep, [float(r[f"train_{key}"]) for r in rows], "--", label=f"{name} train")
            ax.plot(ep, [float(r[f"val_{key}"]) for r in rows], "-", color=l1.get_color(),
                    label=f"{name} val")
            ax.scatter([best_epoch], [float(best_row[f"val_{key}"])], color=l1.get_color(),
                       s=35, zorder=3)
            ax.axvline(best_epoch, color=l1.get_color(), alpha=0.15)
    for ax, t in zip(axes, ["Loss", "Accuracy"]):
        ax.set_title(t)
        ax.set_xlabel("epoch")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_filters(model, out, n=16):
    conv = next(m for m in model.features if isinstance(m, nn.Conv2d))
    w = conv.weight.detach().cpu()[:n]                 # (n, 3, k, k)
    fig, axes = plt.subplots(2, 8, figsize=(12, 3.5))
    for ax, f in zip(axes.ravel(), w):
        f = (f - f.min()) / (f.max() - f.min() + 1e-8)
        ax.imshow(f.permute(1, 2, 0), interpolation="nearest")
        ax.axis("off")
    plt.suptitle("First-layer filters")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_feature_maps(model, image, device, out, n_maps=8):
    acts = []
    hooks = [m.register_forward_hook(lambda mod, i, o: acts.append(o.detach().cpu().clone()))
             for m in model.features if isinstance(m, nn.ReLU)]
    with torch.no_grad():
        model(image.unsqueeze(0).to(device))
    for h in hooks:
        h.remove()
    mean, std = get_stats()
    fig = plt.figure(figsize=(2 * (n_maps + 1), 2 * len(acts)))
    grid = fig.add_gridspec(len(acts), n_maps + 1, width_ratios=[1.5] + [1] * n_maps)
    original_ax = fig.add_subplot(grid[:, 0])
    original_ax.imshow(denormalize(image.cpu(), mean, std).permute(1, 2, 0))
    original_ax.set_title("Original image")
    original_ax.axis("off")
    for r, a in enumerate(acts):
        for c in range(n_maps):
            ax = fig.add_subplot(grid[r, c + 1])
            ax.imshow(a[0, c], cmap="viridis")
            ax.axis("off")
            if c == 0:
                ax.set_title(f"block {r + 1}: before pool\n{a.shape[2]}×{a.shape[3]}",
                             fontsize=8, loc="left")
    fig.suptitle("Feature maps are captured after ReLU and before MaxPool2d.",
                 fontsize=11)
    save_figure(fig, out)


def plot_misclassified(model, device, out, max_n=16):
    _, _, test_loader = get_loaders(augment=False)
    probs, labels = predict(model, test_loader, device)
    preds = probs.argmax(1)
    items = load_split()["test"]
    wrong = [i for i in range(len(labels)) if preds[i] != labels[i]][:max_n]
    if not wrong:
        print("No misclassified images.")
        return
    cols = 4
    rows = (len(wrong) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), wrong):
        ax.imshow(Image.open(config.DATA_DIR / items[i][0]).convert("RGB"))
        ax.set_title(f"true: {config.CLASSES[labels[i]]}\npred: {config.CLASSES[preds[i]]} "
                     f"({probs[i].max():.0%})", fontsize=9, color="red")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def plot_own_pets(model, device, out):
    files = sorted(p for p in config.PETS_DIR.glob("*")
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not files:
        print(f"No images in {config.PETS_DIR}, skipping.")
        return
    tf = get_eval_transform()
    cols = min(4, len(files))
    rows = (len(files) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.5 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for index, (ax, p) in enumerate(zip(axes.ravel(), files), start=1):
        img = Image.open(p).convert("RGB")
        with torch.no_grad():
            prob = torch.softmax(model(tf(img).unsqueeze(0).to(device)), dim=1)[0].cpu()
        ax.imshow(img)
        ax.set_title(f"pet {index}: {config.CLASSES[prob.argmax()]} ({prob.max():.0%})",
                     fontsize=10)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="checkpoint/run name of the final model")
    ap.add_argument("--compare", nargs="*", default=["fair_noaug", "fair_aug"],
                    help="other runs for the curves")
    args = ap.parse_args()

    device = get_device()
    model, model_cfg = load_model(config.CKPT_DIR / f"{args.name}.pth", device)
    fig = config.FIG_DIR

    plot_curves(args.compare, fig / "curves.png")
    plot_filters(model, fig / "filters.png")
    img, _ = get_loaders(augment=False)[2].dataset[0]
    plot_feature_maps(model, img, device, fig / "feature_maps.png")
    plot_misclassified(model, device, fig / "misclassified.png")
    plot_own_pets(model, device, fig / "own_pets.png")
    plot_ablation_results(fig / "ablation_results.png")
    plot_accuracy_vs_parameters(fig / "accuracy_vs_parameters.png")
    plot_seed_variability(fig / "seed_variability.png")
    plot_gradcam(model, device, fig / "gradcam_examples.png")
    probs, labels, _ = _test_predictions(model, device)
    plot_confidence(fig / "confidence_histogram.png", probs, labels)
    plot_correct_examples(model, device, fig / "correct_examples.png")
    plot_architecture(fig / "architecture.png", model_cfg)
    print("figures saved to", fig)
import json
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import config


def load_split():
    with open(config.SPLIT_FILE) as f:
        return json.load(f)


class CatDogDataset(Dataset):
    """items: list of [relative_path, label]."""

    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        rel, label = self.items[i]
        img = Image.open(config.DATA_DIR / rel).convert("RGB")
        return self.transform(img), label


def get_stats():
    """Per-channel mean/std computed on the TRAINING split only (cached)"""
    if config.STATS_FILE.exists():
        with open(config.STATS_FILE) as f:
            s = json.load(f)
        return s["mean"], s["std"]

    tf = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor()])
    loader = DataLoader(CatDogDataset(load_split()["train"], tf),
                        batch_size=64, num_workers=0)
    n, s, sq = 0, torch.zeros(3), torch.zeros(3)
    for x, _ in loader:
        n += x.shape[0] * x.shape[2] * x.shape[3]
        s += x.sum(dim=(0, 2, 3))
        sq += (x ** 2).sum(dim=(0, 2, 3))
    mean = s / n
    std = (sq / n - mean ** 2).sqrt()
    with open(config.STATS_FILE, "w") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist()}, f)
    return mean.tolist(), std.tolist()


def get_transform(train, augment, mean, std):
    size = config.IMG_SIZE
    if train and augment:
        steps = [
            transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ]
    else:
        steps = [transforms.Resize((size, size))]
    steps += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(steps)


def get_eval_transform():
    mean, std = get_stats()
    return get_transform(False, False, mean, std)


def get_loaders(augment=True, batch_size=32, num_workers=None, subset=None, seed=None):
    """Returns train, val, test loaders. `subset` = use only n train images"""
    num_workers = config.NUM_WORKERS if num_workers is None else num_workers
    split = load_split()
    mean, std = get_stats()
    train_items = split["train"][:subset] if subset else split["train"]

    train_ds = CatDogDataset(train_items, get_transform(True, augment, mean, std))
    eval_tf = get_transform(False, False, mean, std)
    val_ds = CatDogDataset(split["val"], eval_tf)
    test_ds = CatDogDataset(split["test"], eval_tf)

    kw = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available(),
              persistent_workers=num_workers > 0)
    g = torch.Generator().manual_seed(config.SEED if seed is None else seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              generator=g, **kw)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, **kw)
    return train_loader, val_loader, test_loader


def denormalize(x, mean, std):
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    return (x * s + m).clamp(0, 1)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    mean, std = get_stats()
    print("train mean/std:", mean, std)
    train_items = load_split()["train"]
    items = [next(item for item in train_items if item[1] == label)
             for label in (0, 1)]
    aug = CatDogDataset(items, get_transform(True, True, mean, std))
    orig = CatDogDataset(items, get_transform(False, False, mean, std))
    fig, axes = plt.subplots(2, 6, figsize=(12, 5))
    for r in range(2):
        axes[r, 0].imshow(denormalize(orig[r][0], mean, std).permute(1, 2, 0))
        axes[r, 0].set_title(f"original ({config.CLASSES[items[r][1]]})", fontsize=8)
        for c in range(1, 6):
            axes[r, c].imshow(denormalize(aug[r][0], mean, std).permute(1, 2, 0))
            axes[r, c].set_title("augmented", fontsize=8)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle("On-the-fly augmentation examples used by the selected final model",
                 fontsize=11)
    plt.tight_layout()
    out = config.FIG_DIR / "augmentation_examples.png"
    plt.savefig(out, dpi=120)
    print("saved", out)
import argparse
import csv
import torch
import torch.nn as nn

import config
from dataset import get_loaders
from model import build_model
from utils import set_seed, get_device, count_parameters


def run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass over a loader. Trains if an optimizer is given, else evaluates"""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def make_optimizer(model, cfg):
    p = model.parameters()
    if cfg["optimizer"] == "adam":
        return torch.optim.Adam(p, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    if cfg["optimizer"] == "adamw":
        return torch.optim.AdamW(p, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    if cfg["optimizer"] == "sgd":
        return torch.optim.SGD(p, lr=cfg["lr"], momentum=0.9,
                               weight_decay=cfg["weight_decay"])
    raise ValueError(cfg["optimizer"])


def train(cfg, name, overfit=False, verbose=True):
    cfg = dict(cfg)
    subset = None
    if overfit:   # sanity check: the model must be able to memorise 32 images
        cfg.update(augment=False, dropout=0.0, weight_decay=0.0,
                   patience=10 ** 9, epochs=100)
        subset = 32

    set_seed(cfg["seed"])
    device = get_device()
    train_loader, val_loader, _ = get_loaders(
        cfg["augment"], cfg["batch_size"], subset=subset, seed=cfg["seed"])

    model = build_model(cfg).to(device)
    n_params = count_parameters(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, cfg)
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
                 if cfg["scheduler"] else None)

    if verbose:
        print(f"[{name}] device={device} params={n_params:,}")
        l0, _ = run_epoch(model, val_loader, criterion, device)
        print(f"[{name}] initial val loss {l0:.3f} (about 0.69 = ln 2 expected)")

    history, best = [], dict(val_loss=float("inf"), epoch=0)
    wait = 0
    ckpt_path = config.CKPT_DIR / f"{name}.pth"

    for epoch in range(1, cfg["epochs"] + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
        if scheduler:
            scheduler.step(va_loss)
        history.append(dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc,
                            val_loss=va_loss, val_acc=va_acc))
        if verbose:
            print(f"[{name}] epoch {epoch:3d} | train {tr_loss:.4f}/{tr_acc:.3f} "
                  f"| val {va_loss:.4f}/{va_acc:.3f}")

        if va_loss < best["val_loss"]:                       # early stopping
            best = dict(val_loss=va_loss, val_acc=va_acc, train_acc=tr_acc, epoch=epoch)
            wait = 0
            if not overfit:
                torch.save({"cfg": cfg, "state_dict": model.state_dict()}, ckpt_path)
        else:
            wait += 1
            if wait >= cfg["patience"]:
                if verbose:
                    print(f"[{name}] early stopping at epoch {epoch}")
                break

    if not overfit:
        with open(config.HIST_DIR / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            w.writeheader()
            w.writerows(history)

    if verbose:
        print(f"[{name}] best epoch {best['epoch']}: val loss {best['val_loss']:.4f}, "
              f"val acc {best.get('val_acc', 0):.3f}")
    return dict(name=name, params=n_params, best_epoch=best["epoch"],
                best_val_loss=best["val_loss"], best_val_acc=best.get("val_acc", 0),
                train_acc_at_best=best.get("train_acc", 0), cfg=cfg)


def str2bool(v):
    return str(v).lower() in ("1", "true", "yes", "y")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="run")
    p.add_argument("--overfit", action="store_true", help="sanity check on 32 images")
    p.add_argument("--baseline", action="store_true", help="no regularization")
    for k, v in config.DEFAULTS.items():
        p.add_argument(f"--{k}", type=str2bool if isinstance(v, bool) else type(v),
                       default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = dict(config.DEFAULTS)
    if args.baseline:
        cfg.update(config.BASELINE)
    for k in config.DEFAULTS:
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)
    train(cfg, "overfit_test" if args.overfit else args.name, overfit=args.overfit)
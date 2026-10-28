import argparse
import csv
import config
from train import train

# Original short-budget search configuration. The fair stage 5 is the
# reproducible source of the promoted final configuration.
BEST = dict(
    base=64,
    n_blocks=4,
    augment=False,
    dropout=0.0,
    weight_decay=1e-4,
    optimizer="adamw",
    lr=1e-3,
    batch_size=32,
    scheduler=False,
)

STAGES = {
    1: ("s1", config.BASELINE, {
        "baseline": {},
        "aug": dict(augment=True),
        "drop0.2": dict(dropout=0.2),
        "drop0.5": dict(dropout=0.5),
        "wd1e-4": dict(weight_decay=1e-4),
        "wd1e-3": dict(weight_decay=1e-3),
        "nobatchnorm": dict(batchnorm=False),
    }),
    2: ("s2", BEST, {
        "depth3": dict(n_blocks=3),
        "depth4": dict(n_blocks=4),
        "depth5": dict(n_blocks=5),
        "width8": dict(base=8),
        "width32": dict(base=32),
        "width64": dict(base=64),
    }),
    3: ("s3", BEST, {
        "lr1e-2": dict(lr=1e-2),
        "lr1e-3": dict(lr=1e-3),
        "lr1e-4": dict(lr=1e-4),
        "bs16": dict(batch_size=16),
        "bs64": dict(batch_size=64),
        "sgd": dict(optimizer="sgd", lr=1e-2),
        "scheduler": dict(scheduler=True),
    }),
    4: ("final", BEST, {
        "": dict(seed=42),
        "_s1": dict(seed=1),
        "_s2": dict(seed=2),
    }),
    5: ("fair", dict(base=64, n_blocks=4, augment=True, dropout=0.0,
                     batchnorm=True, weight_decay=1e-4, optimizer="adamw",
                     lr=1e-3, batch_size=32, epochs=100, patience=20,
                     scheduler=True), {
        "_aug": dict(seed=42),
        "_aug_s1": dict(seed=1),
        "_aug_s2": dict(seed=2),
        "_noaug": dict(augment=False, seed=42),
        "_noaug_s1": dict(augment=False, seed=1),
        "_noaug_s2": dict(augment=False, seed=2),
        "_aug_width8": dict(base=8, seed=42),
        "_aug_width16": dict(base=16, seed=42),
    }),
}

FIELDS = ["name", "params", "best_epoch", "best_val_loss", "best_val_acc",
          "train_acc_at_best", "gap"] + list(config.DEFAULTS.keys())


def done_names():
    if not config.RESULTS_FILE.exists():
        return set()
    with open(config.RESULTS_FILE) as f:
        return {r["name"] for r in csv.DictReader(f)}


def append(row):
    new = not config.RESULTS_FILE.exists()
    with open(config.RESULTS_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True, choices=STAGES)
    stage = ap.parse_args().stage

    prefix, base, experiments = STAGES[stage]
    for exp, override in experiments.items():
        name = f"{prefix}_{exp}" if stage != 4 else f"{prefix}{exp}"
        if stage == 5:
            name = f"fair{exp}"
        if name in done_names():
            print("skip", name)
            continue
        cfg = {**config.DEFAULTS, **base, **override}
        res = train(cfg, name, verbose=False)
        row = {"name": name, "params": res["params"], "best_epoch": res["best_epoch"],
               "best_val_loss": round(res["best_val_loss"], 4),
               "best_val_acc": round(res["best_val_acc"], 4),
               "train_acc_at_best": round(res["train_acc_at_best"], 4),
               "gap": round(res["train_acc_at_best"] - res["best_val_acc"], 4)}
        row.update({k: cfg[k] for k in config.DEFAULTS})
        append(row)
        print(f"{name:20s} params={res['params']:>8,} val_acc={res['best_val_acc']:.3f} "
              f"val_loss={res['best_val_loss']:.3f} (epoch {res['best_epoch']})")
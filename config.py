"""Central configuration: paths, constants and default hyperparameters."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "catdog_data"      # contains cats/ and dogs/
PETS_DIR = ROOT / "my_pets"                   # your own photos for the fun test
OUT_DIR = ROOT / "outputs"
CKPT_DIR = OUT_DIR / "checkpoints"
HIST_DIR = OUT_DIR / "history"
FIG_DIR = OUT_DIR / "figures"
SPLIT_FILE = ROOT / "split.json"
STATS_FILE = ROOT / "stats.json"              # mean/std of the training set
RESULTS_FILE = ROOT / "experiments" / "results.csv"  # one row per experiment

CLASS_FOLDERS = {"cats": 0, "dogs": 1}
CLASSES = ["cat", "dog"]

SEED = 42                                     # seed for the data split
IMG_SIZE = 128
NUM_WORKERS = 2                               # set to 0 if you get DataLoader errors

SPLIT = dict(train=0.70, val=0.15, test=0.15)

# Default hyperparameters of one training run
DEFAULTS = dict(
    base=16,             # channels in first conv block (doubles every block)
    n_blocks=4,          # number of conv blocks
    dropout=0.3,
    batchnorm=True,
    augment=True,
    optimizer="adamw",   # adam | adamw | sgd
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=32,
    epochs=60,
    patience=10,         # early stopping patience (epochs)
    scheduler=False,     # ReduceLROnPlateau on validation loss
    seed=42,             # seed for weight init / batch order
)

# "Baseline" = no regularization at all (expected to overfit)
BASELINE = dict(augment=False, dropout=0.0, weight_decay=0.0)

for d in (OUT_DIR, CKPT_DIR, HIST_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)
    
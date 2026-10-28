# Cat & Dog classification (PyTorch)

Put the unzipped data in `data/catdog_data/cats` and `data/catdog_data/dogs`,
your own photos in `my_pets/`.

    python3 -m pip install -r requirements.txt
    export PYTHONPATH=.:src
    python3 src/split.py                    # 1. split -> split.json
    python3 src/dataset.py                  # 2. check augmentations (figure)
    python3 src/model.py                    # 3. check output shape / parameters
    python3 src/train.py --overfit           # 4. sanity check: should reach ~100% train acc
    python3 src/train.py --name test --baseline  # 5. first baseline run
    python3 experiments/run_search.py --stage 1  # 6. experiments (edit BEST in run_search.py between stages)
    python3 experiments/run_search.py --stage 2
    python3 experiments/run_search.py --stage 3
    python3 experiments/run_search.py --stage 4  # original short-budget seeds
    python3 experiments/run_search.py --stage 5  # fair augmented/no-augmented retest
    python3 src/evaluate.py --name final       # selected augmented checkpoint
    python3 src/visualize.py --name final

The submitted final rule is: select the augmented checkpoint with the lowest
validation loss among fair_aug, fair_aug_s1, and fair_aug_s2. This selects
fair_aug_s1, which is copied to final.pth. The original final checkpoints
remain available for comparison.

Run the tests with:

    python3 -m unittest discover -s tests
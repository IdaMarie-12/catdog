import torch
import torch.nn as nn
from utils import count_parameters


class SmallCNN(nn.Module):
    """[Conv3x3 -> (BatchNorm) -> ReLU -> MaxPool] x n_blocks
       -> global average pooling -> dropout -> linear (2 logits)"""

    def __init__(self, base=16, n_blocks=4, dropout=0.3, batchnorm=True, n_classes=2):
        super().__init__()
        layers, in_ch = [], 3
        for i in range(n_blocks):
            out_ch = base * 2 ** i
            layers.append(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=not batchnorm))
            if batchnorm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers += [nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(in_ch, n_classes)

    def forward(self, x):
        x = self.pool(self.features(x)).flatten(1)
        return self.fc(self.drop(x))          # raw logits (softmax is in the loss)


def build_model(cfg):
    return SmallCNN(cfg["base"], cfg["n_blocks"], cfg["dropout"], cfg["batchnorm"])


def load_model(ckpt_path, device):
    """Rebuilds a model from a checkpoint saved by train.py"""
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(ckpt["cfg"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["cfg"]


if __name__ == "__main__":
    import config
    m = build_model(config.DEFAULTS)
    out = m(torch.randn(8, 3, config.IMG_SIZE, config.IMG_SIZE))
    print(m)
    print("output shape:", tuple(out.shape), "(expected (8, 2))")
    print("parameters:", count_parameters(m))
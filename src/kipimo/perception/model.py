"""Tiny UNet, 4 down and 4 up, trained on synthetic tiles and exported as int8 ONNX.

uv run --group train python -m kipimo.perception.model --minutes 10
"""

import argparse
import time

import numpy as np

from kipimo.config import ROOT
from kipimo.world.generate import generate
from kipimo.world.render import render_tile, tile_truth

MODEL_DIR = ROOT / "models"
FP32 = MODEL_DIR / "unet_fp32.onnx"
INT8 = MODEL_DIR / "unet_int8.onnx"
TRAIN_SEEDS = range(1000, 1040)  # disjoint from demo and benchmark seeds


def dataset(n: int, seed: int, seeds=TRAIN_SEEDS[:36]) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    worlds = [generate(s) for s in seeds]
    X = np.empty((n, 3, 64, 64), np.float32)
    Y = np.empty((n, 2, 64, 64), np.float32)
    for i in range(n):
        w = worlds[i % len(worlds)]
        span = w.size * w.cell_m
        x, y = rng.uniform(60, span - 60, 2)
        img, ids = render_tile(
            w, x, y, float(rng.uniform(20, 50)), float(rng.uniform(0.25, 1)), rng
        )
        X[i] = img.transpose(2, 0, 1)
        Y[i] = tile_truth(w, ids)
    return X, Y


def build():
    import torch
    from torch import nn

    def block(i, o):
        return nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1), nn.ReLU(), nn.Conv2d(o, o, 3, padding=1), nn.ReLU()
        )

    class UNet(nn.Module):
        def __init__(self, ch=(8, 16, 32, 64)):
            super().__init__()
            self.down = nn.ModuleList()
            c = 3
            for o in ch:
                self.down.append(block(c, o))
                c = o
            self.mid = block(ch[-1], ch[-1] * 2)
            self.up = nn.ModuleList()
            self.dec = nn.ModuleList()
            c = ch[-1] * 2
            for o in reversed(ch):
                self.up.append(nn.ConvTranspose2d(c, o, 2, stride=2))
                self.dec.append(block(2 * o, o))
                c = o
            self.head = nn.Conv2d(ch[0], 2, 1)
            self.pool = nn.MaxPool2d(2)

        def forward(self, x):
            skips = []
            for d in self.down:
                x = d(x)
                skips.append(x)
                x = self.pool(x)
            x = self.mid(x)
            for up, dec, s in zip(self.up, self.dec, reversed(skips), strict=True):
                x = dec(torch.cat([up(x), s], 1))
            return torch.sigmoid(self.head(x))

    return UNet()


def train(minutes: float, n: int = 6000) -> None:
    import torch

    torch.manual_seed(0)
    X, Y = dataset(n, seed=1)
    Xv, Yv = dataset(400, seed=2, seeds=TRAIN_SEEDS[36:])  # worlds training never sees
    X, Y, Xv, Yv = map(torch.from_numpy, (X, Y, Xv, Yv))
    model = build()
    params = sum(p.numel() for p in model.parameters())
    print(f"params {params:,}  train tiles {n}")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(0)
    end = time.time() + minutes * 60
    epoch = 0
    while time.time() < end:
        model.train()
        for idx in torch.randperm(len(X), generator=g).split(64):
            opt.zero_grad()
            loss = torch.nn.functional.l1_loss(model(X[idx]), Y[idx])
            loss.backward()
            opt.step()
            if time.time() > end:
                break
        model.eval()
        with torch.no_grad():
            val = (model(Xv) - Yv).abs().mean((0, 2, 3)).tolist()
        epoch += 1
        print(f"epoch {epoch}  val MAE water {val[0]:.3f} loss {val[1]:.3f}", flush=True)
    MODEL_DIR.mkdir(exist_ok=True)
    torch.onnx.export(
        model,
        X[:1],
        FP32,
        input_names=["image"],
        output_names=["pred"],
        dynamic_axes={"image": {0: "n"}, "pred": {0: "n"}},
        opset_version=17,
        dynamo=False,
    )
    quantize()


def quantize() -> None:
    """Dynamic int8: int8 weights, activation ranges measured per image at run time.

    Static int8 with fixed activation ranges squashed deep water readings to about 0.5 in every
    calibration mode tried, because flood water and wet crop differ by a few grey levels.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(FP32), str(INT8), weight_type=QuantType.QUInt8)
    for p in (FP32, INT8):
        print(f"{p.name}: {p.stat().st_size / 1e6:.3f} MB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--tiles", type=int, default=6000)
    ap.add_argument("--quantize-only", action="store_true")
    a = ap.parse_args()
    if a.quantize_only:
        quantize()
    else:
        train(a.minutes, a.tiles)


if __name__ == "__main__":
    main()

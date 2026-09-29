import argparse
import csv
import json
from pathlib import Path

import PIL.Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLAHE_CLIP, CLAHE_GRID = 1.0, (8, 8)
DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "weights"
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.json"


ID2LABEL = {0: "Normal", 1: "TB"}


def load_config(config_path=DEFAULT_CONFIG):
    return json.loads(Path(config_path).read_text())


def minmax_uint8(arr):
    arr = np.asarray(arr).astype(np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    return ((arr - mn) / (mx - mn + 1e-8) * 255.0).astype(np.uint8)


def png_to_image(png_path):
    import cv2

    base = minmax_uint8(np.asarray(PIL.Image.open(png_path).convert("L")))
    eq = cv2.equalizeHist(base)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID).apply(base)
    return np.stack([eq, clahe, base], axis=-1)


def pad_to_square(img):
    h, w = img.shape[:2]
    if h == w:
        return img
    s = max(h, w)
    top, left = (s - h) // 2, (s - w) // 2
    out = np.zeros((s, s, img.shape[2]), dtype=img.dtype)
    out[top : top + h, left : left + w, :] = img
    return out


def to_input(img3, size, mean, std, device, dtype):
    x = torch.from_numpy(np.ascontiguousarray(pad_to_square(img3))).permute(2, 0, 1)[None].float()
    if tuple(x.shape[-2:]) != (size, size):
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
    x = x / 255.0
    x = (x - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)
    return x[0].to(device=device, dtype=dtype)


def build_vit(tower_cfg):
    from transformers import AutoConfig, AutoModel

    cfg = AutoConfig.for_model(tower_cfg["model_type"], **{k: v for k, v in tower_cfg.items() if k != "model_type"})
    return AutoModel.from_config(cfg)


class AttnPool(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens):
        q = self.query.expand(tokens.shape[0], -1, -1)
        pooled, _ = self.attn(q, tokens, tokens, need_weights=False)
        return self.norm(pooled.squeeze(1))


class Head(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_classes, dropout):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.mlp(x).squeeze(-1)


class TBModel(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.vision_model = build_vit(spec["tower"])
        self.patch_start = int(spec["patch_start"])
        d = int(spec["head"]["vision_hidden_dim"])
        self.pool = AttnPool(d, num_heads=int(spec["pool"]["num_heads"]))
        self.head = Head(d, int(spec["head"]["hidden_dim"]), int(spec["head"]["num_classes"]), float(spec["head"]["dropout"]))

    def _tokens(self, pixel_values):
        return self.vision_model(pixel_values).last_hidden_state[:, self.patch_start :, :]

    def forward(self, pixel_values):
        return self.head(self.pool(self._tokens(pixel_values)))


class Task2Model:
    def __init__(self, nets, spec):
        self.nets = list(nets)
        self.size = int(spec["img_size"])
        self.thr = float(spec["threshold"])
        p = next(self.nets[0].parameters())
        self.device, self.dtype = p.device, p.dtype

    @torch.no_grad()
    def predict_batch(self, paths):
        x = torch.stack([
            to_input(png_to_image(p), self.size, IMAGENET_MEAN, IMAGENET_STD, self.device, self.dtype)
            for p in paths
        ])
        logits = []
        for net in self.nets:
            logits.append(net(x).reshape(-1).float())
            logits.append(net(torch.flip(x, dims=[-1])).reshape(-1).float())
        probs = torch.sigmoid(torch.stack(logits).mean(0)).cpu().numpy()
        return [ID2LABEL[int(pr >= self.thr)] for pr in probs]


def load_model(weights_path=DEFAULT_WEIGHTS, device="auto", config_path=DEFAULT_CONFIG):
    from safetensors.torch import load_file

    paths = [Path(weights_path) / f"model_{i}.safetensors" for i in range(4)]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing model weights: {path}")
    spec = load_config(config_path)["cls_spec"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
    nets = []
    for path in paths:
        net = TBModel(spec)
        net.load_state_dict(load_file(str(path)), strict=True)
        nets.append(net.to(dev).eval())
        print(f"[weights] loaded {path.name}", flush=True)
    return Task2Model(nets, spec)


def discover_images(input_path):
    path = Path(input_path)
    if path.is_file():
        if path.suffix.lower() != ".png":
            raise ValueError(f"Expected a PNG file: {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {path}")
    paths = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".png")
    if not paths:
        raise ValueError(f"No PNG files found in: {path}")
    return paths


def predict(input_path, output_dir, weights_path=DEFAULT_WEIGHTS, batch_size=32, device="auto", config_path=DEFAULT_CONFIG):
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    paths = discover_images(input_path)
    model = load_model(weights_path, device=device, config_path=config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(0, len(paths), batch_size):
        batch = paths[i:i + batch_size]
        rows.extend((path.name, label) for path, label in zip(batch, model.predict_batch(batch)))
    with (output_dir / "prediction.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "TB/Normal"])
        writer.writerows(rows)
    print(f"[done] wrote {len(rows)} predictions -> {output_dir}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description="MultimodalAI: TB/Normal classification")
    parser.add_argument("--input", default="/input", help="PNG file or directory of PNG files")
    parser.add_argument("--output", default="/output", help="output directory")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="directory containing model_0.safetensors through model_3.safetensors")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="config.json holding model and threshold settings")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    args = parser.parse_args()
    predict(args.input, args.output, args.weights, args.batch_size, args.device, args.config)


if __name__ == "__main__":
    main()

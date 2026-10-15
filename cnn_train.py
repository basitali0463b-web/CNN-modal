"""CNN image classifier (PyTorch).

Real data   : python cnn_train.py --data_dir path/to/images --epochs 15
              (layout: path/to/images/<class_name>/*.jpg)
Smoke test  : python cnn_train.py --synthetic --epochs 3
              (auto-generates circles / squares / triangles - no download needed)
Predict     : python cnn_train.py --predict some.jpg
"""
import argparse, json, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

IMG = 64
MEAN, STD = [0.5] * 3, [0.5] * 3
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SyntheticShapes(Dataset):
    """Tiny offline dataset so the pipeline can be tested without any images."""
    classes = ["circle", "square", "triangle"]

    def __init__(self, n=900, tf=None, seed=0):
        self.n, self.tf, self.rng = n, tf, random.Random(seed)
        self.items = [(self.rng.randrange(3), self.rng.random()) for _ in range(n)]

    def __len__(self): return self.n

    def __getitem__(self, i):
        label, _ = self.items[i]
        r = random.Random(i)
        im = Image.new("RGB", (IMG, IMG), tuple(r.randrange(0, 60) for _ in range(3)))
        d, c, s = ImageDraw.Draw(im), tuple(r.randrange(150, 256) for _ in range(3)), r.randrange(12, 22)
        x, y = r.randrange(s, IMG - s), r.randrange(s, IMG - s)
        if label == 0: d.ellipse([x - s, y - s, x + s, y + s], fill=c)
        elif label == 1: d.rectangle([x - s, y - s, x + s, y + s], fill=c)
        else: d.polygon([(x, y - s), (x - s, y + s), (x + s, y + s)], fill=c)
        return (self.tf(im) if self.tf else im), label


class CNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        def block(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128))   # 64 -> 8
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.4), nn.Linear(128 * 8 * 8, 128), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, x): return self.head(self.features(x))


def evaluate(model, loader):
    model.eval(); correct = total = 0; preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            p = model(x.to(DEV)).argmax(1).cpu()
            correct += (p == y).sum().item(); total += len(y); preds += p.tolist(); labels += y.tolist()
    return correct / total, preds, labels


def train(args):
    torch.manual_seed(42); random.seed(42)
    train_tf = transforms.Compose([transforms.Resize((IMG, IMG)), transforms.RandomHorizontalFlip(),
                                   transforms.RandomRotation(10), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    eval_tf = transforms.Compose([transforms.Resize((IMG, IMG)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    if args.synthetic:
        full_tr, full_ev, classes = SyntheticShapes(tf=train_tf), SyntheticShapes(tf=eval_tf), SyntheticShapes.classes
    else:
        full_tr, full_ev = datasets.ImageFolder(args.data_dir, train_tf), datasets.ImageFolder(args.data_dir, eval_tf)
        classes = full_tr.classes
    n = len(full_tr); n_tr, n_va = int(.7 * n), int(.15 * n)
    g = torch.Generator().manual_seed(42)
    idx_tr, idx_va, idx_te = random_split(range(n), [n_tr, n_va, n - n_tr - n_va], generator=g)
    sub = lambda ds, idx: torch.utils.data.Subset(ds, list(idx))
    tr = DataLoader(sub(full_tr, idx_tr), batch_size=args.bs, shuffle=True)
    va = DataLoader(sub(full_ev, idx_va), batch_size=args.bs)
    te = DataLoader(sub(full_ev, idx_te), batch_size=args.bs)

    model = CNN(len(classes)).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf, best, best_state = nn.CrossEntropyLoss(), 0, None
    for ep in range(1, args.epochs + 1):
        model.train(); tot = 0
        for x, y in tr:
            x, y = x.to(DEV), y.to(DEV); opt.zero_grad(); l = lossf(model(x), y); l.backward(); opt.step(); tot += l.item() * len(y)
        acc, _, _ = evaluate(model, va)
        print(f"epoch {ep:02d}  train_loss={tot / len(tr.dataset):.4f}  val_acc={acc:.3f}")
        if acc > best: best, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    acc, p, y = evaluate(model, te)
    print(f"\nTEST accuracy: {acc:.3f}")
    from sklearn.metrics import classification_report, confusion_matrix
    print(classification_report(y, p, target_names=classes, zero_division=0)); print(confusion_matrix(y, p))
    torch.save(model.state_dict(), "cnn_model.pt"); json.dump(classes, open("cnn_classes.json", "w"))
    print("Saved cnn_model.pt, cnn_classes.json")


def predict(path):
    classes = json.load(open("cnn_classes.json")); model = CNN(len(classes)).to(DEV)
    model.load_state_dict(torch.load("cnn_model.pt", map_location=DEV)); model.eval()
    tf = transforms.Compose([transforms.Resize((IMG, IMG)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    with torch.no_grad(): pr = torch.softmax(model(tf(Image.open(path).convert("RGB"))[None].to(DEV)), 1)[0]
    for c, p in sorted(zip(classes, pr.tolist()), key=lambda t: -t[1]): print(f"{c:20s} {p:.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir"); ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--epochs", type=int, default=10); ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--predict")
    a = ap.parse_args()
    if a.predict: predict(a.predict)
    elif a.synthetic or a.data_dir: train(a)
    else: ap.error("give --data_dir, --synthetic or --predict")

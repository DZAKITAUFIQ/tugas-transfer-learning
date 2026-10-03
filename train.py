"""
Transfer learning ResNet-18: bandingkan 3 mode
  feature : backbone beku, hanya head (fc) dilatih
  partial : layer4 + fc dilatih
  scratch : semua dilatih dari bobot acak (pembanding)

Struktur data yang dibutuhkan:
  dataset_raw/
    kelas_a/ foto1.jpg foto2.jpg ...
    kelas_b/ ...

Jalankan:  python train.py
"""
import csv
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

# ====== PENGATURAN (ubah seperlunya) ======
DATA_DIR = "dataset_raw"
EPOCHS = 10
BATCH = 16
VAL_RATIO = 0.2
SEED = 42
MODES = ["feature", "partial", "scratch"]
# ==========================================

MEAN = [0.485, 0.456, 0.406]  # statistik ImageNet
STD = [0.229, 0.224, 0.225]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_loaders():
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    base = datasets.ImageFolder(DATA_DIR)
    by_class = {}
    for i, (_, c) in enumerate(base.samples):
        by_class.setdefault(c, []).append(i)

    rng = random.Random(SEED)
    train_idx, val_idx = [], []
    for idxs in by_class.values():
        rng.shuffle(idxs)
        n_val = max(1, int(len(idxs) * VAL_RATIO))
        val_idx += idxs[:n_val]
        train_idx += idxs[n_val:]

    train_ds = Subset(datasets.ImageFolder(DATA_DIR, transform=train_tf), train_idx)
    val_ds = Subset(datasets.ImageFolder(DATA_DIR, transform=val_tf), val_idx)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)
    return train_dl, val_dl, base.classes, len(train_idx), len(val_idx)


def build_model(mode, n_classes):
    if mode == "scratch":
        m = models.resnet18(weights=None)
    else:
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, n_classes)  # head baru

    if mode == "feature":
        for p in m.parameters():
            p.requires_grad = False
        for p in m.fc.parameters():
            p.requires_grad = True
        groups = [{"params": m.fc.parameters(), "lr": 1e-3}]
    elif mode == "partial":
        for p in m.parameters():
            p.requires_grad = False
        for p in m.layer4.parameters():
            p.requires_grad = True
        for p in m.fc.parameters():
            p.requires_grad = True
        groups = [
            {"params": m.layer4.parameters(), "lr": 1e-4},
            {"params": m.fc.parameters(), "lr": 1e-3},
        ]
    else:  # scratch
        groups = [{"params": m.parameters(), "lr": 1e-3}]

    return m.to(DEVICE), torch.optim.Adam(groups)


def set_train_mode(m, mode):
    # Lapisan beku tetap eval() agar statistik BatchNorm ImageNet tidak berubah
    if mode == "scratch":
        m.train()
    elif mode == "feature":
        m.eval()
        m.fc.train()
    else:
        m.eval()
        m.layer4.train()
        m.fc.train()


def run_epoch(m, dl, loss_fn, opt=None, mode=None):
    training = opt is not None
    if training:
        set_train_mode(m, mode)
    else:
        m.eval()
    correct, total = 0, 0
    with torch.set_grad_enabled(training):
        for x, y in dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = m(x)
            if training:
                loss = loss_fn(out, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


def train_mode(mode, train_dl, val_dl, n_classes):
    torch.manual_seed(SEED)
    m, opt = build_model(mode, n_classes)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.CrossEntropyLoss()

    hist = {"train_acc": [], "val_acc": []}
    best, epoch_90 = 0.0, None
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        tr = run_epoch(m, train_dl, loss_fn, opt, mode)
        va = run_epoch(m, val_dl, loss_fn)
        sched.step()
        hist["train_acc"].append(tr)
        hist["val_acc"].append(va)
        if va > best:
            best = va
            Path("models").mkdir(exist_ok=True)
            torch.save(m.state_dict(), f"models/resnet18_{mode}.pt")
        if epoch_90 is None and va >= 0.90:
            epoch_90 = ep
        print(f"[{mode}] epoch {ep}/{EPOCHS}  train={tr:.3f}  val={va:.3f}")
    return {
        "mode": mode,
        "best_val_acc": best,
        "train_time_s": time.time() - t0,
        "epoch_val_90": epoch_90,
        "history": hist,
    }


def main():
    print("Device:", DEVICE)
    train_dl, val_dl, classes, n_tr, n_va = build_loaders()
    print(f"Kelas: {classes} | train={n_tr} val={n_va}")

    results = [train_mode(md, train_dl, val_dl, len(classes)) for md in MODES]

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open("results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "best_val_acc", "train_time_s", "epoch_val_90"])
        for r in results:
            w.writerow([r["mode"], f"{r['best_val_acc']:.4f}",
                        f"{r['train_time_s']:.1f}", r["epoch_val_90"] or "-"])

    plt.figure(figsize=(7, 4))
    for r in results:
        plt.plot(range(1, EPOCHS + 1), r["history"]["val_acc"], marker="o", label=r["mode"])
    plt.xlabel("Epoch")
    plt.ylabel("Akurasi validasi")
    plt.title("Akurasi validasi per epoch (ResNet-18)")
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("accuracy_per_epoch.png", dpi=150)

    print("\n| Mode | Akurasi val terbaik | Waktu latih (s) | Epoch akurasi >= 90% |")
    print("|---|---|---|---|")
    for r in results:
        print(f"| {r['mode']} | {r['best_val_acc']*100:.1f}% | "
              f"{r['train_time_s']:.0f} | {r['epoch_val_90'] or '-'} |")


if __name__ == "__main__":
    main()

import argparse, json, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from dataset import FER2013
from models import get_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="baseline")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--root", default="../data")      # folder containing train/ and test/
    p.add_argument("--class_weights", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # reproducibility
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    train_ds = FER2013(args.root, "train", train=True)
    val_ds = FER2013(args.root, "val")
    test_ds = FER2013(args.root, "test")
    train_dl = DataLoader(train_ds, args.bs, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, args.bs, num_workers=2)
    test_dl = DataLoader(test_ds, args.bs, num_workers=2)

    model = get_model(args.model).to(device)

    weight = None
    if args.class_weights:
        counts = np.bincount(train_ds.labels, minlength=7)
        weight = torch.tensor(counts.sum() / (7 * counts), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=0.1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    @torch.no_grad()
    def evaluate(dl):
        model.eval()
        preds, labels = [], []
        for x, y in dl:
            preds.append(model(x.to(device)).argmax(1).cpu())
            labels.append(y)
        preds, labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()
        return (preds == labels).mean(), f1_score(labels, preds, average="macro")

    best_val, patience, bad = 0.0, 15, 0
    history = []
    ckpt = f"../checkpoints/{args.model}.pt"

    for epoch in range(args.epochs):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
        scheduler.step()

        val_acc, val_f1 = evaluate(val_dl)
        history.append(dict(epoch=epoch, loss=loss_sum / total, train_acc=correct / total,
                            val_acc=float(val_acc), val_f1=float(val_f1)))
        print(f"ep {epoch+1:03d} loss {loss_sum/total:.3f} train {correct/total:.3f} "
              f"val {val_acc:.3f} f1 {val_f1:.3f}")

        if val_acc > best_val:
            best_val, bad = val_acc, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= patience:
                print("early stopping"); break

    # final test with the best-validation checkpoint
    model.load_state_dict(torch.load(ckpt, map_location=device))
    test_acc, test_f1 = evaluate(test_dl)
    print(f"\nFINAL {args.model}: PrivateTest acc {test_acc:.4f}, macro-F1 {test_f1:.4f}")

    with open(f"../results/{args.model}_metrics.json", "w") as f:
        json.dump(dict(model=args.model, test_acc=float(test_acc), test_f1=float(test_f1),
                       best_val=float(best_val), history=history), f, indent=2)


if __name__ == "__main__":
    main()
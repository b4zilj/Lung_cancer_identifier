import argparse, os, json, yaml
import torch, torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as T
from src.model import build_model
import numpy as np

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    img_size = 224
    tfm = T.Compose([T.Resize((img_size, img_size)), T.ToTensor(),
                     T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
    ds = ImageFolder(args.test, transform=tfm)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)

    ckpt = torch.load(args.model, map_location="cpu")
    classes = ckpt.get("classes", ds.classes)
    model = build_model("resnet18", num_classes=len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for x, y in dl:
            logits = model(x)
            probs = logits.softmax(dim=1)[:,1].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            y_true.extend(y.numpy()); y_pred.extend(preds); y_prob.extend(probs)

    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = None

    
    metrics = {"report": report, "auc": auc}
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.xlabel("Predicted"); plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    fig.tight_layout()
    figpath = os.path.join(args.out, "confusion_matrix.png")
    plt.savefig(figpath, dpi=200)
    print("Saved:", figpath)
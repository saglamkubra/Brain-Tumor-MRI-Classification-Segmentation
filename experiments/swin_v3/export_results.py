"""Export numeric artifacts for a completed, validation-selected Swin V3 run."""
import argparse
import csv
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

BASELINE = np.array([[317,15,49,2],[2,367,22,5],[0,0,400,0],[0,2,0,398]])
REPORTED_FINE = np.array([[314,25,39,5],[0,374,15,7],[0,0,400,0],[0,2,0,398]])


def matrix_stats(matrix):
    labels = range(len(matrix))
    actual, predicted = [], []
    for row, column in np.ndindex(matrix.shape):
        actual.extend([row] * int(matrix[row, column]))
        predicted.extend([column] * int(matrix[row, column]))
    f1 = precision_recall_fscore_support(actual, predicted, labels=list(labels), zero_division=0)[2]
    return np.trace(matrix) / matrix.sum(), matrix[0,0] / matrix[0].sum(), int(matrix[0,2]), f1.mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--run-label", default="yerel çalışma", help="CSV ve grafiklerde kullanılacak çalışma etiketi.")
    args = parser.parse_args()
    out = args.run
    data = json.loads((out/"metrics.json").read_text(encoding="utf-8"))
    epochs = data["epochs"]
    model = np.array(data["test"]["confusion_matrix"])
    matrices = [BASELINE, REPORTED_FINE, model]
    names = ["V3 · GitHub / paylaşılan", "Fine-tuned · paylaşılan", f"Fine-tuned · {args.run_label}"]
    labels = data["metadata"]["class_names"]
    with (out/"comparison.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run","accuracy","glioma_recall","glioma_to_notumor","macro_f1"])
        writer.writerows([name,*matrix_stats(matrix)] for name,matrix in zip(names,matrices))
    with (out/"training_history.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        keys = ["loss","accuracy","glioma_recall","glioma_to_notumor"]
        writer.writerow(["epoch","split",*keys])
        for epoch in epochs:
            for split in ("train","validation"):
                writer.writerow([epoch["epoch"],split,*[epoch[split][key] for key in keys]])
    with (out/"classification_report.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        keys = ["precision","recall","f1-score","support"]
        writer.writerow(["class",*keys])
        for name,stats in data["test"]["classification"].items():
            if isinstance(stats,dict):
                writer.writerow([name,*[stats[key] for key in keys]])
    fig, axes = plt.subplots(1,2,figsize=(9,3.8),layout="constrained")
    for ax, metric, title in zip(axes,["loss","accuracy"],["Çapraz entropi", "Doğruluk (%)"]):
        for split,label in (("train","Eğitim"),("validation","Doğrulama")):
            values=[e[split][metric] *(100 if metric=="accuracy" else 1) for e in epochs]
            ax.plot([e["epoch"] for e in epochs],values,marker="o",label=label)
        ax.set(title=title,xlabel="Epoch",xticks=[e["epoch"] for e in epochs])
        ax.legend()
    fig.suptitle(f"Swin-Tiny V3 · {args.run_label}")
    fig.savefig(out/"training_curves.png",dpi=155)
    plt.close(fig)
    fig, axes = plt.subplots(1,3,figsize=(12,4.4),layout="constrained")
    for name, matrix, ax in zip(names,matrices,axes):
        im=ax.imshow(matrix,vmin=0,vmax=400,cmap="Blues")
        for row,column in np.ndindex(matrix.shape):
            value=matrix[row,column]
            ax.text(column,row,str(value),ha="center",va="center",color="white" if value>220 else "#123")
        ax.set(title=name,xticks=range(4),yticks=range(4),xticklabels=labels,yticklabels=labels,xlabel="Tahmin")
        ax.tick_params(axis="x", labelrotation=25)
    axes[0].set_ylabel("Gerçek etiket")
    fig.colorbar(im,ax=axes,label="Görüntü",shrink=.8)
    fig.savefig(out/"confusion_comparison.png",dpi=155)
    plt.close(fig)


if __name__ == "__main__":
    main()

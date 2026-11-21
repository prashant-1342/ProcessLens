import os
from pathlib import Path
from typing import Any


def compute_ece(
    confidences: list[float],
    accuracies: list[int],
    num_bins: int = 10,
) -> tuple[float, list[dict[str, Any]]]:
    if not confidences or len(confidences) != len(accuracies):
        return 0.0, []

    total_samples = len(confidences)
    bin_size = 1.0 / num_bins
    bin_data = []
    total_ece = 0.0

    for i in range(num_bins):
        bin_lower = i * bin_size
        bin_upper = (i + 1) * bin_size

        in_bin_indices = [
            idx for idx, c in enumerate(confidences)
            if (bin_lower <= c < bin_upper) or (i == num_bins - 1 and bin_lower <= c <= bin_upper)
        ]

        count = len(in_bin_indices)
        if count > 0:
            avg_confidence = sum(confidences[idx] for idx in in_bin_indices) / count
            avg_accuracy = sum(accuracies[idx] for idx in in_bin_indices) / count
            bin_weight = count / total_samples
            bin_error = abs(avg_accuracy - avg_confidence)
            total_ece += bin_weight * bin_error
        else:
            avg_confidence = 0.0
            avg_accuracy = 0.0
            bin_error = 0.0

        bin_data.append({
            "bin_index": i,
            "range": (round(bin_lower, 2), round(bin_upper, 2)),
            "count": count,
            "avg_confidence": round(avg_confidence, 4),
            "avg_accuracy": round(avg_accuracy, 4),
            "error": round(bin_error, 4),
        })

    return round(total_ece, 4), bin_data


def generate_reliability_diagram(
    confidences: list[float],
    accuracies: list[int],
    output_path: str = "data_output/reliability_diagram.png",
    num_bins: int = 10,
) -> float:
    ece, bin_data = compute_ece(confidences, accuracies, num_bins=num_bins)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 6))

        bin_centers = [(b["range"][0] + b["range"][1]) / 2 for b in bin_data]
        acc_values = [b["avg_accuracy"] for b in bin_data]
        conf_values = [b["avg_confidence"] for b in bin_data]
        counts = [b["count"] for b in bin_data]

        width = 1.0 / num_bins * 0.8
        ax.bar(
            bin_centers,
            acc_values,
            width=width,
            color="#2563eb",
            alpha=0.7,
            edgecolor="#1d4ed8",
            label="Observed Accuracy",
        )

        ax.plot([0, 1], [0, 1], linestyle="--", color="#dc2626", label="Perfect Calibration")

        for bc, acc, count in zip(bin_centers, acc_values, counts):
            if count > 0:
                ax.annotate(
                    f"n={count}",
                    (bc, acc),
                    textcoords="offset points",
                    xytext=(0, 5),
                    ha="center",
                    fontsize=8,
                )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Reliability Diagram (ECE = {ece:.4f})")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper left")

        plt.tight_layout()
        plt.savefig(str(out_file), dpi=300)
        plt.close(fig)

    except Exception:
        pass

    return ece

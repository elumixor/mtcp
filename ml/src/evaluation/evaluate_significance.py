import torch
import matplotlib.pyplot as plt
import wandb

from src.data import Data
from src.nn import Model

from .find_significance_threshold import find_significance_threshold


def evaluate_significance(model: Model, val: Data, F: int, batch_size: int, device="cpu", wandb_run=None):
    plt.rcParams.update({'font.size': 22})

    with torch.no_grad():
        logits = torch.cat([model(batch.to(device)) for batch in val.batches(batch_size=batch_size, shuffle=False)]).cpu()

    # Probabilities
    probs = torch.softmax(logits, dim=1)

    # Now let's check all the thresholds where the significance is the highest
    threshold, significance, stats = find_significance_threshold(
        probs,
        val.y,
        val.w,
        val.y_names.index("ttH"),
        F,
        n_significance_thresholds=1000,
        return_stats=True,
        min_background=1,
    )

    threshold_simple, significance_simple, stats_simple = find_significance_threshold(
        probs,
        val.y,
        val.w,
        val.y_names.index("ttH"),
        F,
        n_significance_thresholds=1000,
        return_stats=True,
        include_signal=False,
        min_background=1,
    )

    print(f"Maximum significance: {significance} at threshold {threshold}")

    # Let's also calculate the best possible significance we can get on the validation set. This happens if we classify all events correctly
    # S_max = trn.max_significance()

    # Plot two figures one below another
    # First sub-figure:
    fig, ax1 = plt.subplots(1, 1, figsize=(9, 8))

    # Plot the (expected) number of signal and background events
    thresholds, significances, signals, backgrounds = zip(*[[si.cpu() for si in s] for s in stats])
    thresholds_simple, significances_simple, *_ = zip(*[[si.cpu() for si in s] for s in stats_simple])

    ax1.plot(thresholds, backgrounds, label=f"Background ($b$)")
    ax1.plot(thresholds, signals, label=f"Signal ($s$)")
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Expected events")

    threshold = threshold.cpu()
    significance = significance.cpu()
    threshold_simple = threshold_simple.cpu()
    significance_simple = significance_simple.cpu()

    # Plot the dashed vertical line at the best threshold and annotate it
    ax1.axvline(threshold, linestyle="--", color="black")
    ax1.annotate(f"threshold = ${threshold:.3f}$", (threshold, 0.5), (threshold + 0.05, 0.5), arrowprops=dict(arrowstyle="->"))

    ax1.axvline(threshold_simple, linestyle="--", color="black")
    ax1.annotate(f"threshold = ${threshold_simple:.3f}$", (threshold_simple, 0.5), (threshold_simple + 0.05, 0.5), arrowprops=dict(arrowstyle="->"))

    # Plot points at the number of signal and background events at the best threshold and annotate them
    # i = thresholds.index(threshold)
    # ax1.scatter(threshold, signals[i], color="C0", zorder=10)
    # ax1.annotate(f"$s={signals[threshold]:.3f}$", (thresholds[threshold], signals[threshold]), (thresholds[threshold] + 0.02, signals[threshold] + 0.5))
    # ax1.scatter(thresholds[threshold], backgrounds[threshold], color="C1", zorder=10)
    # ax1.annotate(f"$b={backgrounds[threshold]:.3f}$", (thresholds[threshold], backgrounds[threshold]), (thresholds[threshold] - 0.2, backgrounds[threshold] - 0))
    ax1.legend()

    # Second subfigure
    fig, ax2 = plt.subplots(1, 1, figsize=(9, 8))

    # Plot the significance
    ax2.plot(thresholds, significances, label=r"$\frac{s}{\sqrt{s+b}}$")
    ax2.plot(thresholds_simple, significances_simple, label=r"$\frac{s}{\sqrt{b}}$")

    # Plot the dashed vertical line at the best threshold and annotate it
    ax2.axvline(threshold, linestyle="--", color="black")
    ax2.annotate(f"threshold = ${threshold:.3f}$", (threshold, 0.5), (threshold + 0.05, 0.5), arrowprops=dict(arrowstyle="->"))

    ax2.axvline(threshold_simple, linestyle="--", color="black")
    ax2.annotate(f"threshold = ${threshold_simple:.3f}$", (threshold_simple, 0.5), (threshold_simple + 0.05, 0.5), arrowprops=dict(arrowstyle="->"))

    # Plot points at the significance at the best threshold and annotate it
    ax2.scatter(threshold, significance, color="C0", zorder=10)
    ax2.annotate(f"$S={significance:.3f}$", (threshold, significance), (threshold + 0.03, significance - 0.0))

    ax2.scatter(threshold_simple, significance_simple, color="C1", zorder=10)
    ax2.annotate(f"$S={significance_simple:.3f}$", (threshold_simple, significance_simple), (threshold_simple + 0.03, significance_simple - 0.0))

    ax2.legend(loc="lower left")
    ax2.set_xlabel("Threshold")
    ax2.set_ylabel("Significance");

    if wandb_run is not None:
        # Log the image
        wandb.log({"significance/thresholds": wandb.Image(fig)})

    # Return the thresholds 
    return threshold, threshold_simple
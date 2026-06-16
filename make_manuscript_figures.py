from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch


ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT / "outputs"
PAPER_DIR = ROOT / "_paper_extract"


INK = "#26384D"
MUTED = "#5B6F86"
GRID = "#D9E0E8"
EDGE = "#BCC9D8"
PANEL_BG = "#F7F9FC"
BASELINE = "#4D678C"
GROUP = "#C24C4C"
TEXT_ONLY = "#6785A1"
BEHAVIOR = "#C18359"
TEXT_BEHAVIOR = "#43864A"
NO_PICTURE = "#9A7D72"
RAW_PICTURE = "#6079D8"
PICTURE_CONTEXT = "#3AA39A"
COMBINED_RAW = "#5E7E8D"
COMBINED_CONTEXT = "#3D7B5C"
HIGHLIGHT_NEG = "#2F6FA5"
HIGHLIGHT_POS = "#D38642"
COEFF_MUTED = "#9EB3C7"


FEATURE_LABELS = {
    "text_only": "Text only",
    "behavior_only": "Behavior only",
    "combined_no_haspicture": "Text + behavior",
    "picture_only_raw": "Raw picture only",
    "picture_only_context": "Picture-context only",
    "combined_raw_haspicture": "Combined + raw picture",
    "combined_pic_context": "Combined + picture-context",
}


COEFF_LABELS = {
    "HasPicture": "HasPicture",
    "dup_breadth": "Duplicate breadth",
    "dup_mass": "Duplicate mass",
    "restaurant_degree": "Restaurant degree",
    "user_degree": "User degree",
    "user_med_iri_hours": "Median IRI (hours)",
    "user_iqr_iri_hours": "IQR IRI (hours)",
    "wc": "Word count",
    "char_len": "Character length",
    "multi_poster": "Multi-poster",
    "burst_z": "Burst z-score",
    "ttr": "Type-token ratio",
}


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(colors=INK)
    ax.set_axisbelow(True)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)


def panel_label(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.0,
        1.03,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color=INK,
    )


def save_figure(fig: plt.Figure, name: str, *, dpi: int = 220) -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_DIR / name, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_multi_format(fig: plt.Figure, stem: str) -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_DIR / f"{stem}.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(PAPER_DIR / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(PAPER_DIR / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def annotate_vertical_bars(ax: plt.Axes, bars, values: list[float], *, pad: float = 0.01) -> None:
    ymin, ymax = ax.get_ylim()
    offset = (ymax - ymin) * pad
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + offset,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=INK,
        )


def annotate_horizontal_bars(ax: plt.Axes, bars, values: list[float], *, pad: float = 0.012) -> None:
    xmin, xmax = ax.get_xlim()
    offset = (xmax - xmin) * pad
    for bar, value in zip(bars, values):
        ax.text(
            value + offset,
            bar.get_y() + bar.get_height() / 2.0,
            f"{value:.3f}",
            ha="left",
            va="center",
            fontsize=10,
            color=INK,
        )


def error_from_ci(values: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.vstack([values - low, high - values])


def read_protocol_csv(protocol: str, name: str) -> pd.DataFrame:
    return pd.read_csv(OUTPUTS_DIR / protocol / name)


def draw_feature_family_comparison() -> None:
    set_style()
    files = {
        "baseline": read_protocol_csv("baseline", "text_behavior_combined.csv"),
        "group": read_protocol_csv("group", "text_behavior_combined.csv"),
    }
    order = ["text_only", "behavior_only", "combined_no_haspicture"]
    colors = [TEXT_ONLY, BEHAVIOR, TEXT_BEHAVIOR]

    fig, axes = plt.subplots(1, 2, figsize=(11.3, 4.5), sharey=True)
    for ax, (protocol, df) in zip(axes, files.items()):
        values = [float(df.loc[df["feature_set"] == key, "pr_auc"].iloc[0]) for key in order]
        labels = [FEATURE_LABELS[key] for key in order]
        bars = ax.bar(np.arange(len(order)), values, width=0.62, color=colors, edgecolor="white", linewidth=0.9)
        style_axis(ax, grid_axis="y")
        panel_label(ax, "Baseline split" if protocol == "baseline" else "Group-aware split")
        ax.set_xticks(np.arange(len(order)), labels)
        ax.tick_params(axis="x", rotation=12)
        ax.set_ylim(0.0, 0.55)
        annotate_vertical_bars(ax, bars, values, pad=0.014)
    axes[0].set_ylabel("PR-AUC")
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.21, top=0.88, wspace=0.08)
    save_figure(fig, "feature_family_comparison.png")


def draw_picture_signal_comparison() -> None:
    set_style()
    files = {
        "baseline": read_protocol_csv("baseline", "haspicture_ablation.csv"),
        "group": read_protocol_csv("group", "haspicture_ablation.csv"),
    }
    files["baseline"] = pd.concat(
        [
            files["baseline"],
            read_protocol_csv("baseline", "haspicture_vs_piccontext.csv").query("feature_set in ['picture_only_raw', 'picture_only_context']"),
        ],
        ignore_index=True,
    )
    files["group"] = pd.concat(
        [
            files["group"],
            read_protocol_csv("group", "haspicture_vs_piccontext.csv").query("feature_set in ['picture_only_raw', 'picture_only_context']"),
        ],
        ignore_index=True,
    )
    order = [
        "combined_no_haspicture",
        "picture_only_raw",
        "picture_only_context",
        "combined_raw_haspicture",
        "combined_pic_context",
    ]
    ordered_keys = list(reversed(order))
    color_map = {
        "combined_no_haspicture": NO_PICTURE,
        "picture_only_raw": RAW_PICTURE,
        "picture_only_context": PICTURE_CONTEXT,
        "combined_raw_haspicture": COMBINED_RAW,
        "combined_pic_context": COMBINED_CONTEXT,
    }
    label_map = dict(FEATURE_LABELS)
    label_map["combined_no_haspicture"] = "No picture"
    labels = [label_map[key] for key in ordered_keys]
    y = np.arange(len(ordered_keys))

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), sharey=True)
    for ax, (protocol, df) in zip(axes, files.items()):
        values = [float(df.loc[df["feature_set"] == key, "pr_auc"].iloc[0]) for key in ordered_keys]
        bars = ax.barh(
            y,
            values,
            height=0.68,
            color=[color_map[key] for key in ordered_keys],
            edgecolor="white",
            linewidth=0.9,
        )
        style_axis(ax, grid_axis="x")
        panel_label(ax, "Baseline split" if protocol == "baseline" else "Group-aware split")
        ax.set_xlim(0.0, 1.02)
        ax.set_xlabel("PR-AUC")
        ax.set_yticks(y, labels)
        annotate_horizontal_bars(ax, bars, values, pad=0.012)
    axes[1].tick_params(axis="y", labelleft=False)
    fig.subplots_adjust(left=0.21, right=0.99, bottom=0.13, top=0.90, wspace=0.10)
    save_figure(fig, "picture_signal_comparison.png")


def draw_split_effect_comparison() -> None:
    set_style()
    split_df = pd.read_csv(OUTPUTS_DIR / "tables" / "fixed_model_split_effect.csv")
    boot_df = pd.read_csv(OUTPUTS_DIR / "tables" / "bootstrap_ci.csv")
    boot_df = boot_df.loc[boot_df["feature_set"] == "combined_pic_context"].copy()
    merged = split_df.merge(boot_df, on=["protocol", "feature_set"], how="left", suffixes=("", "_boot"))

    merged = merged.set_index("protocol").loc[["baseline", "group"]].reset_index()
    colors = [BASELINE, GROUP]

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), gridspec_kw={"width_ratios": [1.0, 1.18]})

    pr_vals = merged["pr_auc"].to_numpy(dtype=float)
    pr_low = merged["pr_auc_ci_low"].to_numpy(dtype=float)
    pr_high = merged["pr_auc_ci_high"].to_numpy(dtype=float)
    pr_err = error_from_ci(pr_vals, pr_low, pr_high)
    x0 = np.arange(2)
    bars0 = axes[0].bar(
        x0,
        pr_vals,
        width=0.62,
        color=colors,
        edgecolor="white",
        linewidth=0.9,
        yerr=pr_err,
        capsize=3,
        error_kw={"elinewidth": 1.0, "ecolor": MUTED},
    )
    style_axis(axes[0], grid_axis="y")
    axes[0].set_xticks(x0, ["Baseline", "Group-aware"])
    axes[0].set_ylabel("PR-AUC")
    axes[0].set_ylim(0.90, 1.01)
    annotate_vertical_bars(axes[0], bars0, list(pr_vals), pad=0.012)

    metrics = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]
    x1 = np.arange(len(metrics))
    width = 0.28
    base_vals = np.array([merged.loc[merged["protocol"] == "baseline", key].iloc[0] for key, _ in metrics], dtype=float)
    group_vals = np.array([merged.loc[merged["protocol"] == "group", key].iloc[0] for key, _ in metrics], dtype=float)
    base_low = np.array([merged.loc[merged["protocol"] == "baseline", f"{key}_ci_low"].iloc[0] for key, _ in metrics], dtype=float)
    base_high = np.array([merged.loc[merged["protocol"] == "baseline", f"{key}_ci_high"].iloc[0] for key, _ in metrics], dtype=float)
    group_low = np.array([merged.loc[merged["protocol"] == "group", f"{key}_ci_low"].iloc[0] for key, _ in metrics], dtype=float)
    group_high = np.array([merged.loc[merged["protocol"] == "group", f"{key}_ci_high"].iloc[0] for key, _ in metrics], dtype=float)
    bars1 = axes[1].bar(
        x1 - width / 2,
        base_vals,
        width=width,
        color=BASELINE,
        edgecolor="white",
        linewidth=0.9,
        yerr=error_from_ci(base_vals, base_low, base_high),
        capsize=3,
        error_kw={"elinewidth": 1.0, "ecolor": MUTED},
        label="Baseline split",
    )
    bars2 = axes[1].bar(
        x1 + width / 2,
        group_vals,
        width=width,
        color=GROUP,
        edgecolor="white",
        linewidth=0.9,
        yerr=error_from_ci(group_vals, group_low, group_high),
        capsize=3,
        error_kw={"elinewidth": 1.0, "ecolor": MUTED},
        label="Group-aware split",
    )
    style_axis(axes[1], grid_axis="y")
    axes[1].set_xticks(x1, [label for _, label in metrics])
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.90, 1.01)
    annotate_vertical_bars(axes[1], bars1, list(base_vals), pad=0.009)
    annotate_vertical_bars(axes[1], bars2, list(group_vals), pad=0.009)

    legend_handles = [
        Patch(facecolor=BASELINE, edgecolor="none", label="Baseline split"),
        Patch(facecolor=GROUP, edgecolor="none", label="Group-aware split"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        fontsize=10,
    )
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.17, top=0.82, wspace=0.18)
    save_figure(fig, "split_effect_fixed_model.png")


def draw_picture_context_coefficients() -> None:
    set_style()
    base_df = pd.read_csv(OUTPUTS_DIR / "baseline" / "picture_context_coefficients.csv")
    group_df = pd.read_csv(OUTPUTS_DIR / "group" / "picture_context_coefficients.csv")
    merged = base_df[["feature", "logit_coef"]].merge(
        group_df[["feature", "logit_coef"]],
        on="feature",
        suffixes=("_baseline", "_group"),
    )
    merged["abs_max"] = merged[["logit_coef_baseline", "logit_coef_group"]].abs().max(axis=1)
    order = (
        merged.sort_values(["logit_coef_baseline", "abs_max"], ascending=[True, True])["feature"].tolist()
    )
    display_labels = [COEFF_LABELS.get(feature, feature) for feature in order]
    common_min = float(merged[["logit_coef_baseline", "logit_coef_group"]].min().min())
    common_max = float(merged[["logit_coef_baseline", "logit_coef_group"]].max().max())
    margin = 0.35
    y = np.arange(len(order))

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), sharey=True)
    for ax, df, coef_col, label in [
        (axes[0], merged.set_index("feature"), "logit_coef_baseline", "Baseline split"),
        (axes[1], merged.set_index("feature"), "logit_coef_group", "Group-aware split"),
    ]:
        values = [float(df.loc[feature, coef_col]) for feature in order]
        colors = []
        for feature in order:
            if feature == "HasPicture":
                colors.append(HIGHLIGHT_NEG)
            elif feature == "dup_mass":
                colors.append(HIGHLIGHT_POS)
            else:
                colors.append(COEFF_MUTED)
        ax.barh(y, values, height=0.70, color=colors, edgecolor="white", linewidth=0.8)
        ax.axvline(0, color=MUTED, linewidth=1.0)
        style_axis(ax, grid_axis="x")
        panel_label(ax, label)
        ax.set_xlim(common_min - margin, common_max + margin)
        ax.set_xlabel("Logit coefficient")
        ax.set_yticks(y, display_labels)
    axes[0].set_ylabel("Feature")
    axes[1].tick_params(axis="y", labelleft=False)
    fig.subplots_adjust(left=0.26, right=0.99, bottom=0.15, top=0.88, wspace=0.10)
    save_figure(fig, "picture_context_coefficients_compare.png")


def rounded_card(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    facecolor: str = "white",
    edgecolor: str = EDGE,
    linewidth: float = 1.0,
    fontsize: float = 9.8,
    weight: str = "normal",
    color: str = MUTED,
    align: str = "center",
    boxstyle: str = "round,pad=0.010,rounding_size=0.018",
    linestyle: str = "-",
) -> None:
    card = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=boxstyle,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
    )
    ax.add_patch(card)
    if align == "left":
        tx = x + 0.018
        ha = "left"
    else:
        tx = x + w / 2
        ha = "center"
    ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fontsize, weight=weight, color=color)


def draw_framework() -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(15.8, 5.7), dpi=220)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    section_specs = [
        (0.02, 0.17, "1", "Data basis"),
        (0.205, 0.17, "2", "Preprocessing"),
        (0.39, 0.17, "3", "Split protocols"),
        (0.575, 0.23, "4", "Feature families"),
        (0.82, 0.16, "5", "Fixed-model outputs"),
    ]

    for x, w, badge, title in section_specs:
        outer = FancyBboxPatch(
            (x, 0.08),
            w,
            0.82,
            boxstyle="round,pad=0.008,rounding_size=0.022",
            facecolor=PANEL_BG,
            edgecolor=EDGE,
            linewidth=1.0,
        )
        ax.add_patch(outer)
        ax.text(x + 0.045, 0.855, badge, ha="center", va="center", fontsize=11, weight="bold", color="white", bbox={"boxstyle": "circle,pad=0.18", "facecolor": BASELINE, "edgecolor": "none"})
        ax.text(x + 0.075, 0.855, title, ha="left", va="center", fontsize=11.5, weight="bold", color=INK)

    # arrows between sections
    for start_x, end_x in [(0.19, 0.205), (0.375, 0.39), (0.56, 0.575), (0.805, 0.82)]:
        ax.add_patch(
            FancyArrowPatch(
                (start_x, 0.49),
                (end_x, 0.49),
                arrowstyle="simple",
                mutation_scale=30,
                linewidth=0,
                color="#8EA2B7",
                alpha=0.92,
            )
        )

    # section 1
    x, w = 0.02, 0.17
    rounded_card(
        ax,
        x + 0.016,
        0.18,
        w - 0.032,
        0.48,
        (
            "Observed fields\n\n"
            "- Review and menu text\n\n"
            "- User, restaurant, and time\n\n"
            "- Raw HasPicture indicator\n\n"
            "- BiasFree silver target"
        ),
        facecolor="white",
        edgecolor=EDGE,
        fontsize=10.0,
        weight="normal",
        color=MUTED,
        align="left",
    )
    ax.text(x + 0.03, 0.63, "Translated Yogiyo\nreview-event dataset", ha="left", va="center", fontsize=11.5, weight="bold", color=INK)
    rounded_card(
        ax,
        x + 0.03,
        0.11,
        w - 0.06,
        0.06,
        "126,653 cleaned reviews",
        facecolor="#EDF3FA",
        edgecolor="none",
        fontsize=10.1,
        weight="bold",
        color=BASELINE,
    )

    # section 2
    x, w = 0.205, 0.17
    rounded_card(ax, x + 0.025, 0.61, w - 0.05, 0.12, "Canonicalize review strings\nremove malformed timestamps\nremove empty text", fontsize=9.9, color=MUTED)
    rounded_card(ax, x + 0.025, 0.40, w - 0.05, 0.12, "Create duplicate groups\nfrom canonical review text", fontsize=9.9, color=MUTED)
    rounded_card(ax, x + 0.025, 0.19, w - 0.05, 0.12, "Keep the label convention fixed\npositive class = BiasFree = 1", fontsize=9.8, color=MUTED)

    # section 3
    x, w = 0.39, 0.17
    rounded_card(ax, x + 0.025, 0.56, w - 0.05, 0.15, "Baseline stratified split\n60/20/20\nsame template may recur", facecolor="#E8F0FA", edgecolor=BASELINE, fontsize=10.2, weight="bold", color=INK)
    rounded_card(ax, x + 0.025, 0.31, w - 0.05, 0.18, "Group-aware split on\ncanonical review strings\none string -> one partition", facecolor="#FFF1E3", edgecolor="#DE8A34", fontsize=10.0, weight="bold", color=INK)
    rounded_card(ax, x + 0.035, 0.17, w - 0.07, 0.08, "Leakage audit:\ncanonical overlap = 0", facecolor="white", edgecolor=EDGE, fontsize=9.4, color=MUTED, linestyle="--")

    # section 4
    x, w = 0.575, 0.23
    rounded_card(ax, x + 0.022, 0.61, 0.080, 0.11, "Text only\nHashing 1-2g\nSRP 128", fontsize=9.7, color=MUTED)
    rounded_card(ax, x + 0.115, 0.61, 0.080, 0.11, "Behavior only\nduplication\nburstiness\ncadence", fontsize=9.4, color=MUTED)
    rounded_card(ax, x + 0.022, 0.40, 0.173, 0.09, "Text + behavior\n(no picture)", fontsize=10.0, color=MUTED)
    rounded_card(ax, x + 0.022, 0.18, 0.075, 0.14, "Raw\nHasPicture", facecolor="#E8F0FA", edgecolor=BASELINE, fontsize=10.7, weight="bold", color=HIGHLIGHT_NEG)
    rounded_card(ax, x + 0.118, 0.15, 0.082, 0.19, "Train-only\nPicture-Context\nScore", facecolor="#F2F8E9", edgecolor="#8CB64A", fontsize=10.2, weight="bold", color="#6D9731")
    ax.text(x + 0.159, 0.19, "raw picture +\ncoordination features", ha="center", va="center", fontsize=9.1, color=MUTED)
    ax.add_patch(FancyArrowPatch((x + 0.098, 0.25), (x + 0.118, 0.25), arrowstyle="-|>", mutation_scale=18, linewidth=1.1, color="#8EA2B7"))

    # section 5
    x, w = 0.82, 0.16
    rounded_card(ax, x + 0.025, 0.60, w - 0.05, 0.14, "Same classifier\nacross all runs", fontsize=11.0, weight="bold", color=INK)
    ax.text(x + w / 2, 0.61, "logistic regression\nvalidation threshold", ha="center", va="center", fontsize=9.5, color=MUTED)
    rounded_card(ax, x + 0.03, 0.39, w - 0.06, 0.09, "RQ1\nfeature-family\ncomparison", facecolor="#E8F0FA", edgecolor=BASELINE, fontsize=10.0, weight="bold", color=INK)
    rounded_card(ax, x + 0.03, 0.24, w - 0.06, 0.09, "RQ2\nraw picture vs\npicture-context", facecolor="#F2F8E9", edgecolor="#8CB64A", fontsize=10.0, weight="bold", color=INK)
    rounded_card(ax, x + 0.03, 0.09, w - 0.06, 0.09, "RQ3\nsplit effect\nwith model fixed", facecolor="#FFF1E3", edgecolor="#DE8A34", fontsize=10.0, weight="bold", color=INK)

    fig.tight_layout(pad=0.1)
    save_multi_format(fig, "framework")


def main() -> None:
    draw_feature_family_comparison()
    draw_picture_signal_comparison()
    draw_split_effect_comparison()
    draw_picture_context_coefficients()
    draw_framework()
    print("Updated manuscript figures in", PAPER_DIR)


if __name__ == "__main__":
    main()

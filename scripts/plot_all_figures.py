#!/usr/bin/env python3
"""Draw the evaluation figures of the paper from the CSV files in results/csv/.

Every figure is written to figures/ as PDF and PNG.  A figure whose CSV file is missing is
skipped with a message.  A figure whose CSV file does not yet cover every model
configuration is skipped as well, unless --allow-partial is given: it is then drawn for
the complete model configurations only and is marked as partial.  With --from-expected the
figures are drawn from the reference values in expected/paper/ instead.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Callable, Hashable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import sweeplib
from experiments import context
from sweeplib import GEOMEAN_LABEL, KV_HISTORIES, MODELS

FIGURES_DIR = sweeplib.ROOT / "figures"

# ---------------------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------------------

INK = "#000000"
EDGE = "#333333"
GRID = "#d9d9d9"
GRAY = "#b3b3b3"
PEACH = "#f5ae85"
LIGHT = "#dedede"
DARK = "#7f7f7f"
BLUE = "#8fc1de"
GREEN = "#a8d8b9"
PINK = "#f2a3a3"
PALE = "#e6e6e6"
DRAM_BLUE = "#a9c4de"
SHADE = "#ececec"
PARTIAL_RED = "#b2182b"
# Ordered shades (light to dark) for an ordered parameter.
PEACH_RAMP = ("#fde6d8", "#fbd0b9", "#f8ba99", "#f5ae85", "#e58f5f", "#c96f3f", "#9c5028")
GRAY_RAMP = ("#ededed", "#d2d2d2", "#b3b3b3", "#8c8c8c")

BAR = {"edgecolor": EDGE, "linewidth": 0.5, "zorder": 3}
ERRORBAR = {"ecolor": EDGE, "elinewidth": 0.6, "capsize": 1.6, "capthick": 0.6}
DASHED = (0, (4, 2))
PNG_DPI = 500

RC_PARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 7.0,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7.0,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.3,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.7,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "legend.frameon": True,
    "legend.edgecolor": INK,
    "legend.fancybox": True,
    "legend.framealpha": 1.0,
    "legend.borderpad": 0.35,
    "legend.handlelength": 1.4,
    "legend.handleheight": 0.8,
    "legend.labelspacing": 0.25,
    "patch.linewidth": 0.5,
    "hatch.linewidth": 0.5,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.default": "regular",
}


class Skip(Exception):
    """A figure cannot be drawn from the available data; the message says why."""


# ---------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------


def load_table(csv_dir: Path, name: str, key_columns: Sequence[str]) -> dict[tuple, dict[str, str]]:
    """Rows of one CSV file by their key columns (as text)."""
    path = csv_dir / name
    if not path.is_file():
        raise Skip(f"{path} does not exist")
    rows = sweeplib.read_csv(path)
    missing = [column for column in key_columns if rows and column not in rows[0]]
    if missing:
        raise Skip(f"{path} has no column {missing}")
    return {tuple(row[column] for column in key_columns): row for row in rows}


def number(row: Mapping[str, str], column: str) -> float:
    """A cell as a number; an empty cell (value not reported) is NaN."""
    if column not in row:
        raise Skip(f"the CSV file has no column {column!r}")
    return float(row[column]) if row[column] else math.nan


def complete_models(
    table: Mapping[tuple, Any],
    candidates: Sequence[str],
    required: Callable[[str], Sequence[tuple]],
    allow_partial: bool,
) -> tuple[list[str], str | None]:
    """The models whose required rows are all present, and a note if some models are not.

    Raises :class:`Skip` if no model is complete, or if some are not and partial figures
    are not allowed.
    """
    models = [
        model
        for model in candidates
        if all(tuple(str(part) for part in key) in table for key in required(model))
    ]
    if len(models) == len(candidates):
        return models, None
    absent = ", ".join(model for model in candidates if model not in models)
    note = f"partial data: {len(models)} of {len(candidates)} model configurations"
    if not models:
        raise Skip(f"no model configuration is complete (missing: {absent})")
    if not allow_partial:
        raise Skip(f"{note} (missing: {absent}); use --allow-partial to draw them")
    return models, f"{note} (missing: {absent})"


def geomean(values: Sequence[float]) -> float:
    """Geometric mean of the values that are reported (not NaN)."""
    return sweeplib.geomean(value for value in values if not math.isnan(value))


# ---------------------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------------------


def grouped_bars(
    ax: Axes,
    groups: Sequence[str],
    series: Sequence[Hashable],
    values: Mapping[Hashable, Sequence[float]],
    colors: Sequence[str],
    width: float,
    labels: Sequence[str],
) -> np.ndarray:
    """Grouped bars of ``values[series][group]``; returns the center of every group.

    The last group is the geometric mean: its bars get whiskers that span the minimum and
    the maximum of the other groups.
    """
    bar_width = width / len(series)
    x = np.arange(len(groups))
    for index, name in enumerate(series):
        positions = x - width / 2 + bar_width * (index + 0.5)
        ax.bar(positions, values[name], width=bar_width, color=colors[index], label=labels[index], **BAR)
        others = np.asarray(values[name][:-1], dtype=float)
        others = others[~np.isnan(others)]
        mean = values[name][-1]
        if others.max() - others.min() > 1e-9:
            ax.errorbar(
                positions[-1],
                mean,
                yerr=[[mean - others.min()], [others.max() - mean]],
                fmt="none",
                zorder=5,
                **ERRORBAR,
            )
    ax.set_xticks(x, groups)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.tick_params(axis="x", length=0)
    ax.axvspan(len(groups) - 1.5, len(groups) - 0.4, color=SHADE, zorder=0)
    for label in ax.get_xticklabels()[-1:]:  # No labels on an axis that shares its ticks.
        label.set_fontweight("bold")
    return x


def value_labels(
    ax: Axes,
    series: Sequence[Hashable],
    values: Mapping[Hashable, Sequence[float]],
    labeled: Sequence[Hashable],
    width: float,
    offset: float,
    size: float = 4.6,
) -> None:
    """Write the value above the bars of the ``labeled`` series (rotated).

    The label of a geometric-mean bar (last group) is placed above its whisker.
    """
    bar_width = width / len(series)
    for index, name in enumerate(series):
        if name not in labeled:
            continue
        top = float(np.nanmax(values[name][:-1]))
        for group, value in enumerate(values[name]):
            if math.isnan(value):
                continue
            y = max(value, top) if group == len(values[name]) - 1 else value
            ax.text(
                group - width / 2 + bar_width * (index + 0.5),
                y + offset,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=size,
            )


def framed(legend: Any) -> None:
    legend.get_frame().set_linewidth(0.6)


def patches(entries: Sequence[tuple[str, str]]) -> list[Patch]:
    """Legend handles for (label, color) pairs."""
    return [Patch(facecolor=color, edgecolor=EDGE, linewidth=0.5, label=label) for label, color in entries]


# ---------------------------------------------------------------------------------------
# Figures.  Each returns the figure and a note that is not None if the data is partial.
# ---------------------------------------------------------------------------------------


def context_sensitivity(csv_dir: Path, allow_partial: bool) -> tuple[Figure, str | None]:
    table = load_table(csv_dir, "context_sensitivity.csv", ("model", "kv_history"))
    models, note = complete_models(
        table, MODELS, lambda model: [(model, kv) for kv in KV_HISTORIES], allow_partial
    )
    values = {}
    for kv in KV_HISTORIES:
        speedups = [number(table[(model, str(kv))], "speedup_over_bitnet_cpp") for model in models]
        values[kv] = [*speedups, geomean(speedups)]

    fig, ax = plt.subplots(figsize=(7.1, 1.95))
    fig.subplots_adjust(left=0.062, right=0.995, top=0.97, bottom=0.2)
    labels = [str(kv) for kv in KV_HISTORIES]
    grouped_bars(ax, [*models, GEOMEAN_LABEL], KV_HISTORIES, values, PEACH_RAMP, 0.84, labels)
    value_labels(ax, KV_HISTORIES, values, (KV_HISTORIES[0], KV_HISTORIES[-1]), 0.84, 0.06)
    ax.axhline(1.0, color=INK, linewidth=0.7, linestyle=DASHED, zorder=4)
    ax.set_ylim(0, 6.4)
    ax.set_yticks(range(0, 7))
    ax.set_ylabel("Speedup over\nBitNet.cpp [$\\times$]")
    ax.set_xlabel("Model configuration", labelpad=2)
    framed(
        ax.legend(
            title="KV history [tokens]",
            ncol=len(KV_HISTORIES),
            loc="upper right",
            columnspacing=0.9,
            title_fontsize=6.3,
        )
    )
    return fig, note


def end_to_end_throughput(csv_dir: Path, allow_partial: bool) -> tuple[Figure, str | None]:
    table = load_table(csv_dir, context.END_TO_END_CSV, ("model",))
    published = context.load_kernels()
    candidates = [model for model in MODELS if model in published]
    models, note = complete_models(table, candidates, lambda model: [(model,)], allow_partial)
    series = {
        "llama.cpp (FP16)": ("fp16_normalized", LIGHT),
        "T-MAC": ("t_mac_normalized", GRAY),
        "BitNet.cpp (I2_S)": ("bitnet_cpp_i2_s_normalized", DARK),
        "BitAcc": ("bitacc_normalized", PEACH),
    }
    names = list(series)
    values = {}
    for name, (column, _) in series.items():
        normalized = [number(table[(model,)], column) for model in models]
        values[name] = [*normalized, geomean(normalized)]

    fig, ax = plt.subplots(figsize=(7.1, 1.8))
    fig.subplots_adjust(left=0.062, right=0.995, top=0.97, bottom=0.2)
    colors = [color for _, color in series.values()]
    x = grouped_bars(ax, [*models, GEOMEAN_LABEL], names, values, colors, 0.8, names)
    value_labels(ax, names, values, names[:3], 0.8, 0.04, size=4.8)
    bar_width = 0.8 / len(names)
    bitacc = values["BitAcc"]
    for position, value in zip(x, bitacc):
        top = max(bitacc[:-1]) if position == x[-1] else value
        ax.text(
            position + 0.4 - bar_width / 2,
            top + 0.05,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=5.8,
        )
    for position, value in zip(x, values["llama.cpp (FP16)"]):
        if math.isnan(value):  # Not reported: the CPU cannot host the model with this kernel.
            ax.text(
                position - 0.4 + bar_width / 2,
                0.04,
                "n/a",
                ha="center",
                va="bottom",
                fontsize=5.0,
                rotation=90,
            )
    ax.set_ylim(0, 3.6)
    ax.set_yticks(range(0, 4))
    ax.set_ylabel("Decoding throughput\nnorm. to BitNet.cpp")
    ax.set_xlabel("Model configuration", labelpad=2)
    framed(ax.legend(ncol=4, loc="upper left", columnspacing=1.0))
    return fig, note


def dimm_sensitivity(csv_dir: Path, allow_partial: bool) -> tuple[Figure, str | None]:
    table = load_table(csv_dir, "dimm_sensitivity.csv", ("variant", "model", "kv_history"))
    series = {
        "dimm_x2": ("DIMM-level PE 2$\\times$ faster", GRAY),
        "dimm_x4": ("DIMM-level PE 4$\\times$ faster", PEACH),
    }
    histories = (128, 1024)
    models, note = complete_models(
        table,
        MODELS,
        lambda model: [(variant, model, kv) for variant in series for kv in histories],
        allow_partial,
    )
    variants = list(series)
    labels = [label for label, _ in series.values()]
    colors = [color for _, color in series.values()]

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 1.75), sharey=True)
    fig.subplots_adjust(left=0.062, right=0.995, top=0.87, bottom=0.245, wspace=0.03)
    for ax, kv in zip(axes, histories):
        values = {}
        for variant in variants:
            speedups = [
                number(table[(variant, model, str(kv))], "speedup_over_base") for model in models
            ]
            values[variant] = [*speedups, geomean(speedups)]
        grouped_bars(ax, [*models, "Geo\nMean"], variants, values, colors, 0.84, labels)
        value_labels(ax, variants, values, variants, 0.84, 0.02, size=4.2)
        ax.axhline(1.0, color=INK, linewidth=0.7, linestyle=DASHED, zorder=4)
        ax.set_ylim(0.8, 1.95)
        ax.set_yticks((1.0, 1.25, 1.5))
        ax.tick_params(axis="x", labelsize=5.4)
        ax.set_xlabel("Model configuration", labelpad=2)
        ax.text(
            0.015,
            0.95,
            f"KV history = {kv} tokens",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.3,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": INK, "lw": 0.6},
        )
    axes[0].set_ylabel("Throughput norm.\nto BitAcc")
    framed(
        fig.legend(
            handles=patches(list(series.values())),
            ncol=2,
            loc="upper center",
            bbox_to_anchor=(0.53, 1.0),
            columnspacing=1.4,
        )
    )
    return fig, note


LOCALITY_DESIGNS = ("no_pim", "fc_pim", "attention_pim", "bitacc")
LOCALITY_KEYS = ("model", "history_tokens", "design")


def locality_breakdown(csv_dir: Path, allow_partial: bool) -> tuple[Figure, str | None]:
    table = load_table(csv_dir, "locality_breakdown.csv", LOCALITY_KEYS)
    histories = (128, 512, 1024)
    models, note = complete_models(
        table,
        MODELS,
        lambda model: [(model, kv, design) for kv in histories for design in LOCALITY_DESIGNS],
        allow_partial,
    )
    components = (
        ("FC computation", "fc_computing_normalized", PEACH),
        ("Attention computation", "attention_computing_normalized", GREEN),
        ("On-chip data movement", "onchip_data_movement_normalized", PINK),
        ("Off-chip data movement", "offchip_data_movement_normalized", BLUE),
        ("Other", "others_normalized", PALE),
    )
    letters = ("N", "F", "A", "B")
    history_gap, model_gap = 1.0, 2.2
    half = math.ceil(len(models) / 2)
    fig, axes = plt.subplots(2, 1, figsize=(7.1, 3.5))
    fig.subplots_adjust(left=0.062, right=0.995, top=0.89, bottom=0.115, hspace=0.62)
    for ax, row in zip(axes, (models[:half], models[half:])):
        x = 0.0
        ticks: list[tuple[float, str]] = []
        for model in row:
            first = x
            for kv in histories:
                start = x
                for design, letter in zip(LOCALITY_DESIGNS, letters):
                    bottom = 0.0
                    for _, column, color in components:
                        value = number(table[(model, str(kv), design)], column)
                        ax.bar(x, value, width=1.0, bottom=bottom, color=color, **BAR)
                        bottom += value
                    ax.text(
                        x,
                        bottom + 0.03,
                        f"{bottom:.2f}",
                        ha="center",
                        va="bottom",
                        rotation=90,
                        fontsize=4.4,
                    )
                    ticks.append((x, letter))
                    x += 1.0
                ax.text(
                    (start + x - 1.0) / 2,
                    -0.105,
                    "KV: 1K" if kv == 1024 else f"KV: {kv}",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=5.0,
                )
                x += history_gap
            ax.text(
                (first + x - history_gap - 1.0) / 2,
                -0.215,
                model,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=6.5,
                fontweight="bold",
            )
            x += model_gap - history_gap
        ax.set_xlim(-1.2, x - model_gap + 0.2)
        ax.set_ylim(0, 1.45)
        ax.set_yticks((0, 0.25, 0.5, 0.75, 1.0))
        ax.axhline(1.0, color=INK, linewidth=0.6, linestyle=DASHED, zorder=4)
        ax.set_xticks([position for position, _ in ticks], [letter for _, letter in ticks])
        ax.tick_params(axis="x", length=0, pad=1, labelsize=4.8)
        ax.set_ylabel("Decoding time\nnorm. to No-PIM")
    framed(
        fig.legend(
            handles=patches([(label, color) for label, _, color in components]),
            ncol=5,
            loc="upper center",
            bbox_to_anchor=(0.53, 1.0),
            columnspacing=1.2,
        )
    )
    fig.text(
        0.53,
        0.905,
        "N: No-PIM    F: FC-PIM    A: Attention-PIM    B: BitAcc",
        ha="center",
        va="bottom",
        fontsize=6.0,
    )
    return fig, note


def locality_distribution(csv_dir: Path, allow_partial: bool) -> tuple[Figure, str | None]:
    table = load_table(csv_dir, "locality_breakdown.csv", LOCALITY_KEYS)
    histories = (128, 512, 1024)
    models, note = complete_models(
        table,
        MODELS,
        lambda model: [(model, kv, design) for kv in histories for design in LOCALITY_DESIGNS],
        allow_partial,
    )
    # (design, label, color of the box, color of the line through the medians)
    series = (
        ("fc_pim", "FC-PIM", BLUE, "#3f86b3"),
        ("attention_pim", "Attention-PIM", GREEN, "#4f9a6b"),
        ("bitacc", "BitAcc", PEACH, "#c96f3f"),
    )
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    fig.subplots_adjust(left=0.2, right=0.985, top=0.885, bottom=0.19)
    ax.axhline(1.0, color=INK, linewidth=0.8, linestyle=DASHED, zorder=2)  # No-PIM
    for index, (design, _, color, line_color) in enumerate(series):
        positions = [group - 0.27 + index * 0.27 for group in range(len(histories))]
        data = [
            [
                number(table[(model, str(kv), design)], "normalized_time_vs_no_pim")
                for model in models
            ]
            for kv in histories
        ]
        ax.boxplot(
            data,
            positions=positions,
            widths=0.22,
            patch_artist=True,
            showmeans=True,
            manage_ticks=False,
            boxprops={"facecolor": color, "edgecolor": EDGE, "linewidth": 0.5},
            medianprops={"color": INK, "linewidth": 0.7},
            whiskerprops={"color": EDGE, "linewidth": 0.5},
            capprops={"color": EDGE, "linewidth": 0.5},
            flierprops={
                "marker": "o",
                "markersize": 2.2,
                "markerfacecolor": "none",
                "markeredgecolor": color,
                "markeredgewidth": 0.6,
            },
            meanprops={
                "marker": "D",
                "markersize": 2.6,
                "markerfacecolor": "white",
                "markeredgecolor": INK,
                "markeredgewidth": 0.5,
            },
            zorder=3,
        )
        medians = [float(np.median(values)) for values in data]
        ax.plot(positions, medians, color=line_color, linewidth=0.9, zorder=4)
    for boundary in range(1, len(histories)):
        ax.axvline(boundary - 0.5, color=GRID, linewidth=0.5, zorder=1)
    ax.set_xticks(range(len(histories)), [str(kv) for kv in histories])
    ax.set_xlim(-0.5, len(histories) - 0.5)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 1.1)
    ax.set_yticks((0, 0.25, 0.5, 0.75, 1.0))
    ax.set_xlabel("KV history [tokens]", labelpad=2)
    ax.set_ylabel("Decoding time\nnorm. to No-PIM")
    reference = Line2D([], [], color=INK, linewidth=0.8, linestyle=DASHED, label="No-PIM")
    framed(
        fig.legend(
            handles=[reference, *patches([(label, color) for _, label, color, _ in series])],
            ncol=4,
            loc="upper center",
            bbox_to_anchor=(0.57, 1.0),
            columnspacing=0.9,
            handlelength=1.3,
        )
    )
    return fig, note


def scaling(csv_dir: Path, allow_partial: bool) -> tuple[Figure, str | None]:
    table = load_table(csv_dir, "scaling.csv", ("model", "kv_history", "chips", "bank_pes"))
    kv, chip_steps, bank_pe_steps = 1024, (1, 2, 4, 8), (4, 8, 16, 32)
    # (steps, key of a step, legend title, reference of the normalization)
    panels = (
        (
            chip_steps,
            lambda model, chips: (model, kv, chips, bank_pe_steps[-1]),
            f"Chips per rank ({bank_pe_steps[-1]} bank-level PEs per chip)",
            f"{chip_steps[0]} chip",
        ),
        (
            bank_pe_steps,
            lambda model, bank_pes: (model, kv, chip_steps[-1], bank_pes),
            f"Active bank-level PEs per chip ({chip_steps[-1]} chips)",
            f"{bank_pe_steps[0]} bank PEs",
        ),
    )
    models, note = complete_models(
        table,
        MODELS,
        lambda model: [key(model, step) for steps, key, _, _ in panels for step in steps],
        allow_partial,
    )

    fig, axes = plt.subplots(2, 1, figsize=(7.1, 2.9), sharex=True)
    fig.subplots_adjust(left=0.062, right=0.995, top=0.975, bottom=0.12, hspace=0.12)
    for ax, (steps, key, title, reference) in zip(axes, panels):
        throughput = {
            (model, step): number(table[tuple(str(part) for part in key(model, step))], "tokens_per_s")
            for model in models
            for step in steps
        }
        values = {}
        for step in steps:
            speedups = [throughput[(model, step)] / throughput[(model, steps[0])] for model in models]
            values[step] = [*speedups, geomean(speedups)]
        colors = [*GRAY_RAMP[:3], PEACH]
        labels = [str(step) for step in steps]
        grouped_bars(ax, [*models, GEOMEAN_LABEL], steps, values, colors, 0.8, labels)
        value_labels(ax, steps, values, steps, 0.8, 0.08)
        ax.set_ylim(0, math.ceil(max(values[steps[-1]]) * 1.3))
        ax.set_ylabel(f"Throughput norm. to\n{reference}")
        framed(ax.legend(title=title, ncol=4, loc="upper left", columnspacing=1.0, title_fontsize=6.3))
    axes[1].set_xlabel("Model configuration", labelpad=2)
    return fig, note


def energy_per_token(csv_dir: Path, allow_partial: bool) -> tuple[Figure, str | None]:
    table = load_table(csv_dir, "energy_per_token.csv", ("model", "kv_history"))
    models, note = complete_models(
        table, MODELS, lambda model: [(model, kv) for kv in KV_HISTORIES], allow_partial
    )
    components = (
        ("Host (active)", ("host_active_j",), GRAY),
        ("Host (idle)", ("host_idle_j",), LIGHT),
        ("DRAM", ("dram_array_j", "dram_transfer_j", "dram_refresh_j", "dram_background_j"), DRAM_BLUE),
        ("Processing engines", ("pe_logic_j", "pe_buffer_j", "pe_leakage_j"), PEACH),
    )
    x = range(len(KV_HISTORIES))
    fig, axes = plt.subplots(2, 6, figsize=(7.1, 2.75))
    fig.subplots_adjust(left=0.06, right=0.995, top=0.89, bottom=0.14, hspace=0.42, wspace=0.42)
    for ax in axes.ravel()[len(models) :]:
        ax.set_visible(False)
    for ax, model in zip(axes.ravel(), models):
        rows = [table[(model, str(kv))] for kv in KV_HISTORIES]
        largest = max(number(row, "total_j") for row in rows)
        scale, unit = (1e3, "mJ") if largest < 0.1 else (1.0, "J")
        bottom = np.zeros(len(rows))
        for _, columns, color in components:
            energy = scale * np.array([sum(number(row, column) for column in columns) for row in rows])
            ax.bar(x, energy, width=0.72, bottom=bottom, color=color, **BAR)
            bottom += energy
        for position, total in enumerate(bottom):
            ax.text(
                position,
                total + bottom.max() * 0.025,
                f"{total:.3g}" if total < 100 else f"{total:.0f}",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=4.6,
            )
        ax.set_title(f"{model}  [{unit}/token]", pad=2, fontsize=6.5)
        ax.set_xticks(x, ["32", "", "128", "", "512", "", "2K"])
        ax.tick_params(axis="x", length=0, labelsize=5.8)
        ax.tick_params(axis="y", labelsize=5.8, pad=1.5)
        ax.set_ylim(0, bottom.max() * 1.3)
        ax.locator_params(axis="y", nbins=4)
    for ax in axes[1]:
        ax.set_xlabel("KV history [tokens]", labelpad=1.5, fontsize=6.3)
    for ax in axes[:, 0]:
        ax.set_ylabel("Energy per token", fontsize=6.5)
    framed(
        fig.legend(
            handles=patches([(label, color) for label, _, color in components]),
            ncol=4,
            loc="upper center",
            bbox_to_anchor=(0.53, 1.0),
            columnspacing=1.4,
        )
    )
    return fig, note


FIGURES: dict[str, Callable[[Path, bool], tuple[Figure, str | None]]] = {
    "end_to_end_throughput": end_to_end_throughput,
    "context_sensitivity": context_sensitivity,
    "locality_breakdown": locality_breakdown,
    "locality_distribution": locality_distribution,
    "dimm_sensitivity": dimm_sensitivity,
    "scaling": scaling,
    "energy_per_token": energy_per_token,
}


def save(fig: Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    options = {"bbox_inches": "tight", "pad_inches": 0.02}
    # Without a creation date, the same data gives the same PDF file.
    fig.savefig(out_dir / f"{name}.pdf", metadata={"CreationDate": None}, **options)
    fig.savefig(out_dir / f"{name}.png", dpi=PNG_DPI, **options)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "figures",
        nargs="*",
        metavar="FIGURE",
        help=f"figures to draw (default: all of {', '.join(FIGURES)})",
    )
    parser.add_argument(
        "--csv-dir", type=Path, default=sweeplib.CSV_DIR, help="directory of the CSV files"
    )
    parser.add_argument(
        "--from-expected",
        action="store_true",
        help="read the reference CSV files in expected/paper/ instead of --csv-dir",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=FIGURES_DIR, help="output directory of the figures"
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="draw a figure for the complete model configurations if some are missing, "
        "and mark it as partial",
    )
    args = parser.parse_args()
    unknown = [name for name in args.figures if name not in FIGURES]
    if unknown:
        parser.error(f"unknown figure(s) {', '.join(unknown)}; available: {', '.join(FIGURES)}")
    csv_dir = sweeplib.EXPECTED_DIR if args.from_expected else args.csv_dir

    plt.rcParams.update(RC_PARAMS)
    skipped = 0
    for name in args.figures or FIGURES:
        try:
            fig, note = FIGURES[name](csv_dir, args.allow_partial)
        except Skip as reason:
            skipped += 1
            print(f"skipped {name}: {reason}")
            continue
        if note is not None:
            fig.text(0.995, 0.004, note, ha="right", va="bottom", fontsize=5.0, color=PARTIAL_RED)
        save(fig, args.out_dir, name)
        plt.close(fig)
        print(f"wrote {args.out_dir / name}.pdf and .png" + (f" ({note})" if note else ""))
    print(f"{len(args.figures or FIGURES) - skipped} figure(s) written, {skipped} skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())

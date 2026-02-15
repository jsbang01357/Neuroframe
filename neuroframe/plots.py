# neuroframe/plots.py
from __future__ import annotations

from typing import Optional
import matplotlib.pyplot as plt
import datetime as dt

from .engine import CurveOutput


def _mask_to_spans(t, mask):
    """Convert boolean mask to (start, end) spans over time grid."""
    spans = []
    if not t or not mask or len(t) != len(mask):
        return spans

    in_span = False
    start = None

    for i in range(len(mask)):
        if mask[i] and not in_span:
            in_span = True
            start = t[i]
        elif not mask[i] and in_span:
            in_span = False
            end = t[i]
            spans.append((start, end))
            start = None

    if in_span and start is not None:
        spans.append((start, t[-1] + (t[-1] - t[-2] if len(t) >= 2 else dt.timedelta(minutes=10))))
    return spans


def plot_net_energy(
    out: CurveOutput,
    title: str = "NeuroFrame — Net Energy Curve",
    show_components: bool = False,
    ax: Optional[plt.Axes] = None
):
    """
    Plot net energy (0..1) and shade zones.
    If show_components=True, also plots circadian/sleep_pressure/drug/load (raw-ish).
    Returns matplotlib Figure.
    """
    t = out.t
    net = out.net

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.plot(t, net, label="net (0..1)")

    # Shade zones
    zones = out.zones or {}
    prime_spans = _mask_to_spans(t, zones.get("prime", []))
    crash_spans = _mask_to_spans(t, zones.get("crash", []))
    gate_spans = _mask_to_spans(t, zones.get("sleep_gate", []))

    for (s, e) in prime_spans:
        ax.axvspan(s, e, alpha=0.15, label="_prime")
    for (s, e) in crash_spans:
        ax.axvspan(s, e, alpha=0.15, label="_crash")
    for (s, e) in gate_spans:
        ax.axvspan(s, e, alpha=0.15, label="_sleep_gate")

    # Legend labels (avoid duplicates)
    handles, labels = ax.get_legend_handles_labels()
    # Add zone legend entries as dummy handles (once)
    zone_labels = ["Prime Zone", "Crash Zone", "Sleep Gate"]
    for zl in zone_labels:
        if zl not in labels:
            ax.plot([], [], label=zl)

    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Net Energy (normalized)")
    ax.set_xlabel("Time")
    ax.grid(True, alpha=0.2)

    # Nice x ticks
    fig.autofmt_xdate()

    if show_components:
        # Components are not normalized; plot with secondary axis to avoid flattening net
        ax2 = ax.twinx()
        ax2.plot(t, out.circadian, label="circadian", linestyle="--")
        ax2.plot(t, out.sleep_pressure, label="sleep_pressure", linestyle="--")
        ax2.plot(t, out.drug, label="drug", linestyle="--")
        ax2.plot(t, out.load, label="load", linestyle="--")
        ax2.set_ylabel("Components (raw scale)")
        ax2.grid(False)

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper right")
    else:
        ax.legend(loc="upper right")

    return fig

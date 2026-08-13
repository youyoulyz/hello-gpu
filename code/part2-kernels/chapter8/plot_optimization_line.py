"""Line chart: Vector Add effective bandwidth across all seven implementations.

Compares the current platform's measured evidence with the committed gfx1201
reference.  The x axis follows the ordered implementation list so the
optimization trajectory is directly visible: baseline (v0) -> coalesced
(v1-contiguous) -> controlled strided counter-example (v1-strided) ->
grid-stride / float4 / Triton variants (v2, v3, t0, t1).

Style follows plot_vector_add_ch7.py (figures4papers publication house style).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]

ORDER = (
    "hip-v0",
    "hip-v1-contiguous",
    "hip-v1-strided",
    "hip-v2",
    "hip-v3",
    "triton-t0",
    "triton-t1",
)
LABELS = {
    "hip-v0": "HIP v0\n标量逐元素",
    "hip-v1-contiguous": "HIP v1\n连续",
    "hip-v1-strided": "HIP v1\n跨步",
    "hip-v2": "HIP v2\ngrid-stride",
    "hip-v3": "HIP v3\nfloat4",
    "triton-t0": "Triton t0\nBLOCK=256",
    "triton-t1": "Triton t1\nBLOCK=1024",
}

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_2": "#AADCA9",
    "green_3": "#8BCF8B",
    "red_strong": "#B64342",
    "neutral": "#CFCECE",
    "ink": "#17202a",
    "muted": "#5b6675",
}

PUBLICATION_RCPARAMS = {
    "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 15,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 2,
    "legend.frameon": False,
    "svg.fonttype": "none",
}


def pick_cjk_font(plt: object, font_manager: object) -> None:
    candidates = [
        "PingFang SC",
        "Hiragino Sans GB",
        "Source Han Sans SC",
        "Noto Sans CJK SC",
        "Microsoft YaHei",
        "WenQuanYi Zen Hei",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = [name, "Arial", "DejaVu Sans", "sans-serif"]
            return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-summary", type=Path)
    parser.add_argument("--current-manifest", type=Path)
    parser.add_argument("--reference-summary", type=Path)
    parser.add_argument("--reference-manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", type=str)
    return parser.parse_args()


def read_summary(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    by_name: dict[str, dict[str, str]] = {}
    for row in rows:
        implementation = row.get("implementation", "")
        if implementation in by_name:
            raise ValueError(f"summary has duplicate implementation {implementation}")
        by_name[implementation] = row
    return by_name


def bandwidths(rows: dict[str, dict[str, str]]) -> list[float | None]:
    values: list[float | None] = []
    for name in ORDER:
        row = rows.get(name)
        if row is None:
            values.append(None)
            continue
        try:
            value = float(row["effective_bandwidth_gbs"])
        except (KeyError, TypeError, ValueError):
            value = math.nan
        values.append(value if math.isfinite(value) and value > 0 else None)
    return values


def platform_label(manifest_path: Path | None, fallback: str) -> str:
    if manifest_path is None or not manifest_path.is_file():
        return fallback
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback
    hardware = str(manifest.get("hardware", "")).strip()
    return hardware or fallback


def experiment_subtitle(manifest_path: Path | None) -> str:
    if manifest_path is None or not manifest_path.is_file():
        return "N=16,777,216 FP32 | kernel-only GPU event 计时"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "N=16,777,216 FP32 | kernel-only GPU event 计时"
    software = manifest.get("software", {})
    benchmark = manifest.get("benchmark", {})
    size = benchmark.get("size", "16,777,216")
    rocm = software.get("rocm", "")
    torch_version = software.get("torch", "")
    parts = [f"N={size:,} FP32" if isinstance(size, int) else f"N={size} FP32"]
    if rocm:
        parts.append(f"ROCm {rocm}")
    if torch_version:
        parts.append(f"torch {torch_version}")
    return " | ".join(parts) + " | kernel-only GPU event 计时"


def speedup_notes(values: list[float | None]) -> list[str]:
    base = next((v for v in values if v is not None), None)
    notes: list[str] = []
    for name, value in zip(ORDER, values, strict=True):
        if value is None or base is None or base <= 0:
            notes.append("—")
        else:
            notes.append(f"×{value / base:.2f}")
    return notes


def main() -> None:
    args = parse_args()
    current = read_summary(args.current_summary)
    reference = read_summary(args.reference_summary)
    if not current and not reference:
        raise SystemExit("至少需要 current 或 reference 的 summary.csv")

    current_bw = bandwidths(current)
    reference_bw = bandwidths(reference)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    import logging

    # Arial/Helvetica are aspirational on Linux; fallback is handled by
    # pick_cjk_font, so keep the noisy "Font family not found" warnings out of
    # notebook output.
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

    plt.rcParams.update(PUBLICATION_RCPARAMS)
    pick_cjk_font(plt, font_manager)
    plt.rcParams["axes.unicode_minus"] = False

    positions = list(range(len(ORDER)))
    figure, ax = plt.subplots(figsize=(13.5, 6.8), dpi=200)
    figure.patch.set_facecolor("white")

    current_label = platform_label(args.current_manifest, "当前平台")
    reference_label = platform_label(args.reference_manifest, "gfx1201 Reference")
    subtitle = (
        experiment_subtitle(args.current_manifest)
        if current
        else experiment_subtitle(args.reference_manifest)
    )

    if any(v is not None for v in current_bw):
        ax.plot(
            positions,
            current_bw,
            color=PALETTE["blue_main"],
            marker="o",
            markersize=9,
            linewidth=2.6,
            label=current_label,
            zorder=3,
        )
        for x, value in zip(positions, current_bw, strict=True):
            if value is None:
                continue
            ax.annotate(
                f"{value:.0f}",
                (x, value),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=12,
                fontweight="bold",
                color=PALETTE["ink"],
            )

    if any(v is not None for v in reference_bw):
        ax.plot(
            positions,
            reference_bw,
            color=PALETTE["muted"],
            marker="s",
            markersize=7,
            linewidth=2.0,
            linestyle="--",
            label=reference_label,
            zorder=2,
        )
        for x, value in zip(positions, reference_bw, strict=True):
            if value is None:
                continue
            ax.annotate(
                f"{value:.0f}",
                (x, value),
                textcoords="offset points",
                xytext=(0, -18),
                ha="center",
                fontsize=11,
                color=PALETTE["muted"],
            )

    strided_index = ORDER.index("hip-v1-strided")
    if any(v is not None for v in current_bw) and current_bw[strided_index] is not None:
        ax.annotate(
            "受控负例：跨步访存",
            (strided_index, current_bw[strided_index]),
            textcoords="offset points",
            xytext=(0, 24),
            ha="center",
            fontsize=11,
            color=PALETTE["red_strong"],
            fontweight="bold",
        )

    ax.set_xticks(positions, [LABELS[name] for name in ORDER], fontsize=13)
    ax.set_ylabel("有效带宽 (GB/s，逻辑字节)", fontsize=15, labelpad=10)
    ax.set_ylim(0, max(v for v in [*current_bw, *reference_bw] if v is not None) * 1.25)
    ax.yaxis.grid(True, color=PALETTE["neutral"], linewidth=0.8, alpha=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=6, width=1.5, labelsize=13)
    ax.tick_params(axis="x", length=0, width=1.5)

    legend = ax.legend(loc="upper left", fontsize=13, ncols=2, columnspacing=1.2)
    for handle in legend.legend_handles:
        handle.set_linewidth(2.4)

    title = args.title or "Vector Add 七个实现的有效带宽：优化效果一览"
    ax.set_title(title, loc="left", fontsize=20, fontweight="bold", pad=18, color=PALETTE["ink"])
    figure.text(0.02, 0.985, subtitle, ha="left", fontsize=12, color=PALETTE["muted"])

    notes = speedup_notes(current_bw)
    note_text = "当前平台相对 v0 的带宽倍率：" + "  ".join(
        f"{LABELS[name].splitlines()[0]} {notes[index]}"
        for index, name in enumerate(ORDER)
        if current_bw[index] is not None
    )
    figure.text(
        0.02,
        0.015,
        note_text
        + "\n有效带宽 = 3 × N × 4 Byte / median_ms；跨步版为受控负例，其余版本受内存带宽上限约束。",
        ha="left",
        va="bottom",
        fontsize=10.5,
        color=PALETTE["muted"],
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, bbox_inches="tight", facecolor="white")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()

"""Turn results/raw.jsonl into results/table.md and the ladder chart.

usage: uv run bench/report.py [RESULTS_DIR] [THREADS, e.g. 1,4,8]

Every number is a median over repetitions with a 95% bootstrap interval.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

COLORS = {1: "#9aa5b1", 2: "#7048e8", 4: "#4c8bf5", 8: "#e8590c", 16: "#2b8a3e"}
METRICS = [("decode_tok_s", "Decode"), ("prefill_tok_s", "Prefill")]
LLAMA_STYLE = {"llama.cpp Q8_0": "--", "llama.cpp Q4_0": ":"}


def median_ci(samples, n_boot=10000, seed=0):
    x = np.asarray(samples, dtype=float)
    boots = np.median(np.random.default_rng(seed).choice(x, size=(n_boot, len(x))), axis=1)
    return float(np.median(x)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def fmt(v):
    return f"{v:.0f}" if v >= 100 else f"{v:.1f}" if v >= 10 else f"{v:.2f}"


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "results")
    keep = {int(t) for t in sys.argv[2].split(",")} if len(sys.argv) > 2 else None
    rows = [json.loads(line) for line in (out / "raw.jsonl").read_text().splitlines()]
    rows = [r for r in rows if keep is None or r["threads"] in keep]
    pp, tg = rows[0]["pp"], rows[0]["tg"]
    threads = sorted({r["threads"] for r in rows})
    samples = defaultdict(list)
    labels = {}
    for r in rows:  # abacus runs are keyed (step, flags), llama.cpp runs by label
        key = (r["index"], r.get("flags", "")) if r["runner"] == "abacus" else r["label"]
        labels[key] = r["label"]
        for metric, _ in METRICS:
            samples[key, r["threads"], metric].append(r[metric])
    stats = {k: median_ci(v) for k, v in samples.items()}
    steps = sorted(k for k in labels if isinstance(k, tuple) and not k[1])
    q4 = sorted(k for k in labels if isinstance(k, tuple) and k[1])
    llama = sorted(k for k in labels if isinstance(k, str))
    reps = len(samples[steps[0], threads[0], "decode_tok_s"])

    lines = [f"Qwen3-0.6B, {pp}-token prefill and {tg}-token decode. Median tokens/s of {reps} interleaved runs; "
             "brackets are 95% bootstrap intervals.", ""]
    for metric, name in METRICS:
        lines += [f"## {name} tokens/s", "", "| # | Commit | " + " | ".join(f"{t} thr" for t in threads) + " |",
                  "|---:|---|" + "---:|" * len(threads)]
        for key in steps + q4 + llama:
            cells = []
            for t in threads:
                m, lo, hi = stats[key, t, metric]
                cells.append(f"{fmt(m)} [{fmt(lo)}, {fmt(hi)}]")
            num = str(key[0] + 1) if isinstance(key, tuple) else ""
            lines.append(f"| {num} | {labels[key]} | " + " | ".join(cells) + " |")
        lines.append("")
    pairs = [(steps[-1], "llama.cpp Q8_0")] + ([(q4[-1], "llama.cpp Q4_0")] if q4 else [])
    pairs = [(a, b) for a, b in pairs if b in labels]
    lines += ["## abacus vs llama.cpp", "", "| Threads | " + " | ".join(
        f"{name}: {labels[a]} vs {b}" for a, b in pairs for _, name in METRICS) + " |",
              "|---:|" + "---:|" * (len(METRICS) * len(pairs))]
    for t in threads:
        ratios = [stats[a, t, metric][0] / stats[b, t, metric][0] for a, b in pairs for metric, _ in METRICS]
        lines.append(f"| {t} | " + " | ".join(f"{r:.2f}x" for r in ratios) + " |")
    (out / "table.md").write_text("\n".join(lines) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharex=True)
    x = {key[0]: i + 1 for i, key in enumerate(steps)}
    for ax, (metric, name) in zip(axes, METRICS):
        for t in threads:
            color = COLORS.get(t, "black")
            for series, style, face in ((steps, "-", color), (q4, ":", "white")):
                if series:
                    m = np.array([stats[k, t, metric] for k in series])
                    ax.errorbar([x[k[0]] for k in series], m[:, 0], yerr=[m[:, 0] - m[:, 1], m[:, 2] - m[:, 0]],
                                color=color, ls=style, marker="o", ms=4, mfc=face, lw=1.8, capsize=2)
            for key in llama:
                ax.axhline(stats[key, t, metric][0], color=color, ls=LLAMA_STYLE.get(key, "-."), lw=1, alpha=0.8)
        ax.set_yscale("log")
        ax.set_title(f"{name} ({pp if metric == 'prefill_tok_s' else tg} tokens)", fontsize=12, loc="left")
        ax.set_ylabel("tokens/s")
        ax.set_xticks(list(x.values()), [labels[k] for k in steps], rotation=40, ha="right", fontsize=8)
        ax.set_xlim(0.5, len(steps) + 0.5)
        ax.grid(True, which="major", color="#e9ecef")
        ax.grid(True, which="minor", color="#f4f5f7", lw=0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    legend = [Line2D([], [], color=COLORS.get(t, "black"), lw=1.8, label=f"{t} thread{'s' if t > 1 else ''}")
              for t in threads]
    legend += [Line2D([], [], color="#495057", lw=1.8, marker="o", ms=4, label="abacus")]
    if q4:
        legend += [Line2D([], [], color="#495057", lw=1.8, ls=":", marker="o", ms=4, mfc="white",
                          label="abacus --q4")]
    legend += [Line2D([], [], color="#495057", lw=1, ls=LLAMA_STYLE.get(k, "-."), label=k) for k in llama]
    axes[0].legend(handles=legend, frameon=False, fontsize=8, loc="lower right")
    cpu = json.loads((out / "machine.json").read_text())
    fig.suptitle(f"abacus commit by commit vs llama.cpp, Qwen3-0.6B on {cpu['cpu']}, {cpu['cores']} CPUs",
                 fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "ladder.svg")

    print((out / "table.md").read_text())


if __name__ == "__main__":
    main()

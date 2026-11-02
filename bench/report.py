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

COLORS = {1: "#9aa5b1", 4: "#4c8bf5", 8: "#e8590c", 16: "#2b8a3e"}
METRICS = [("decode_tok_s", "Decode"), ("prefill_tok_s", "Prefill")]


def median_ci(samples, n_boot=10000, seed=0):
    x = np.asarray(samples, dtype=float)
    boots = np.median(np.random.default_rng(seed).choice(x, size=(n_boot, len(x))), axis=1)
    return float(np.median(x)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def fmt(v):
    return f"{v:.0f}" if v >= 100 else f"{v:.1f}" if v >= 10 else f"{v:.2f}"


def short(subject):
    return subject.split(":")[0]


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "results")
    keep = {int(t) for t in sys.argv[2].split(",")} if len(sys.argv) > 2 else None
    rows = [json.loads(line) for line in (out / "raw.jsonl").read_text().splitlines()]
    rows = [r for r in rows if keep is None or r["threads"] in keep]
    pp, tg = rows[0]["pp"], rows[0]["tg"]
    threads = sorted({r["threads"] for r in rows})
    samples = defaultdict(list)
    labels = {}
    for r in rows:
        key = r["index"] if r["runner"] == "abacus" else r["label"]
        labels[key] = r["label"]
        for metric, _ in METRICS:
            samples[key, r["threads"], metric].append(r[metric])
    stats = {k: median_ci(v) for k, v in samples.items()}
    commits = sorted(k for k in labels if isinstance(k, int))
    llama = sorted(k for k in labels if not isinstance(k, int))
    reps = len(samples[commits[0], threads[0], "decode_tok_s"])

    lines = [f"Qwen3-0.6B, {pp}-token prefill and {tg}-token decode. Median tokens/s of {reps} interleaved runs; "
             "brackets are 95% bootstrap intervals.", ""]
    for metric, name in METRICS:
        lines += [f"## {name} tokens/s", "", "| # | Commit | " + " | ".join(f"{t} thr" for t in threads) + " |",
                  "|---:|---|" + "---:|" * len(threads)]
        for key in commits + llama:
            cells = []
            for t in threads:
                m, lo, hi = stats[key, t, metric]
                cells.append(f"{fmt(m)} [{fmt(lo)}, {fmt(hi)}]")
            num = str(key + 1) if isinstance(key, int) else ""
            lines.append(f"| {num} | {labels[key]} | " + " | ".join(cells) + " |")
        lines.append("")
    last = commits[-1]
    lines += ["## abacus vs llama.cpp", "", "| Threads | " + " | ".join(
        f"{name} vs {labels[k]}" for metric, name in METRICS for k in llama) + " |",
              "|---:|" + "---:|" * (len(METRICS) * len(llama))]
    for t in threads:
        ratios = [stats[last, t, metric][0] / stats[k, t, metric][0] for metric, _ in METRICS for k in llama]
        lines.append(f"| {t} | " + " | ".join(f"{r:.2f}x" for r in ratios) + " |")
    (out / "table.md").write_text("\n".join(lines) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharex=True)
    x = np.arange(1, len(commits) + 1)
    for ax, (metric, name) in zip(axes, METRICS):
        for t in threads:
            m = np.array([stats[c, t, metric] for c in commits])
            color = COLORS.get(t, "black")
            ax.errorbar(x, m[:, 0], yerr=[m[:, 0] - m[:, 1], m[:, 2] - m[:, 0]], color=color, marker="o", ms=4,
                        lw=1.8, capsize=2, label=f"abacus, {t} thread{'s' if t > 1 else ''}")
            ref = stats["llama.cpp Q8_0", t, metric][0] if "llama.cpp Q8_0" in labels else None
            if ref:
                ax.axhline(ref, color=color, ls="--", lw=1, alpha=0.8)
                ax.annotate(f"llama.cpp Q8_0, {t}t", (x[-1] + 0.15, ref), color=color, fontsize=8, va="center")
        ax.set_yscale("log")
        ax.set_title(f"{name} ({pp if metric == 'prefill_tok_s' else tg} tokens)", fontsize=12, loc="left")
        ax.set_ylabel("tokens/s")
        ax.set_xticks(x, [short(labels[c]) for c in commits], rotation=40, ha="right", fontsize=8)
        ax.set_xlim(0.5, len(commits) + 2.2)
        ax.grid(True, which="major", color="#e9ecef")
        ax.grid(True, which="minor", color="#f4f5f7", lw=0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("abacus commit by commit vs llama.cpp Q8_0 (dashed), Qwen3-0.6B on a 16-core Neoverse-V2",
                 fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "ladder.svg")

    print((out / "table.md").read_text())


if __name__ == "__main__":
    main()

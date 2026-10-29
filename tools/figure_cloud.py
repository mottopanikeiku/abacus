"""Plot the retained six-cell cloud comparison from its summary, without new measurements."""
import argparse
import html
import json
import math
from pathlib import Path
import re

from tools.cloud_summary import CONTEXTS, THREADS

LEFT, RIGHT = 230, 590
TOP, ROW = 112, 38


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def host(summary: dict) -> str:
    text = summary["environment"]["lscpu_text"]
    fields = {key: re.search(rf"^{key}:\s*(\S+)", text, re.MULTILINE) for key in ("CPU\\(s\\)", "CPU family", "Model")}
    require(all(fields.values()), "summary lacks CPU count, family or model")
    cpus, family, model = (match.group(1) for match in fields.values())
    return f"One cloud host: {cpus} virtual CPUs, CPUID family {family} model {model}"


def rows(summary: dict) -> list[dict]:
    require(summary.get("schema") == "cpu-decode-cloud-summary", "expected a cloud summary")
    require(summary["matrix"]["complete"], "the cloud matrix is incomplete")
    cells = {(cell["threads"], cell["context"]): cell for cell in summary["cells"]}
    order = [(threads, context) for context in CONTEXTS for threads in THREADS]
    require(sorted(cells) == sorted(order), "expected exactly the six designed cells")
    return [cells[key] for key in order]


def uniform(cells: list[dict], label: str, read) -> str:
    values = {str(read(cell)) for cell in cells}
    require(len(values) == 1, f"cells differ in {label}")
    return values.pop()


def render(summary: dict) -> str:
    cells = rows(summary)
    values = [1.0] + [pair["native_over_llama"] for cell in cells for pair in cell["paired_ratios"]]
    values += [bound for cell in cells for bound in cell["native_over_llama"]["ci95"]]
    low, high = math.floor(min(values) * 10) / 10, math.ceil(max(values) * 10) / 10
    x = lambda value: LEFT + (value - low) / (high - low) * (RIGHT - LEFT)
    bottom = TOP + ROW * len(cells)
    height = bottom + 112
    native = uniform(cells, "native format", lambda cell: f'{cell["native_settings"]["model"]}/{cell["native_settings"]["activation_dtype"]}')
    kv = uniform(cells, "KV dtype", lambda cell: f'{cell["native_settings"]["kv_dtype"]}/{cell["llama_settings"]["kv_dtype"]}')
    require(kv == "f16/f16", "the cloud comparison uses F16 KV in both engines")
    steps = uniform(cells, "steps", lambda cell: cell["steps"])
    title = f"Native {native} over llama.cpp Q8_0 paired decode throughput"
    elements = [f'<svg xmlns="http://www.w3.org/2000/svg" width="820" height="{height}" viewBox="0 0 820 {height}" role="img" aria-labelledby="title">',
                f'<title id="title">{html.escape(title)}: six cells on one cloud host</title>',
                f'<rect width="820" height="{height}" fill="#ffffff"/>',
                '<g font-family="sans-serif" fill="#17212b">',
                f'<text x="24" y="30" font-size="18">{html.escape(title)}</text>',
                f'<text x="24" y="54" font-size="13">{html.escape(host(summary))} · F16 KV · {steps} forwards per sample</text>',
                f'<text x="{LEFT}" y="88" font-size="13">Paired ratio (above 1: native faster)</text>',
                '<text x="610" y="88" font-size="13">Median [95% interval]</text>',
                f'<path d="M {x(1.0):.2f} {TOP - 14} V {bottom}" stroke="#7a8691" stroke-dasharray="4 3"/>']
    for index, cell in enumerate(cells):
        y = TOP + ROW * index + 8
        median = cell["native_over_llama"]["median"]
        lower, upper = cell["native_over_llama"]["ci95"]
        label = f'{cell["threads"]} thread{"s" if cell["threads"] > 1 else ""} · context {cell["context"]}'
        elements.append(f'<text x="24" y="{y + 5}" font-size="13">{label}</text>')
        elements += [f'<circle class="pair" cx="{x(pair["native_over_llama"]):.2f}" cy="{y}" r="2.5" fill="none" stroke="#8a99a6"/>'
                     for pair in cell["paired_ratios"]]
        elements += [f'<rect x="{x(lower):.2f}" y="{y - 3}" width="{x(upper) - x(lower):.2f}" height="6" fill="#29719c" fill-opacity="0.45"/>',
                     f'<circle cx="{x(median):.2f}" cy="{y}" r="5" fill="#29719c"/>',
                     f'<text x="610" y="{y + 5}" font-size="13">{median:.3f}× [{lower:.3f}, {upper:.3f}]</text>']
    ticks = [low + step / 10 for step in range(round((high - low) * 10) + 1)]
    elements.append(f'<path d="M {LEFT} {bottom} H {RIGHT}" stroke="#17212b"/>')
    for tick in ticks[::2] if len(ticks) > 6 else ticks:
        elements += [f'<path d="M {x(tick):.2f} {bottom} v 5" stroke="#17212b"/>',
                     f'<text x="{x(tick):.2f}" y="{bottom + 20}" font-size="12" text-anchor="middle">{tick:.1f}×</text>']
    elements += [f'<text x="24" y="{bottom + 48}" font-size="12">Filled dot: median of {cells[0]["pairs"]} chronological pairs in {cells[0]["blocks"]} ABBA quartets. Bar: individual 95% quartet-bootstrap interval.</text>',
                 f'<text x="24" y="{bottom + 66}" font-size="12">Intervals are per-cell, not simultaneous. Rings: single pairs. Dashed line: parity.</text>',
                 f'<text x="24" y="{bottom + 84}" font-size="12">One host with worker-lifetime CPU binding; not laptop CLI, v3 or other-host performance.</text>',
                 '</g></svg>']
    return "\n".join(elements) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/v2/cloud-vnni/final/summary.json"))
    parser.add_argument("--output", type=Path, default=Path("results/v2/cloud-vnni/ratios.svg"))
    args = parser.parse_args()
    args.output.write_text(render(json.loads(args.input.read_text())))


if __name__ == "__main__":
    main()

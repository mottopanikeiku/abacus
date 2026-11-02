#!/usr/bin/env python3
"""Build the walkthrough page from steps.md, the git history and results/.

usage: python3 walkthrough/build.py OUT_DIR

Standard library only. Each engine commit gets its section of steps.md, the
lines of its diff picked by the section's show comment, and bars of tokens/s
from the saved benchmark runs (median over repetitions).
"""
import html
import json
import math
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = "https://github.com/mottopanikeiku/abacus"
# (results dir, threads, machine name) for the bars
MACHINES = [("results", 8, "Graviton4, 8 threads"), ("results/runners/neoverse-n2", 4, "Neoverse-N2 runner, 4 threads"),
            ("results/runners/epyc-9v74", 4, "EPYC 9V74 runner, 4 threads"),
            ("results/runners/apple-m1", 1, "Apple M1 runner, 1 thread")]
CONTEXT = 12  # most diff lines shown per step


def git(*args):
    return subprocess.run(["git", "-C", ROOT, *args], check=True, capture_output=True, text=True).stdout


def sections():
    """step name -> (show file, show pattern, paragraphs) from steps.md"""
    out = {}
    for block in re.split(r"^## ", (ROOT / "walkthrough" / "steps.md").read_text(), flags=re.M)[1:]:
        name, rest = block.split("\n", 1)
        show = re.match(r'<!-- show (\S+) "(.+)" -->\n', rest)
        text = rest[show.end():] if show else rest
        paras = [" ".join(p.split()) for p in text.strip().split("\n\n")]
        out[name.strip()] = (show.group(1), show.group(2), paras) if show else (None, None, paras)
    return out


def key_lines(sha, path, pattern):
    """The diff hunk of path in commit sha that adds a line containing pattern,
    cut down to CONTEXT lines around that line."""
    diff = git("show", "--format=", "-U4", sha, "--", path)
    for hunk in re.split(r"^@@[^\n]*\n", diff, flags=re.M)[1:]:
        lines = [l for l in hunk.split("\n") if l[:1] in ("+", "-", " ")]
        hit = next((i for i, l in enumerate(lines) if l.startswith("+") and pattern in l), None)
        if hit is not None:
            start = max(0, min(hit - 3, len(lines) - CONTEXT))
            return lines[start:start + CONTEXT]
    return []


def medians(results_dir):
    """(step index, flags, threads) -> (decode, prefill) median tokens/s"""
    path = ROOT / results_dir / "raw.jsonl"
    if not path.exists():
        return {}
    runs = defaultdict(lambda: ([], []))
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r["runner"] == "abacus":
            d, p = runs[r["index"], r.get("flags", ""), r["threads"]]
            d.append(r["decode_tok_s"])
            p.append(r["prefill_tok_s"])
    return {k: (statistics.median(d), statistics.median(p)) for k, (d, p) in runs.items()}


def bar(value, before, top):
    """A log-scale bar from 1 tok/s to top, with the previous step as a tick."""
    width = lambda v: 100 * math.log(max(v, 1)) / math.log(top)  # noqa: E731
    tick = f'<i style="left:{width(before):.1f}%"></i>' if before else ""
    gain = f" <em>{value / before:.2f}x</em>" if before else ""
    return (f'<span class="track"><b style="width:{width(value):.1f}%"></b>{tick}</span>'
            f'<span class="num">{value:,.0f}{gain}</span>')


def bar_rows(label, now, before, top):
    """A row naming the machine, then a bar each for decode and prefill."""
    return [f'<tr><td colspan="2" class="machine">{html.escape(label)}</td></tr>'] + [
        f'<tr><th>{metric}</th><td class="bar">{bar(now[j], before[j] if before else None, top)}</td></tr>'
        for j, metric in enumerate(("decode", "prefill"))]


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "_site")
    text = sections()
    log = git("log", "--reverse", "--format=%H %s", "HEAD", "--", "src").splitlines()
    steps = [(sha, subject.split(":")[0]) for sha, subject in (line.split(" ", 1) for line in log)]
    data = [(medians(d), t, name) for d, t, name in MACHINES]
    tops = {name: max([v for k, vs in m.items() if k[2] == t for v in vs] or [2]) for m, t, name in data}
    lines = sum(len(p.read_text().splitlines()) for p in (ROOT / "src").iterdir())

    body = []
    previous, q4_shown = {}, set()
    for i, (sha, name) in enumerate(steps):
        path, pattern, paras = text.get(name, (None, None, []))
        rows = []
        for m, t, machine in data:
            if (i, "", t) not in m:
                continue
            now = m[i, "", t]
            rows += bar_rows(machine, now, previous.get(machine), tops[machine])
            previous[machine] = now
            if (i, "--q4", t) in m and machine not in q4_shown:  # the step that adds --q4, against int8
                rows += bar_rows(machine + ", --q4", m[i, "--q4", t], now, tops[machine])
                q4_shown.add(machine)
        diff = key_lines(sha, path, pattern) if path else []
        # each line is a block of its own, so a newline between them would add an empty line
        code = "".join(f'<span class="{"add" if l[0] == "+" else "del" if l[0] == "-" else "ctx"}">'
                       f"{html.escape(l)}</span>" for l in diff)
        body.append(f'<section id="{i + 1}"><h2><a href="#{i + 1}">{i + 1}.</a> {html.escape(name)}</h2>\n'
                    + "".join(f"<p>{html.escape(p)}</p>\n" for p in paras)
                    + (f'<figure><figcaption>{html.escape(path)}, <a href="{REPO}/commit/{sha}">'
                       f'full diff</a></figcaption><pre>{code}</pre></figure>\n' if code else "")
                    + (f'<table class="bars">{"".join(rows)}</table>\n' if rows else "")
                    + "</section>")

    page = TEMPLATE.replace("{{steps}}", "\n".join(body)).replace("{{count}}", str(len(steps)))
    page = page.replace("{{lines}}", f"{lines:,}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(page)
    print(f"wrote {out / 'index.html'}: {len(steps)} steps")


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>abacus, one commit at a time</title>
<meta name="description" content="A small C inference engine for Qwen3, built one measured commit at a time.">
<style>
:root { --bg: #fff; --fg: #1d2125; --dim: #5c6670; --line: #e3e6ea; --code: #f6f7f9;
        --add: #e6f4ea; --del: #fbe9eb; --bar: #e8590c; --track: #eef0f3; --link: #1c62c4; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #15181b; --fg: #e3e6ea; --dim: #98a2ad; --line: #2b3036; --code: #1c2024;
          --add: #183a24; --del: #44202a; --bar: #ff8a47; --track: #262b31; --link: #7fb0ff; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg);
       font: 17px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
main { max-width: 760px; margin: 0 auto; padding: 2.5rem 1.2rem 4rem; }
h1 { font-size: 1.9rem; margin: 0 0 .4rem; }
h2 { font-size: 1.25rem; margin: 2.6rem 0 .6rem; padding-top: 1.2rem; border-top: 1px solid var(--line); }
h2 a { color: var(--dim); text-decoration: none; }
a { color: var(--link); }
p { margin: .6rem 0; }
.lede { color: var(--dim); }
figure { margin: 1rem 0; }
figcaption { font-size: .8rem; color: var(--dim); margin-bottom: .2rem; }
pre { background: var(--code); padding: .7rem .8rem; overflow-x: auto; margin: 0; border-radius: 6px;
      font: 12.5px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
pre span { display: block; white-space: pre; }
.add { background: var(--add); }
.del { background: var(--del); }
.ctx { color: var(--dim); }
table.bars { width: 100%; border-collapse: collapse; font-size: .82rem; margin-top: .6rem; }
.bars th { text-align: left; font-weight: 600; width: 4.5rem; padding: .1rem 0; }
.bars td.machine { color: var(--dim); padding: .55rem 0 .1rem; }
.bars td.bar { display: flex; align-items: center; gap: .6rem; }
.track { position: relative; flex: 1; height: .65rem; background: var(--track); border-radius: 3px; }
.track b { position: absolute; inset: 0 auto 0 0; background: var(--bar); border-radius: 3px; }
.track i { position: absolute; top: -.2rem; bottom: -.2rem; width: 2px; background: var(--fg); opacity: .55; }
.num { width: 6.5rem; text-align: right; font-variant-numeric: tabular-nums; }
.num em { font-style: normal; color: var(--dim); }
footer { margin-top: 3rem; color: var(--dim); font-size: .85rem; }
@media (max-width: 560px) { .num { width: 5.6rem; } }
</style>
</head>
<body>
<main>
<h1>abacus, one commit at a time</h1>
<p class="lede">abacus is a CPU inference engine for Qwen3 in {{lines}} lines of C, which started as a
float32 loop and got faster one commit at a time. This page goes through the {{count}} commits that touch
the engine, with what each one does, the lines that do it and the tokens per second it reached on
Qwen3-0.6B. The bars use a log scale that starts at 1 token/s, and the tick on each bar marks the step
before.</p>
<p class="lede">The code and the saved results live at <a href="https://github.com/mottopanikeiku/abacus">github.com/mottopanikeiku/abacus</a>.</p>
{{steps}}
<footer>Decode is 128 tokens and prefill a 128-token prompt, and each bar is the median of interleaved
runs from <code>bench/run.py</code>. The Graviton4 numbers in <code>results/</code> cover steps 1 to 13,
and the GitHub-hosted runners in <code>results/runners/</code> cover step 13 onwards. The page itself is
generated by <code>walkthrough/build.py</code>.</footer>
</main>
</body>
</html>
"""

if __name__ == "__main__":
    main()

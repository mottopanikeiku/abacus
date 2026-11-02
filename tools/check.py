"""Check abacus against the transformers reference written by tools/reference.py.

usage: uv run tools/check.py ABACUS MODEL REF_DIR [--q4] [--json OUT]

The pass rule is the same for every commit:
  tokenizer       ids identical to transformers on every case
  float weights   free-running greedy tokens identical on every prompt, and
                  teacher-forced logits within MAX_ABS_ERROR of the reference
  quantized       teacher-forced argmax agrees with the reference on at least a
                  share of positions, and mean KL stays under a limit; both are
                  in QUANTIZED, by weight format (--q4 runs abacus with --q4)
Teacher forcing runs twice: one token at a time (the decode path) and the
whole sequence in one call (the prefill path).
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

MAX_ABS_ERROR = 0.002
QUANTIZED = {"q8_0": (0.90, 0.02), "q4_0": (0.80, 0.25)}  # minimum argmax agreement, maximum mean KL


def run(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def log_softmax(x):
    x = x.astype(np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def main():
    args = sys.argv[1:]
    json_out = None
    if "--json" in args:
        i = args.index("--json")
        json_out = args[i + 1]
        del args[i:i + 2]
    flags = ["--q4"] if "--q4" in args else []
    args = [a for a in args if a != "--q4"]
    if len(args) != 3:
        sys.exit(__doc__)
    abacus, model, ref_dir = args
    abacus = [abacus, model, *flags]
    ref = json.loads((Path(ref_dir) / "ref.json").read_text())
    ref_logits = np.load(Path(ref_dir) / "logits.npy", mmap_mode="r")
    weights = run([*abacus, "--info"]).split("weights=")[1].split()[0]
    exact = weights in ("f32", "bf16")
    ok = True

    bad = [c["text"] for c in ref["tokenizer"] if run([*abacus, "--tokenize", c["text"]]).split() !=
           [str(i) for i in c["ids"]]]
    print(f"tokenizer: {len(ref['tokenizer']) - len(bad)}/{len(ref['tokenizer'])} cases match")
    for text in bad:
        print(f"  mismatch: {text[:60]!r}")
    ok &= not bad

    report = {"weights": weights, "prompts": []}
    for p, logits in zip(ref["prompts"], ref_logits):
        ids, greedy = p["ids"], p["greedy"]
        out = [int(t) for t in run([*abacus, "--ids", " ".join(map(str, ids)), "-n", str(len(greedy))]).split()]
        same = next((i for i, (a, b) in enumerate(zip(out, greedy)) if a != b), len(greedy))
        row = {"prompt_tokens": len(ids), "greedy_match": same}
        ref_lp = log_softmax(np.asarray(logits))
        for name, chunk in (("decode", 1), ("prefill", 0)):
            with tempfile.NamedTemporaryFile(suffix=".f32") as f:
                seq = " ".join(map(str, ids + greedy[:-1]))
                run([*abacus, "--ids", seq, "--dump-logits", f.name, "--from", str(len(ids)),
                     "--chunk", str(chunk)])
                got = np.fromfile(f.name, dtype=np.float32).reshape(logits.shape)
            lp = log_softmax(got)
            row[name] = {
                "max_abs_error": float(np.abs(got - logits).max()),
                "argmax_agreement": float((got.argmax(-1) == np.asarray(logits).argmax(-1)).mean()),
                "mean_kl": float((np.exp(ref_lp) * (ref_lp - lp)).sum(-1).mean()),
            }
        report["prompts"].append(row)
        d, f = row["decode"], row["prefill"]
        print(f"prompt {len(report['prompts'])} ({len(ids)} tokens): greedy {same}/{len(greedy)} match | "
              f"decode err {d['max_abs_error']:.2g} agree {d['argmax_agreement']:.0%} kl {d['mean_kl']:.2g} | "
              f"prefill err {f['max_abs_error']:.2g} agree {f['argmax_agreement']:.0%} kl {f['mean_kl']:.2g}")

    rows = report["prompts"]
    modes = [r[m] for r in rows for m in ("decode", "prefill")]
    if exact:
        ok &= all(r["greedy_match"] == len(p["greedy"]) for r, p in zip(rows, ref["prompts"]))
        ok &= all(m["max_abs_error"] <= MAX_ABS_ERROR for m in modes)
    else:
        agreement, kl = QUANTIZED[weights]
        for mode in ("decode", "prefill"):
            ok &= np.mean([r[mode]["argmax_agreement"] for r in rows]) >= agreement
            ok &= np.mean([r[mode]["mean_kl"] for r in rows]) <= kl
    report["pass"] = bool(ok)
    print(f"{'PASS' if ok else 'FAIL'} ({weights} weights)")
    if json_out:
        Path(json_out).write_text(json.dumps(report, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

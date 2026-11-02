"""Benchmark every engine commit and llama.cpp the same way, then draw the chart.

usage: uv run bench/run.py [--threads 1,4,8] [--reps 5] [--pp 128] [--tg 128] [--first N] [--lock FILE]

Steps:
  1. build every commit that touches src/ in its own git worktree, from step N on
  2. run tools/check.py against each build (skip with --skip-check)
  3. build llama.cpp at LLAMA_COMMIT and convert the model to GGUF
  4. for each repetition and thread count, run every commit and every llama.cpp
     format once, in a fresh random order, pinned to cores 0..threads-1 on Linux
  5. append one JSON line per run to results/raw.jsonl and call bench/report.py
"""
import argparse
import fcntl
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache"
LLAMA_COMMIT = "785322d17f2d51e5185cf996a1f4e46d87640841"
LLAMA_FORMATS = ["Q8_0", "Q4_0"]
MAC = platform.system() == "Darwin"


def pin(cpus):
    """Linux pins runs to cores; macOS has no way to, so its runs float."""
    return ["taskset", "-c", cpus] if shutil.which("taskset") else []


def sh(cmd, **kw):
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kw).stdout


def ladder():
    """(commit, tree, step name) for every commit that changes the engine, oldest first.
    The step name is the subject up to its first colon; results are keyed by tree."""
    out = sh(["git", "-C", ROOT, "log", "--reverse", "--format=%H %T %s", "HEAD", "--", "src"])
    return [(c, t, s.split(":")[0]) for c, t, s in (line.split(" ", 2) for line in out.splitlines())]


def build(sha):
    binary = CACHE / "bin" / sha
    if not binary.exists():
        tree = CACHE / "worktrees" / sha[:12]
        if not tree.exists():
            sh(["git", "-C", ROOT, "worktree", "add", "--detach", "--force", tree, sha])
        sh(["make", "-C", tree, "-B", "abacus"])
        binary.parent.mkdir(parents=True, exist_ok=True)
        (tree / "abacus").rename(binary)
    return binary


def setup_llama(model_dir, gguf_dir):
    src = CACHE / "llama.cpp"
    if not src.exists():
        sh(["git", "clone", "https://github.com/ggml-org/llama.cpp", src])
    sh(["git", "-C", src, "checkout", "-q", LLAMA_COMMIT])
    bench = src / "build" / "bin" / "llama-bench"
    if not bench.exists():
        # CPU only: on macOS that means no Metal and no Accelerate (BLAS)
        mac = ["-DGGML_METAL=OFF", "-DGGML_BLAS=OFF"] if MAC else []
        sh(["cmake", "-S", src, "-B", src / "build", "-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=ON",
            "-DLLAMA_CURL=OFF", "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_SERVER=OFF", *mac])
        sh(["cmake", "--build", src / "build", "-j", str(os.cpu_count()), "--target", "llama-bench", "llama-quantize"])
    gguf_dir.mkdir(parents=True, exist_ok=True)
    name = model_dir.name
    bf16 = gguf_dir / f"{name}-BF16.gguf"
    if not bf16.exists():
        sh(["uv", "run", "--with", "sentencepiece", "python", src / "convert_hf_to_gguf.py", model_dir,
            "--outtype", "bf16", "--outfile", bf16], env={**os.environ, "PYTHONPATH": str(src / "gguf-py")})
    for fmt in LLAMA_FORMATS:
        out = gguf_dir / f"{name}-{fmt}.gguf"
        if not out.exists():
            sh([src / "build" / "bin" / "llama-quantize", bf16, out, fmt])
    return bench, {fmt: gguf_dir / f"{name}-{fmt}.gguf" for fmt in LLAMA_FORMATS}


def run_abacus(binary, model, threads, pp, tg, flags):
    out = sh([*pin(f"0-{threads - 1}"), binary, model, *flags, "-t", str(threads), "--bench", str(pp), str(tg)])
    r = json.loads(out)
    return r["prefill_tok_s"], r["decode_tok_s"]


def variants(binary):
    """Ways to run a build: int8 always, and --q4 once the engine has it."""
    usage = subprocess.run([binary], capture_output=True, text=True).stderr
    return [[], ["--q4"]] if "--q4" in usage else [[]]


def run_llama(bench, gguf, threads, pp, tg):
    out = sh([*pin(f"0-{threads - 1}"), bench, "-m", gguf, "-t", str(threads), "-p", str(pp),
              "-n", str(tg), "-r", "1", "-o", "jsonl"])
    rows = [json.loads(line) for line in out.splitlines() if line.startswith("{")]
    pp_ts = next(r["avg_ts"] for r in rows if r["n_gen"] == 0)
    tg_ts = next(r["avg_ts"] for r in rows if r["n_prompt"] == 0)
    return pp_ts, tg_ts


def machine(info):
    if MAC:
        cpu = sh(["sysctl", "-n", "machdep.cpu.brand_string"]).strip()
    else:
        cpu = next((line.split(":", 1)[1].strip() for line in sh(["lscpu"]).splitlines()
                    if line.startswith("Model name")), platform.processor())
    return {"cpu": cpu, "cores": os.cpu_count(), "kernel": platform.release(),
            "compiler": sh(["cc", "--version"]).splitlines()[0], "llama_cpp": LLAMA_COMMIT, "abacus": info}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads", default="1,4,8")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--pp", type=int, default=128)
    ap.add_argument("--tg", type=int, default=128)
    ap.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", ROOT / "models" / "Qwen3-0.6B"), type=Path)
    ap.add_argument("--model", default=ROOT / "model.bin", type=Path)
    ap.add_argument("--ref", default=ROOT / "ref", type=Path)
    ap.add_argument("--gguf-dir", default=CACHE / "gguf", type=Path)
    ap.add_argument("--out", default=ROOT / "results", type=Path)
    ap.add_argument("--lock", help="flock this file while measuring")
    ap.add_argument("--max-load", type=float, default=3.0, help="wait until the 1-minute load average is below this")
    ap.add_argument("--skip-check", action="store_true")
    ap.add_argument("--check-cpus", default="0-7", help="cores for the correctness checks")
    ap.add_argument("--first", type=int, default=1, help="first ladder step to check and time")
    ap.add_argument("--no-llama", action="store_true")
    args = ap.parse_args()
    threads = [int(t) for t in args.threads.split(",")]
    model_dir, model = args.model_dir.expanduser().resolve(), args.model.expanduser().resolve()

    steps = list(enumerate(ladder()))[args.first - 1:]
    print(f"{len(steps)} engine commits")
    binaries = {sha: build(sha) for _, (sha, _, _) in steps}
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.skip_check:
        checks, report = [], args.out / "check.tmp.json"
        for i, (sha, tree, name) in steps:
            for flags in variants(binaries[sha]):
                ok = subprocess.run([*pin(args.check_cpus), sys.executable, ROOT / "tools" / "check.py",
                                     binaries[sha], model, args.ref, *flags, "--json", report]).returncode == 0
                checks.append({"index": i, "tree": tree, "step": name, "flags": " ".join(flags),
                               **json.loads(report.read_text())})
                print(f"check {i + 1:2d} {'PASS' if ok else 'FAIL'}  {name} {' '.join(flags)}")
        report.unlink()
        (args.out / "check.json").write_text(json.dumps(checks, indent=1))

    runners = [("abacus", i, sha, tree, f"{name} {' '.join(flags)}".strip(), flags)
               for i, (sha, tree, name) in steps for flags in variants(binaries[sha])]
    if not args.no_llama:
        bench, ggufs = setup_llama(model_dir, args.gguf_dir.expanduser())
        runners += [("llama.cpp", None, fmt, None, f"llama.cpp {fmt}", []) for fmt in LLAMA_FORMATS]

    lock = None
    if args.lock:
        lock = open(args.lock, "w")
        print(f"waiting for {args.lock}")
        fcntl.flock(lock, fcntl.LOCK_EX)
    while os.getloadavg()[0] > args.max_load:  # a busy machine slows spinning thread pools a lot
        print(f"load average {os.getloadavg()[0]:.1f} > {args.max_load}, waiting")
        time.sleep(60)
    last = binaries[steps[-1][1][0]]
    (args.out / "machine.json").write_text(json.dumps(machine(sh([last, model, "--info"]).strip()), indent=1))
    raw = open(args.out / "raw.jsonl", "w")
    print(f"load average {os.getloadavg()[0]:.1f}; {len(runners) * len(threads) * args.reps} runs")
    for rep in range(args.reps):
        for t in threads:
            for kind, index, key, tree, label, flags in random.sample(runners, len(runners)):
                load_s = None
                if kind == "abacus":
                    start = time.perf_counter()  # loading the model, quantization included
                    sh([*pin(f"0-{t - 1}"), binaries[key], model, *flags, "-t", str(t), "--info"])
                    load_s = time.perf_counter() - start
                    pp_ts, tg_ts = run_abacus(binaries[key], model, t, args.pp, args.tg, flags)
                else:
                    pp_ts, tg_ts = run_llama(bench, ggufs[key], t, args.pp, args.tg)
                row = {"runner": kind, "index": index, "tree": tree, "label": label, "flags": " ".join(flags),
                       "threads": t, "rep": rep, "pp": args.pp, "tg": args.tg, "prefill_tok_s": pp_ts,
                       "decode_tok_s": tg_ts, "load_s": load_s, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                       "load": round(os.getloadavg()[0], 2), "run": os.environ.get("GITHUB_RUN_ID", "local")}
                raw.write(json.dumps(row) + "\n")
                raw.flush()
                print(f"rep {rep} t={t:2d} {label[:44]:44s} prefill {pp_ts:8.1f}  decode {tg_ts:7.1f}")
    raw.close()
    if lock:
        lock.close()
    subprocess.run([sys.executable, ROOT / "bench" / "report.py", args.out, args.threads], check=True)


if __name__ == "__main__":
    main()

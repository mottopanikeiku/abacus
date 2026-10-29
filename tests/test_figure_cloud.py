"""The committed cloud figure is regenerated from committed results; other inputs are SIM fixtures."""
import json

import pytest

from test_cloud_summary import synthetic_raw
from tools import cloud_summary as cloud
from tools import figure_cloud as figure
from tools.portable import ROOT


def synthetic_summary(ratio: float = 2.0, *, cells: int = 6) -> dict:
    raw = synthetic_raw(llama_seconds=[ratio] * 16)
    raw["cells"] = raw["cells"][:cells]
    raw["environment"]["lscpu_text"] = "CPU(s): 24\nCPU family: 25\nModel: 17\nModel name: unknown\n"
    for cell in raw["cells"]:
        cell["native_settings"].update(model="g64f16", activation_dtype="int16", kv_dtype="f16")
        cell["llama_settings"]["kv_dtype"] = "f16"
    return cloud.summarize(raw, allow_partial=cells < 6)


def test_committed_figure_matches_committed_summary():
    summary = json.loads((ROOT / "results/v2/cloud-vnni/final/summary.json").read_text())
    assert figure.render(summary) == (ROOT / "results/v2/cloud-vnni/ratios.svg").read_text()


def test_every_cell_pair_and_interval_is_drawn_with_its_scope():
    svg = figure.render(synthetic_summary(2.0))
    assert svg.count('class="pair"') == 6 * 16
    assert svg.count("2.000× [2.000, 2.000]") == 6
    for threads in cloud.THREADS:
        for context in cloud.CONTEXTS:
            assert f'{threads} thread{"s" if threads > 1 else ""} · context {context}' in svg
    assert "One cloud host: 24 virtual CPUs, CPUID family 25 model 17" in svg
    assert "Native g64f16/int16 over llama.cpp Q8_0" in svg
    assert "not simultaneous" in svg and "not laptop CLI" in svg


def test_parity_stays_on_the_axis_when_every_pair_loses():
    svg = figure.render(synthetic_summary(0.5))
    assert "0.500× [0.500, 0.500]" in svg
    assert ">1.0×</text>" in svg


def test_incomplete_or_mixed_matrices_are_rejected():
    with pytest.raises(ValueError, match="incomplete"):
        figure.render(synthetic_summary(cells=5))
    summary = synthetic_summary()
    summary["cells"][0]["native_settings"]["kv_dtype"] = "f32"
    with pytest.raises(ValueError, match="KV dtype"):
        figure.render(summary)

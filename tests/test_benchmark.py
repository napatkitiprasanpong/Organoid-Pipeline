from orgpipe import SimConfig
from orgpipe.benchmark import run_benchmark, summary_table


def test_benchmark_runs_and_tabulates():
    cfg = SimConfig(fs=20_000.0, duration=8.0, n_units=5, amp_range_uv=(100.0, 150.0), stim_interval_s=3.0)
    res = run_benchmark(seeds=(0, 1), base_cfg=cfg, own=("pca+gmm",), progress=lambda *_: None)
    names = {r["sorter"] for r in res["rows"]}
    assert names == {"detector only (no sorting)", "pca+gmm"}
    assert len(res["units"]) > 0
    table = summary_table(res)
    assert "pca+gmm" in table and "detector only" in table and "±" in table


def test_benchmark_survives_a_failing_sorter():
    cfg = SimConfig(fs=20_000.0, duration=4.0, n_units=3, stim_interval_s=None)
    res = run_benchmark(seeds=(0,), base_cfg=cfg, own=("pca+gmm",), spikeinterface=("definitely_not_a_sorter",),
                        progress=lambda *_: None)
    bad = [r for r in res["rows"] if r["sorter"].startswith("spikeinterface:")]
    assert bad and "error" in bad[0]
    assert "failed" in summary_table(res)

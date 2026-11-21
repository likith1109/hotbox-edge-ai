"""Run untouched client benchmarks in their required locations and save evidence.

From the laptop: python experiments/benchmark_mission3.py --location laptop --url PUBLIC_URL
Inside Codespaces: python experiments/benchmark_mission3.py --location codespace --url http://localhost:8000
After collecting both: python experiments/benchmark_mission3.py --location summarise
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time
from urllib.parse import urlparse

import numpy as np
import requests

ROOT = Path("artifacts/team_25/mission3")
STAGES = ["parse", "normalise", "inference", "build"]


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def verify_endpoint(url):
    response = requests.get(url + "/health", timeout=30, allow_redirects=False)
    response.raise_for_status()
    health = response.json()
    assert health == {"status": "ok", "team": 25, "params": 385, "val_mae": 5.31}, health
    response = requests.get(url + "/model_info", timeout=30, allow_redirects=False)
    response.raise_for_status()
    info = response.json()
    assert [(l["in"], l["out"]) for l in info["layers"]] == [(5, 16), (16, 16), (16, 1)]
    assert (info["total_params"], info["total_bytes_f32"], info["total_macs"]) == (385, 1540, 352)
    return {"health": health, "model_info": info}


def run_bench(label, url, no_keepalive=False):
    folder = ROOT / label
    folder.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", str(Path("client/bench.py").resolve()), "--url", url, "--n", "200"]
    if no_keepalive:
        command.append("--no-keepalive")
    setup = dict(command=shlex.join(command), url=url, hostname=platform.node(),
                 python=sys.version, codespace=os.getenv("CODESPACE_NAME"),
                 no_keepalive=no_keepalive, user_guesses_ms={"inference": 1, "public_round_trip": 250},
                 endpoint=verify_endpoint(url),
                 client_sha256=hashlib.sha256(Path("client/bench.py").read_bytes()).hexdigest())
    save(folder / "setup.json", setup)
    print("RUN", setup["command"], flush=True)
    with (folder / "raw.log").open("w") as output:
        subprocess.run(command, cwd=folder, stdout=output, stderr=subprocess.STDOUT, check=True)
    files = sorted(folder.glob("bench_*.csv"))
    assert files, "Benchmark produced no CSV; inspect raw.log"
    with files[-1].open(newline="") as f:
        rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]
    assert len(rows) == 200 and [r["i"] for r in rows] == list(range(200))
    keys = [k for k in rows[0] if k != "i"]
    assert set(keys) == {"round_trip", *STAGES, "server_total", "network_and_framework"}
    assert all(np.isfinite(list(r.values())).all() for r in rows)
    summary = dict(count=200, csv=files[-1].name, first_request=rows[0],
                   percentiles_excluding_first={k: dict(zip(("p50", "p95", "p99"),
                   np.percentile([r[k] for r in rows[1:]], [50, 95, 99]).tolist())) for k in keys})
    save(folder / "summary.json", summary)
    print((folder / "raw.log").read_text(), flush=True)
    return summary


def compare_400(url):
    import pandas as pd
    folder = ROOT / "comparison_400"
    folder.mkdir(exist_ok=True)
    features = ["battery_v", "temp_c", "rings_per_day", "wifi_rssi_dbm", "age_months"]
    frame = pd.read_csv("data/team_25_train.csv")
    mission1 = json.loads(Path("artifacts/team_25/mission1/metrics.json").read_text())
    cleaned = frame.drop(index=mission1["removed_source_indices"]).drop_duplicates(subset=features)
    sample = cleaned.iloc[np.random.default_rng(25).permutation(len(cleaned))[:400]]
    readings = sample[features].to_dict("records")
    assert len(readings) == 400
    save(folder / "readings.json", dict(source_indices=sample.index.tolist(), rows=readings))
    setup = dict(url=url, hostname=platform.node(), endpoint=verify_endpoint(url), count=400,
                 setup="Two fresh requests.Session objects, each warmed by one prediction of one row before timing; persistent HTTP keep-alive in both arms. Sequential singles; batch one POST. Timers include JSON encoding, response decoding and client processing, excluding file I/O and warm-up. Singles run first; one trial per arm, not a statistical comparison.",
                 data_sha256=hashlib.sha256(Path("data/team_25_train.csv").read_bytes()).hexdigest())
    save(folder / "setup.json", setup)
    singles, rows = [], []
    with requests.Session() as session:
        warm = session.post(url + "/predict", json=readings[0], timeout=30)
        warm.raise_for_status()
        started = time.perf_counter()
        for i, reading in enumerate(readings):
            t0 = time.perf_counter()
            response = session.post(url + "/predict", json=reading, timeout=30)
            response.raise_for_status()
            body = response.json()
            elapsed = (time.perf_counter() - t0) * 1000
            singles.append(body["days_left"])
            rows.append(dict(position=i, source_index=int(sample.index[i]),
                             round_trip_ms=elapsed, days_left=body["days_left"], **body["timings_ms"]))
            if (i + 1) % 50 == 0:
                print(f"400-singles comparison: {i + 1}/400 responses", flush=True)
        singles_ms = (time.perf_counter() - started) * 1000
    with requests.Session() as session:
        warm = session.post(url + "/predict_batch", json={"rows": [readings[0]]}, timeout=30)
        warm.raise_for_status()
        started = time.perf_counter()
        response = session.post(url + "/predict_batch", json={"rows": readings}, timeout=30)
        response.raise_for_status()
        batch = response.json()
        batch_ms = (time.perf_counter() - started) * 1000
    assert len(batch["days_left"]) == 400
    difference = np.abs(np.array(singles) - batch["days_left"])
    assert difference.max() <= 0.01 + 1e-9
    with (folder / "singles.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    save(folder / "batch_response.json", batch)
    result = dict(singles_elapsed_ms=singles_ms, batch_elapsed_ms=batch_ms,
                  speedup=singles_ms / batch_ms, count=400, absolute_tolerance_days=0.01,
                  maximum_prediction_difference_days=float(difference.max()), same_order=True)
    save(folder / "summary.json", result)
    print("400 SINGLE VS BATCH:", json.dumps(result), flush=True)


def summarise():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    summaries = {}
    for label in ("public_keepalive", "public_no_keepalive", "codespace_localhost"):
        summaries[label] = json.loads((ROOT / label / "summary.json").read_text())
    folder = ROOT / "public_keepalive"
    with (folder / summaries["public_keepalive"]["csv"]).open(newline="") as f:
        rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)][1:]
    median = summaries["public_keepalive"]["percentiles_excluding_first"]["round_trip"]["p50"]
    eligible = [r for r in rows if r["server_total"] >= sum(r[k] for k in STAGES)
                and r["round_trip"] >= r["server_total"]]
    r = min(eligible, key=lambda row: abs(row["round_trip"] - median))
    components = {k: r[k] for k in STAGES}
    components["server_other"] = r["server_total"] - sum(r[k] for k in STAGES)
    components["network_and_framework"] = r["round_trip"] - r["server_total"]
    assert abs(sum(components.values()) - r["round_trip"]) < 1e-9
    representative = dict(request_index=int(r["i"]), measured_round_trip_ms=r["round_trip"],
                          components_ms=components,
                          selection="Single measured warm request nearest the median with nonnegative residuals; not a sum of marginal medians.",
                          residual_meaning="Network, proxy, HTTP/framework outside run(), JSON serialization and client overhead cannot be separated by these timers.")
    save(ROOT / "representative_breakdown.json", representative)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    left = 0
    for name, value in components.items():
        axes[0].barh([0], [value], left=left, label=name)
        left += value
    axes[0].set(xlabel="Milliseconds", yticks=[], title=f'One representative measured public request: {r["round_trip"]:.3f} ms')
    axes[0].legend(fontsize=8, ncol=3)
    axes[1].barh(list(components)[:-1], list(components.values())[:-1])
    axes[1].set(xlabel="Milliseconds", title="Server stages enlarged (same request)")
    fig.tight_layout()
    fig.savefig(ROOT / "timing_breakdown.png", dpi=160)
    plt.close(fig)
    p = summaries["public_keepalive"]["percentiles_excluding_first"]
    comparison = dict(user_inference_guess_ms=1, user_public_round_trip_guess_ms=250,
                      measured_inference_p50_ms=p["inference"]["p50"],
                      measured_public_round_trip_p50_ms=p["round_trip"]["p50"],
                      inference_difference_ms=p["inference"]["p50"] - 1,
                      public_round_trip_difference_ms=p["round_trip"]["p50"] - 250)
    save(ROOT / "benchmark_summaries.json", summaries)
    save(ROOT / "prediction_comparison.json", comparison)
    print("PREDICTION COMPARISON:", json.dumps(comparison), flush=True)
    print("REPRESENTATIVE BREAKDOWN:", json.dumps(representative), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", choices=["laptop", "codespace", "summarise"], required=True)
    parser.add_argument("--url")
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.location == "summarise":
        summarise()
        return
    assert args.url
    url = args.url.rstrip("/")
    if args.location == "codespace":
        assert os.getenv("CODESPACE_NAME"), "Localhost benchmark must run inside a real Codespace"
        assert url == "http://localhost:8000"
        run_bench("codespace_localhost", url)
    else:
        assert not os.getenv("CODESPACE_NAME"), "Public benchmarks must run on the laptop"
        parsed = urlparse(url)
        assert parsed.scheme == "https" and parsed.hostname.endswith(".app.github.dev")
        run_bench("public_keepalive", url)
        run_bench("public_no_keepalive", url, no_keepalive=True)
        compare_400(url)


if __name__ == "__main__":
    main()

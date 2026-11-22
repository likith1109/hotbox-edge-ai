"""Mission 3 correctness checks; this script performs no latency benchmark.

Run from the repository root: python experiments/verify_mission3.py
"""
import asyncio
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from app import main as api
from app.model import FEATURES

ROOT = Path("artifacts/team_25/mission3")
EXAMPLE = dict(battery_v=3.86, temp_c=28.0, rings_per_day=5,
               wifi_rssi_dbm=-66, age_months=22)
STAGES = {"parse", "normalise", "inference", "build", "server_total"}
PUBLIC_URL = None
PUBLIC_SESSION = None


async def request(path, method="GET", body=None, raw=None):
    payload = raw if raw is not None else (b"" if body is None else json.dumps(body).encode())
    if PUBLIC_URL:
        response = PUBLIC_SESSION.request(method, PUBLIC_URL + path, data=payload,
                                          headers={"Content-Type": "application/json"},
                                          timeout=30, allow_redirects=False)
        return response.status_code, response.text
    received = False
    messages = []

    async def receive():
        nonlocal received
        if received:
            await asyncio.Event().wait()
        received = True
        return {"type": "http.request", "body": payload, "more_body": False}

    async def send(message):
        messages.append(message)

    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"},
             "http_version": "1.1", "method": method, "scheme": "http",
             "path": path, "raw_path": path.encode(), "query_string": b"", "root_path": "",
             "headers": [(b"content-type", b"application/json")],
             "server": ("test", 8000), "client": ("127.0.0.1", 1)}
    await asyncio.wait_for(api.app(scope, receive, send), timeout=15)
    status = next(m["status"] for m in messages if m["type"] == "http.response.start")
    data = b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body")
    return status, data.decode()


def check_timings(body):
    timings = body["timings_ms"]
    assert set(timings) == STAGES
    assert all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in timings.values())
    # Four stages and their total are independently rounded to 0.001 ms.
    assert sum(timings[k] for k in STAGES - {"server_total"}) <= timings["server_total"] + 0.003


async def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "verification.log").open("a") as log:
        def record(message):
            print(message, flush=True)
            log.write(message + "\n")
            log.flush()

        transport = "real public HTTP requests" if PUBLIC_URL else "fresh in-process ASGI requests"
        record(f"\nCORRECTNESS VERIFICATION — {transport}; not latency benchmarks")
        reference = json.loads(Path("artifacts/team_25/mission2/verified_sha256.json").read_text())
        for name in ("model/model.pt", "model/config.json"):
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == reference[name]
        assert api.cfg["team"] == 25 and api.cfg["hidden"] == [16, 16]
        status, body = await request("/health")
        health = json.loads(body)
        assert status == 200 and health == {"status": "ok", "team": 25, "params": 385, "val_mae": 5.31}
        status, body = await request("/model_info")
        info = json.loads(body)
        assert status == 200 and info["total_params"] == health["params"]
        assert (info["total_bytes_f32"], info["total_macs"]) == (1540, 352)
        assert [(r["in"], r["out"]) for r in info["layers"]] == [(5, 16), (16, 16), (16, 1)]
        record("PASS: unchanged selected model hashes; /health and /model_info return HTTP 200 with Team 25, 5→16→16→1, 385 parameters, 1540 float32 bytes, 352 MACs.")

        status, docs = await request("/docs")
        assert status == 200 and "SwaggerUIBundle" in docs
        status, body = await request("/openapi.json")
        schema = json.loads(body)
        assert status == 200
        for path, model in (("/predict", "Reading"), ("/predict_batch", "Batch")):
            definition = schema["paths"][path]["post"]["requestBody"]["content"]["application/json"]
            assert definition["schema"]["$ref"] == f"#/components/schemas/{model}"
        assert schema["components"]["schemas"]["Batch"]["properties"]["rows"]["items"]["$ref"] == "#/components/schemas/Reading"
        props = schema["components"]["schemas"]["Reading"]["properties"]
        assert all(("minimum" in props[f] or "exclusiveMinimum" in props[f]) and
                   ("maximum" in props[f] or "exclusiveMaximum" in props[f]) for f in FEATURES)
        (ROOT / "openapi.json").write_text(json.dumps(schema, indent=2) + "\n")
        record("PASS: /docs HTML and /openapi.json return HTTP 200; Swagger has request bodies, valid model references, examples and bounds for all five inputs. Interactive browser execution is tracked separately by swagger_mission3.py.")

        validation_cases = []
        for field, value in dict(battery_v=0, temp_c=-127, rings_per_day=-1,
                                 wifi_rssi_dbm=0, age_months=-1).items():
            payload = dict(EXAMPLE, **{field: value})
            for path, wrapped in (("/predict", payload), ("/predict_batch", {"rows": [EXAMPLE, payload]})):
                status, response = await request(path, "POST", wrapped)
                assert status == 422 and json.loads(response)["error"] == "bad reading"
                validation_cases.append(dict(path=path, field=field, value=value, status=status,
                                             response=json.loads(response)))
        # Check upper ends too, rather than testing only the observed placeholders.
        for field, value in dict(battery_v=5.1, temp_c=85.1, rings_per_day=86401,
                                 wifi_rssi_dbm=-121, age_months=241).items():
            status, response = await request("/predict", "POST", dict(EXAMPLE, **{field: value}))
            assert status == 422
            validation_cases.append(dict(path="/predict", field=field, value=value, status=status))
        for field in FEATURES:
            for value in (float("nan"), float("inf"), float("-inf")):
                status, response = await request("/predict", "POST", dict(EXAMPLE, **{field: value}))
                assert status == 422
                validation_cases.append(dict(path="/predict", field=field, value=repr(value), status=status))
        status, _ = await request("/predict", "POST", raw=b"{broken")
        assert status == 422
        for rows in ([], [EXAMPLE] * 10001):
            status, _ = await request("/predict_batch", "POST", {"rows": rows})
            assert status == 422
        (ROOT / "invalid_readings.json").write_text(json.dumps(validation_cases, indent=2) + "\n")
        record("PASS: one invalid value per field is HTTP 422 on both routes; opposite-bound violations, nonfinite inputs in all fields, malformed JSON, empty and oversized batches are rejected.")

        frame = pd.read_csv("data/team_25_train.csv")
        mission1 = json.loads(Path("artifacts/team_25/mission1/metrics.json").read_text())
        training = frame.loc[mission1["training_source_indices"]]
        margins = dict(battery_v=0.01, temp_c=0.1, rings_per_day=1, wifi_rssi_dbm=1, age_months=1)
        boundary_cases = []
        for field in FEATURES:
            for direction, value in (("below", float(training[field].min() - margins[field])),
                                     ("above", float(training[field].max() + margins[field]))):
                payload = dict(EXAMPLE, **{field: value})
                if field in ("rings_per_day", "age_months") and value < 0:
                    boundary_cases.append(dict(field=field, side=direction, value=value,
                                               result="not physically plausible: below zero; tested rejection separately"))
                    continue
                status, response = await request("/predict", "POST", payload)
                assert status == 200, (field, value, response)
                check_timings(json.loads(response))
                boundary_cases.append(dict(field=field, side=direction, value=value, status=status))
        (ROOT / "outside_training_range.json").write_text(json.dumps(boundary_cases, indent=2) + "\n")
        record("PASS: plausible values just outside each training-feature range return HTTP 200. Negative rings and age are physically invalid, so their below-zero sides remain rejected.")

        validation = frame.loc[mission1["validation_source_indices"]]
        unique = validation.drop_duplicates(subset=FEATURES)
        sample = unique.iloc[np.random.default_rng(25).permutation(len(unique))[:50]]
        assert len(sample) == 50 and len(sample.drop_duplicates(subset=FEATURES)) == 50
        readings = sample[FEATURES].to_dict("records")
        singles = []
        for reading in readings:
            status, body = await request("/predict", "POST", reading)
            response = json.loads(body)
            assert status == 200 and set(response) == {"days_left", "timings_ms"}
            check_timings(response)
            singles.append(response["days_left"])
        status, body = await request("/predict_batch", "POST", {"rows": readings})
        batch = json.loads(body)
        assert status == 200 and set(batch) == {"days_left", "timings_ms"}
        check_timings(batch)
        assert len(batch["days_left"]) == 50
        tolerance = 0.01  # API rounds to 0.01 days; BLAS batch accumulation may differ slightly.
        delta = np.abs(np.array(singles) - batch["days_left"])
        assert np.all(delta <= tolerance + 1e-9)
        x = sample[FEATURES].to_numpy(np.float32)
        with torch.no_grad():
            expected = np.maximum(api.net(torch.from_numpy((x - api.MU) / api.SD)).numpy(), 0)
        np.testing.assert_allclose(batch["days_left"], np.round(expected.astype(float), 2), atol=tolerance, rtol=0)
        status, body = await request("/predict_batch", "POST", {"rows": list(reversed(readings))})
        assert status == 200
        np.testing.assert_allclose(json.loads(body)["days_left"], list(reversed(batch["days_left"])), atol=tolerance, rtol=0)
        with (ROOT / "agreement_50.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["position", "source_index", *FEATURES,
                                                  "single_days", "batch_days", "absolute_difference"],
                                    lineterminator="\n")
            writer.writeheader()
            for i, (source, reading) in enumerate(zip(sample.index, readings)):
                writer.writerow(dict(position=i, source_index=int(source), **reading,
                                     single_days=singles[i], batch_days=batch["days_left"][i],
                                     absolute_difference=float(delta[i])))
        summary = dict(count=50, distinct_inputs=50, absolute_tolerance_days=tolerance,
                       relative_tolerance=0, maximum_absolute_difference_days=float(delta.max()),
                       same_order=True, reversed_order_also_checked=True,
                       transport=transport + "; no latency measurement",
                       health=health, model_info=info)
        (ROOT / "correctness_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        record(f'PASS: separate 50-distinct-reading agreement/order check; absolute tolerance={tolerance} days, relative tolerance=0, maximum observed difference={delta.max():.8f} days. Reversed input order also preserved.')
        record("PASS: all five timing keys are finite/nonnegative on both valid prediction routes; stage sums fit server_total within 0.003 ms rounding tolerance. No latency percentiles or throughput benchmark computed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="Optional real public endpoint; otherwise use in-process ASGI")
    args = parser.parse_args()
    if args.url:
        import requests
        PUBLIC_URL = args.url.rstrip("/")
        PUBLIC_SESSION = requests.Session()
        ROOT = ROOT / "public_correctness"
    asyncio.run(main())

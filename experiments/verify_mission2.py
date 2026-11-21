"""Verify saved experiments and exercise the fresh API through ASGI, without sockets.

Run from the repository root: python experiments/verify_mission2.py
"""
import ast
import asyncio
import hashlib
import json
from pathlib import Path
import subprocess
import sys

# Permit running the file as a script from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from PIL import Image

from app.model import BatteryNet, FEATURES, describe_model
from app.preprocessing import normalise_inputs, training_statistics


async def asgi_request(app, path, method="GET", body=None):
    payload = b"" if body is None else json.dumps(body).encode()
    sent = []
    received = False

    async def receive():
        nonlocal received
        if received:
            # A live connection waits for another message after its body, rather
            # than repeatedly delivering the same request in a tight loop.
            await asyncio.Event().wait()
        received = True
        return {"type": "http.request", "body": payload, "more_body": False}

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"}, "http_version": "1.1",
             "method": method, "scheme": "http", "path": path,
             "raw_path": path.encode(), "query_string": b"", "root_path": "",
             "headers": [(b"content-type", b"application/json")],
             "server": ("test", 8000), "client": ("127.0.0.1", 1)}
    await asyncio.wait_for(app(scope, receive, send), timeout=15)
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    data = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    assert status == 200, (path, status, data)
    return json.loads(data)


def main():
    root = Path("artifacts/team_25/mission2")
    reference = json.loads(Path("artifacts/team_25/mission1/metrics.json").read_text())
    raw = pd.read_csv("data/team_25_train.csv")
    tr = raw.loc[reference["training_source_indices"]]
    val = raw.loc[reference["validation_source_indices"]]
    x, y = tr[FEATURES].to_numpy(np.float32), tr.days_left.to_numpy(np.float32)
    xv, yv = val[FEATURES].to_numpy(np.float32), val.days_left.to_numpy(np.float32)
    torch.set_num_threads(1)
    with (root / "verification.log").open("a") as log:
        def record(message):
            print(message, flush=True)
            log.write(message + "\n")
            log.flush()

        record("\nINDEPENDENT SAVED-ARTIFACT AND API CHECKS")
        for name in ("train.py", "app/main.py", "app/model.py", "app/preprocessing.py",
                     "experiments/mission2.py", "experiments/verify_mission2.py"):
            ast.parse(Path(name).read_text(), filename=name)
        subprocess.run(["git", "diff", "--check"], check=True)
        record("PASS: changed Python syntax and git diff --check.")
        # Pin the Mission 1 reference; HEAD advances after each mission.
        original = subprocess.check_output(["git", "show", "e1ed7b4:train.py"], text=True)
        split = original[original.index("rng = np.random.default_rng(0)"):original.index("# TODO Mission 2:")]
        assert split in Path("train.py").read_text()
        record("PASS: cleaned split block is unchanged from Mission 1.")
        data_hash = hashlib.sha256(Path("data/team_25_train.csv").read_bytes()).hexdigest()
        expected_hash = json.loads(Path("artifacts/team_25/mission1/verified_artifact_sha256.json").read_text())["data/team_25_train.csv"]
        assert data_hash == expected_hash
        record("PASS: Team 25 CSV hash matches Mission 1; no other team's data loaded.")

        constant = np.array([[2, 1, 0], [2, 3, 1], [2, 5, 2]], np.float32)
        c_mu, c_sd = training_statistics(constant)
        assert c_mu[0] == 2 and c_sd[0] == 1
        z = normalise_inputs(constant, c_mu, c_sd)
        assert np.isfinite(z).all() and np.array_equal(z[:, 0], np.zeros(3))
        record("PASS: constant-feature fixture gives scale=1 and zero finite centred values (fixture is not used for model training).")

        expected_mu = x.astype(np.float64).mean(axis=0).astype(np.float32)
        expected_sd = x.astype(np.float64).std(axis=0).astype(np.float32)
        expected_sd[expected_sd == 0] = 1
        results = pd.read_csv(root / "results.csv")
        assert len(results) == 11
        expected_architectures = {str(w) for w in (4, 8, 16, 32, 64)} | {f"{w},{w}" for w in (4, 8, 16, 32, 64)}
        assert set(results.loc[results.normalised, "hidden"]) == expected_architectures
        assert len(results.loc[~results.normalised]) == 1

        for output in sorted(root.iterdir()):
            if not output.is_dir() or not (output / "metrics.json").exists():
                continue
            m = json.loads((output / "metrics.json").read_text())
            cfg = json.loads((output / "model/config.json").read_text())
            assert m["epochs"] == 1500 and m["torch_seed"] == 0 and m["split_seed"] == 0
            assert m["training_source_indices"] == reference["training_source_indices"]
            assert m["validation_source_indices"] == reference["validation_source_indices"]
            assert m["removed_source_indices"] == reference["removed_source_indices"]
            assert cfg["team"] == 25 and cfg["features"] == FEATURES and cfg["hidden"] == m["hidden"]
            mu, sd = np.array(cfg["mu"], np.float32), np.array(cfg["sd"], np.float32)
            assert cfg["mu"] == m["mu"] and cfg["sd"] == m["sd"]
            if m["normalised"]:
                np.testing.assert_array_equal(mu, expected_mu)
                np.testing.assert_array_equal(sd, expected_sd)
                np.testing.assert_allclose(normalise_inputs(x, mu, sd).mean(axis=0), 0, atol=2e-6)
            else:
                np.testing.assert_array_equal(mu, np.zeros(5))
                np.testing.assert_array_equal(sd, np.ones(5))
            net = BatteryNet(tuple(cfg["hidden"]))
            net.load_state_dict(torch.load(output / "model/model.pt", weights_only=True))
            net.eval()
            with torch.no_grad():
                train_mae = float(np.abs(net(torch.tensor((x - mu) / sd)).numpy() - y).mean())
                val_mae = float(np.abs(net(torch.tensor((xv - mu) / sd)).numpy() - yv).mean())
            np.testing.assert_allclose([train_mae, val_mae], [m["train_mae"], m["val_mae"]], atol=1e-6)
            assert cfg["val_mae"] == round(val_mae, 2)
            info = describe_model(net)
            assert info["total_params"] == sum(p.numel() for p in net.parameters()) == m["params"]
            assert info["total_bytes_f32"] == sum(p.numel() * p.element_size() for p in net.parameters())
            dimensions = [5, *cfg["hidden"], 1]
            assert info["total_macs"] == sum(a * b for a, b in zip(dimensions, dimensions[1:]))
            assert [(l["in"], l["out"]) for l in info["layers"]] == list(zip(dimensions, dimensions[1:]))
            record(f'PASS {output.name}: saved stats, feature order, model counts, train/validation MAE reproduced; validation={val_mae:.6f} days.')

        from app import main as api
        cfg = json.loads(Path("model/config.json").read_text())
        np.testing.assert_array_equal(api.MU, expected_mu)
        np.testing.assert_array_equal(api.SD, expected_sd)
        captured = []
        handle = api.net.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().numpy().copy()))

        async def check_api():
            health = await asgi_request(api.app, "/health")
            info = await asgi_request(api.app, "/model_info")
            assert health["team"] == 25 and health["status"] == "ok"
            assert info["total_params"] == health["params"]
            assert info == describe_model(api.net)
            assert set(info) == {"layers", "total_params", "total_bytes_f32", "total_bytes_int8", "total_macs", "fits_300kb_sram_f32", "fits_300kb_sram_int8"}
            samples = [{f: float(val.iloc[i][f]) for f in reversed(FEATURES)} for i in range(5)]
            single = []
            for i, sample in enumerate(samples):
                response = await asgi_request(api.app, "/predict", "POST", sample)
                np.testing.assert_array_equal(captured[-1], ((xv[i:i+1] - expected_mu) / expected_sd))
                with torch.no_grad():
                    raw_prediction = float(api.net(torch.tensor((xv[i:i+1] - expected_mu) / expected_sd))[0])
                assert response["days_left"] == round(max(raw_prediction, 0), 2)
                single.append(response["days_left"])
            batch = await asgi_request(api.app, "/predict_batch", "POST", {"rows": samples})
            np.testing.assert_allclose(batch["days_left"], single, rtol=0, atol=0.01)
            return {"health": health, "model_info": info, "single_predictions": single,
                    "batch_predictions": batch["days_left"], "transport": "fresh in-process ASGI requests"}

        responses = asyncio.run(check_api())
        handle.remove()
        (root / "api_verification.json").write_text(json.dumps(responses, indent=2) + "\n")
        record("PASS: fresh API ASGI requests return HTTP 200 for /health, /model_info, /predict and /predict_batch. Captured inference inputs exactly match training-only saved statistics despite reversed JSON key order.")
        chosen = json.loads((root / "selection.json").read_text())
        for name in ("model.pt", "config.json"):
            assert (Path("model") / name).read_bytes() == (root / "selected_retrain/model" / name).read_bytes()
        with Image.open(root / "validation_mae_vs_parameters.png") as im:
            im.verify()
        gap = chosen["retrain"]["val_mae"] - chosen["retrain"]["train_mae"]
        record(f'Selected architecture {cfg["hidden"]}: validation minus training MAE = {gap:.6f} days. A positive gap suggests some generalisation loss; a single split does not establish its stability. Model selection reused validation, so its score may be optimistic.')
        record('Training-only means: ' + json.dumps(cfg["mu"]))
        record('Training-only safe scales: ' + json.dumps(cfg["sd"]))
        record('PASS: root weights/config exactly match the selected retrain archive; parameter plot decodes.')
        hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                  [Path("model/model.pt"), Path("model/config.json"), Path("train.py"),
                   Path("app/main.py"), Path("app/model.py"), Path("app/preprocessing.py")]}
        (root / "verified_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
        record("Mission 2 artifact and API checks passed. Mission 3 input ranges and stage timing remain to be implemented.")


if __name__ == "__main__":
    main()

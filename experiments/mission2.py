"""Reproduce Team 25's fixed-budget sweep, compact selection, and final retrain.

Run from the repository root: python experiments/mission2.py
Each experiment keeps its own weights, config, metrics, and console output.
"""
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    root = Path("artifacts/team_25/mission2")
    root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               MPLCONFIGDIR="/tmp/hotbox-team25-matplotlib")
    rows = []
    commands = []
    reference = json.loads(Path("artifacts/team_25/mission1/metrics.json").read_text())
    with (root / "verification.log").open("w") as log:
        def record(message):
            print(message, flush=True)
            log.write(message + "\n")
            log.flush()

        record("MISSION 2 — TEAM 25 HOTBOX")
        record("Started UTC: " + datetime.datetime.now(datetime.timezone.utc).isoformat())
        record("User hand calculation checked: layers 192/1056/33 parameters; total 1281 parameters, 5124 float32 bytes, 1216 weight MACs.")
        record("User prediction: normalised 32,32 validation MAE = 15 days.")
        record("Only data/team_25_train.csv is used. Cleaning and seeded split are unchanged from Mission 1.")
        record("Controls: torch seed=0, split seed=0, epochs=1500, Adam lr=0.003, batch size=128, one CPU math thread.")
        record("Selection policy fixed before measuring: choose the fewest parameters within 0.5 days of the lowest normalised validation MAE; break ties by validation MAE. This tolerance is an engineering choice, not a significance test.")
        provenance = {
            "python": sys.version, "executable": sys.executable,
            "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [Path("train.py"), Path("app/model.py"),
                                        Path("app/preprocessing.py"), Path("data/team_25_train.csv")]},
        }
        (root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

        def run(hidden, normalised=True, final=False):
            label = "selected_retrain" if final else hidden.replace(",", "-") + ("_normalised" if normalised else "_raw")
            output = root / label
            output.mkdir(exist_ok=True)
            model_dir = Path("model") if final else output / "model"
            args = ["train.py", "--team", "25", "--hidden", hidden,
                    "--epochs", "1500", "--skip-plots", "--output-dir", str(output),
                    "--model-dir", str(model_dir)]
            if not normalised:
                args.append("--no-normalise")
            command = shlex.join(["python", *args])
            commands.append(command)
            (root / "commands.sh").write_text(
                "#!/usr/bin/env bash\nset -euo pipefail\nexport OMP_NUM_THREADS=1 MKL_NUM_THREADS=1\n" +
                "\n".join(commands) + "\n")
            record("COMMAND: " + command)
            with (output / "training.log").open("w") as console:
                subprocess.run([sys.executable, "-u", *args], env=env,
                               stdout=console, stderr=subprocess.STDOUT, check=True)
            metrics = json.loads((output / "metrics.json").read_text())
            for key in ("training_source_indices", "validation_source_indices", "removed_source_indices"):
                assert metrics[key] == reference[key], f"Split or cleaning changed: {label} {key}"
            cfg = json.loads((model_dir / "config.json").read_text())
            assert cfg["team"] == 25 and cfg["hidden"] == metrics["hidden"]
            assert cfg["mu"] == metrics["mu"] and cfg["sd"] == metrics["sd"]
            result = {"run": label, "hidden": hidden, "normalised": normalised,
                      "epochs": metrics["epochs"], "params": metrics["params"],
                      "bytes_f32": 4 * metrics["params"],
                      "train_mae": metrics["train_mae"], "val_mae": metrics["val_mae"],
                      "command": command}
            record(f'RESULT {label}: params={result["params"]}, train MAE={result["train_mae"]:.6f}, validation MAE={result["val_mae"]:.6f} days; split/statistics checks passed.')
            return result

        rows.append(run("32,32", normalised=False))
        for width in (4, 8, 16, 32, 64):
            for hidden in (str(width), f"{width},{width}"):
                rows.append(run(hidden))
                pd.DataFrame(rows).to_csv(root / "results.csv", index=False)
        normalised = [r for r in rows if r["normalised"]]
        best = min(r["val_mae"] for r in normalised)
        chosen = min((r for r in normalised if r["val_mae"] <= best + 0.5),
                     key=lambda r: (r["params"], r["val_mae"]))

        fig, ax = plt.subplots(figsize=(9, 6))
        for depth, label in ((1, "One hidden layer"), (2, "Two hidden layers")):
            group = sorted((r for r in normalised if len(r["hidden"].split(",")) == depth),
                           key=lambda r: r["params"])
            ax.plot([r["params"] for r in group], [r["val_mae"] for r in group], "o-", label=label)
            for r in group:
                ax.annotate(r["hidden"], (r["params"], r["val_mae"]),
                            xytext=(4, 5), textcoords="offset points", fontsize=8)
        ax.scatter([chosen["params"]], [chosen["val_mae"]], marker="*", s=200,
                   color="black", label="Selected compact model", zorder=5)
        ax.set(xscale="log", xlabel="Parameters (log scale)", ylabel="Validation MAE (days)",
               title="Team 25: normalised architectures, fixed 1,500 epochs and seed 0")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(root / "validation_mae_vs_parameters.png", dpi=160)
        plt.close(fig)
        comparison = [rows[0], next(r for r in normalised if r["hidden"] == "32,32")]
        pd.DataFrame(comparison).to_csv(root / "normalisation_comparison.csv", index=False)
        record("Normalised 32,32 is shared by the controlled comparison and the ten-model sweep: 11 unique experiments in total, followed by a separate selected-model retrain.")
        record(f'SELECTED {chosen["hidden"]}: {chosen["params"]} parameters, train MAE {chosen["train_mae"]:.6f}, validation MAE {chosen["val_mae"]:.6f} days; best observed validation MAE {best:.6f}.')
        final = run(chosen["hidden"], final=True)
        assert abs(final["val_mae"] - chosen["val_mae"]) < 1e-6
        assert abs(final["train_mae"] - chosen["train_mae"]) < 1e-6
        # Preserve the regenerated pair as well as the root pair served by the API.
        (root / "selected_retrain/model").mkdir(exist_ok=True)
        for name in ("model.pt", "config.json"):
            shutil.copy2(Path("model") / name, root / "selected_retrain/model" / name)
        estimate = math.ceil(chosen["val_mae"] + 1)
        selection = {"policy": "fewest parameters within 0.5 days of best validation MAE",
                     "chosen": chosen, "retrain": final,
                     "unseen_mae_estimate_days": estimate,
                     "estimate_caveat": "Informal estimate: validation MAE plus one day, rounded up. Not measured, not a confidence bound; ten-model selection reuses validation and can be optimistic."}
        (root / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
        record(f'UNSEEN MAE ESTIMATE: {estimate} days (informal, not measured or a calibrated confidence bound).')
        record("Selected artifacts regenerated; separate saved-weight and API verification follows.")


if __name__ == "__main__":
    main()

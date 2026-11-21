"""
train.py -- STARTER. Clean the data, fit baselines, train the network, save it for the server.

    python train.py --team 7 --hidden 32,32
Writes model/model.pt and model/config.json, which app/main.py loads.

Look for TODO. The training loop is given; the decisions are yours.
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from app.model import FEATURES, BatteryNet

ap = argparse.ArgumentParser()
ap.add_argument("--team", type=int, required=True)
ap.add_argument("--hidden", default="32,32")
ap.add_argument("--epochs", type=int, default=1500)
ap.add_argument("--output-dir", help="Directory for data plots and baseline measurements")
a = ap.parse_args()
hidden = tuple(int(h) for h in a.hidden.split(","))

df = pd.read_csv(f"data/team_{a.team:02d}_train.csv")
output_dir = Path(a.output_dir or f"artifacts/team_{a.team:02d}/mission1")
output_dir.mkdir(parents=True, exist_ok=True)
raw = df.copy()
# These rules identify the observed placeholders without trimming plausible tails.
# In this doorbell log, zero battery voltage and zero RSSI are treated as invalid
# readings; -127 C is treated as a temperature error placeholder. These are
# log-specific assumptions, not universal sensor specifications.
rules = {
    "nonfinite": ~np.isfinite(df[FEATURES + ["days_left"]].to_numpy()).all(axis=1),
    "battery_nonpositive": (df.battery_v <= 0).to_numpy(),
    "temperature_error_minus127": (df.temp_c == -127).to_numpy(),
    "rings_negative": (df.rings_per_day < 0).to_numpy(),
    "wifi_nonnegative": (df.wifi_rssi_dbm >= 0).to_numpy(),
    "age_negative": (df.age_months < 0).to_numpy(),
    "target_negative": (df.days_left < 0).to_numpy(),
}
bad = np.logical_or.reduce(list(rules.values()))
for name, mask in rules.items():
    print(f"glitch rule {name}: {int(mask.sum())} rows")
print(f"{len(df)} rows, {bad.sum()} glitch rows dropped")
df = df[~bad]

# Show raw and retained data separately so placeholders cannot hide the shape
# of the plausible distribution. These plots include all six columns.
for column in raw.columns:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, frame, title in zip(axes, (raw, df), ("Raw", "After cleaning")):
        values = frame[column].to_numpy()
        ax.hist(values[np.isfinite(values)], bins=30, edgecolor="white")
        ax.set(xlabel=column, ylabel="Rows", title=title)
    fig.suptitle(f"Team {a.team}: {column}")
    fig.tight_layout()
    fig.savefig(output_dir / f"hist_{column}.png", dpi=150)
    plt.close(fig)
for feature in FEATURES:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(raw.loc[~bad, feature], raw.loc[~bad, "days_left"],
                    s=8, alpha=0.35, label="Retained")
    axes[0].scatter(raw.loc[bad, feature], raw.loc[bad, "days_left"],
                    s=25, marker="x", color="tab:red", label="Removed")
    axes[0].legend()
    axes[1].scatter(df[feature], df.days_left, s=8, alpha=0.35)
    for ax, title in zip(axes, ("Raw, flagged rows marked", "After cleaning")):
        ax.set(xlabel=feature, ylabel="days_left (days)", title=title)
    fig.suptitle(f"Team {a.team}: days_left against {feature}")
    fig.tight_layout()
    fig.savefig(output_dir / f"scatter_days_left_vs_{feature}.png", dpi=150)
    plt.close(fig)

rng = np.random.default_rng(0)
idx = rng.permutation(len(df)); n_val = len(df) // 5
val, tr = df.iloc[idx[:n_val]], df.iloc[idx[n_val:]]
X, y = tr[FEATURES].to_numpy(np.float32), tr.days_left.to_numpy(np.float32)
Xv, yv = val[FEATURES].to_numpy(np.float32), val.days_left.to_numpy(np.float32)
# TODO Mission 2: normalisation. Why must mu and sd come from the TRAINING rows only?
mu, sd = np.zeros(X.shape[1], np.float32), np.ones(X.shape[1], np.float32)

mae = lambda p, t: float(np.mean(np.abs(p - t)))
mean_mae = mae(np.full_like(yv, y.mean()), yv)
print(f"baseline (predict the mean): {mean_mae:.2f} days")
# The final column of ones supplies an intercept. Only training rows fit beta.
design = np.column_stack((X, np.ones(len(X))))
design_val = np.column_stack((Xv, np.ones(len(Xv))))
beta, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
linear_predictions = design_val @ beta
linear_mae = mae(linear_predictions, yv)
print(f"baseline (linear regression): {linear_mae:.2f} days")
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(yv, linear_predictions, s=14, alpha=0.5)
limits = (float(min(yv.min(), linear_predictions.min())),
          float(max(yv.max(), linear_predictions.max())))
ax.plot(limits, limits, "k--", label="Perfect prediction")
ax.set(xlabel="Actual days_left (days)", ylabel="Predicted days_left (days)",
       title=f"Team {a.team}: linear baseline, validation rows",
       xlim=limits, ylim=limits, aspect="equal")
ax.legend()
fig.tight_layout()
fig.savefig(output_dir / "linear_predicted_vs_actual.png", dpi=150)
plt.close(fig)
pd.DataFrame({"source_index": val.index, "actual": yv,
              "linear_prediction": linear_predictions,
              "absolute_error": np.abs(linear_predictions - yv)}).to_csv(
                  output_dir / "linear_validation_predictions.csv", index=False)
metrics = {
    "team": a.team, "raw_rows": len(raw), "rule_counts": {
        name: int(mask.sum()) for name, mask in rules.items()},
    "unique_rows_removed": int(bad.sum()), "removed_source_indices": raw.index[bad].tolist(),
    "clean_rows": len(df), "split_seed": 0, "training_source_indices": tr.index.tolist(),
    "validation_source_indices": val.index.tolist(), "training_rows": len(tr),
    "validation_rows": len(val), "training_target_mean": float(y.mean()),
    "mean_baseline_val_mae": mean_mae, "linear_baseline_val_mae": linear_mae,
    "linear_coefficients": beta.tolist(), "linear_design_rank": int(rank),
}

torch.manual_seed(0)
net = BatteryNet(hidden)
opt = torch.optim.Adam(net.parameters(), lr=3e-3)
Xt, yt = torch.tensor((X - mu) / sd), torch.tensor(y)
Xvt = torch.tensor((Xv - mu) / sd)
for ep in range(a.epochs):
    perm = torch.randperm(len(Xt))
    for i in range(0, len(Xt), 128):
        b = perm[i:i + 128]
        loss = torch.nn.functional.mse_loss(net(Xt[b]), yt[b])
        opt.zero_grad(); loss.backward(); opt.step()
    if ep % 300 == 0 or ep == a.epochs - 1:
        with torch.no_grad():
            print(f"  epoch {ep:4d}  val MAE {mae(net(Xvt).numpy(), yv):.2f} days")
with torch.no_grad():
    val_mae = mae(net(Xvt).numpy(), yv)
print(f"network {hidden}:            {val_mae:.2f} days")
metrics["cleaned_network_val_mae"] = val_mae
(output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

os.makedirs("model", exist_ok=True)
torch.save(net.state_dict(), "model/model.pt")
json.dump(dict(team=a.team, hidden=list(hidden), mu=mu.tolist(), sd=sd.tolist(),
               features=FEATURES, val_mae=round(val_mae, 2)), open("model/config.json", "w"), indent=2)
print("saved model/model.pt and model/config.json")

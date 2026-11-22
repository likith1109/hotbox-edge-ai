"""model.py -- the network, shared by train.py and the server so both build the same thing."""
import torch
from torch import nn

FEATURES = ["battery_v", "temp_c", "rings_per_day", "wifi_rssi_dbm", "age_months"]
# Explicit binary capacities used for the assignment's parameter-storage tests.
SRAM_BUDGET_BYTES = 300 * 1024
PSRAM_BUDGET_BYTES = 8 * 1024 * 1024


class BatteryNet(nn.Module):
    def __init__(self, hidden=(32, 32)):
        super().__init__()
        layers, n_in = [], len(FEATURES)
        for h in hidden:
            layers += [nn.Linear(n_in, h), nn.ReLU()]
            n_in = h
        layers.append(nn.Linear(n_in, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def describe_model(net):
    """Count Linear layers; bias additions and ReLU do not add weight MACs."""
    layers = []
    for layer in net.net:
        if isinstance(layer, nn.Linear):
            params = sum(p.numel() for p in layer.parameters())
            layers.append({
                "name": f"linear{len(layers) + 1}",
                "in": layer.in_features, "out": layer.out_features,
                "params": params, "bytes_f32": 4 * params,
                "macs": layer.in_features * layer.out_features,
            })
    params = sum(layer["params"] for layer in layers)
    # These are parameter-storage checks, not a whole-runtime RAM guarantee.
    return {
        "layers": layers, "total_params": params,
        "total_bytes_f32": 4 * params, "total_bytes_int8": params,
        "total_macs": sum(layer["macs"] for layer in layers),
        "fits_300kb_sram_f32": 4 * params <= SRAM_BUDGET_BYTES,
        "fits_300kb_sram_int8": params <= SRAM_BUDGET_BYTES,
    }

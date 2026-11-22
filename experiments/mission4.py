"""Reproduce Mission 4 storage, activation, width-boundary and time estimates.

Run from the repository root: python experiments/mission4.py [--url PUBLIC_URL]
No model is trained, quantised, or overwritten by this script.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from app.model import (BatteryNet, FEATURES, PSRAM_BUDGET_BYTES,
                       SRAM_BUDGET_BYTES, describe_model)

ROOT = Path("artifacts/team_25/mission4")
SOURCES = {
    "float32_dot_product": {
        "url": "https://docs.espressif.com/projects/esp-dsp/en/latest/esp32/esp-dsp-benchmarks.html",
        "fact": "ESP32S3 O2 optimized dsps_dotprod_f32, N=256: 432 CPU cycles.",
    },
    "clock_frequency": {
        "url": "https://docs.espressif.com/projects/esp-idf/en/stable/esp32s3/api-reference/system/power_management.html",
        "fact": "ESP_PM_CPU_FREQ_MAX supports 80, 160 or 240 MHz on ESP32-S3.",
    },
    "int8_bias_type": {
        "url": "https://github.com/espressif/esp-nn/blob/master/include/esp_nn_esp32s3.h",
        "fact": "esp_nn_fully_connected_s8_esp32s3 takes int8 input/filter and int32 bias.",
    },
}


def save(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def maximum_width(budget, bytes_per_parameter):
    # P(w) = w^2 + 8w + 1 = (w + 4)^2 - 15, for 5→w→w→1.
    width = math.isqrt(budget // bytes_per_parameter + 15) - 4
    params = lambda w: w * w + 8 * w + 1
    assert params(width) * bytes_per_parameter <= budget
    assert params(width + 1) * bytes_per_parameter > budget
    return dict(maximum_width=width, bytes_at_maximum=params(width) * bytes_per_parameter,
                first_failing_width=width + 1,
                bytes_at_first_failure=params(width + 1) * bytes_per_parameter)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="Also verify live /health and /model_info")
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path("model/config.json").read_text())
    assert cfg["team"] == 25 and cfg["hidden"] == [16, 16] and cfg["features"] == FEATURES
    assert cfg["val_mae"] == 5.31
    reference = json.loads(Path("artifacts/team_25/mission2/verified_sha256.json").read_text())
    for path in ("model/model.pt", "model/config.json"):
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == reference[path]
    net = BatteryNet(tuple(cfg["hidden"]))
    net.load_state_dict(torch.load("model/model.pt", weights_only=True))
    net.eval()
    selected = describe_model(net)
    assert (selected["total_params"], selected["total_bytes_f32"], selected["total_bytes_int8"],
            selected["total_macs"]) == (385, 1540, 385, 352)
    assert sum(p.numel() * p.element_size() for p in net.parameters()) == 1540
    weights = sum(layer.weight.numel() for layer in net.net if isinstance(layer, torch.nn.Linear))
    biases = sum(layer.bias.numel() for layer in net.net if isinstance(layer, torch.nn.Linear))
    assert (weights, biases) == (352, 33)
    dimensions = [5, 16, 16, 1]
    buffers = [dict(layer=i + 1, input_elements=a, output_elements=b,
                    simultaneous_elements=a + b, bytes_f32=4 * (a + b), bytes_int8=a + b)
               for i, (a, b) in enumerate(zip(dimensions, dimensions[1:]))]
    peak = max(row["simultaneous_elements"] for row in buffers)
    assert peak == 32
    memory = dict(model_info=selected, weight_elements=weights, bias_elements=biases,
                  float32_weights_bytes=4 * weights, float32_bias_bytes=4 * biases,
                  ideal_int8_all_parameters_bytes=weights + biases,
                  realistic_int8_weights_int32_biases_bytes=weights + 4 * biases,
                  activation_schedule=buffers, peak_activation_elements=peak,
                  peak_activation_bytes_f32=4 * peak, peak_activation_bytes_int8=peak,
                  all_four_tensors_retained_bytes_f32=4 * sum(dimensions),
                  normalisation_statistics_bytes_f32=4 * 2 * len(FEATURES),
                  native_f32_parameters_activations_and_statistics_bytes=1540 + 4 * peak + 40,
                  assumptions="Batch size one; two reusable 16-slot buffers. First input occupies five slots; a layer input and output coexist, then old input is released. ReLU and input normalisation are in place. No saved training activations or autograd. Excludes kernel scratch, scalar/vector accumulators, alignment padding, runtime/interpreter, stack, JSON/network, allocator caches and code. Int8 activation count is hypothetical; no int8 model was produced. Full quantised kernels need int32 biases/accumulators and quantisation metadata.")
    save("selected_memory.json", memory)

    table = []
    for width in (32, 128, 256, 512, 2048):
        info = describe_model(BatteryNet((width, width)))
        assert info["total_params"] == width * width + 8 * width + 1
        assert info["total_macs"] == width * width + 6 * width
        table.append(dict(width=width, params=info["total_params"],
                          bytes_f32=info["total_bytes_f32"], bytes_int8=info["total_bytes_int8"],
                          macs=info["total_macs"],
                          fits_sram_f32=info["fits_300kb_sram_f32"],
                          fits_sram_int8=info["fits_300kb_sram_int8"],
                          fits_psram_f32=info["total_bytes_f32"] <= PSRAM_BUDGET_BYTES,
                          fits_psram_int8=info["total_bytes_int8"] <= PSRAM_BUDGET_BYTES))
    with (ROOT / "width_table.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)
    boundaries = dict(sram_budget_bytes=SRAM_BUDGET_BYTES,
                      psram_budget_bytes=PSRAM_BUDGET_BYTES,
                      unit_convention="300 KiB SRAM and 8 MiB PSRAM; matches existing /model_info. Fits flags concern parameters only, not whole-program memory.",
                      float32=maximum_width(SRAM_BUDGET_BYTES, 4),
                      ideal_int8=maximum_width(SRAM_BUDGET_BYTES, 1),
                      decimal_300000_byte_alternative={
                          "float32": maximum_width(300000, 4),
                          "ideal_int8": maximum_width(300000, 1)})
    for field, bytes_per_parameter in (("float32", 4), ("ideal_int8", 1)):
        width = boundaries[field]["maximum_width"]
        flag = "fits_300kb_sram_f32" if bytes_per_parameter == 4 else "fits_300kb_sram_int8"
        assert describe_model(BatteryNet((width, width)))[flag]
        assert not describe_model(BatteryNet((width + 1, width + 1)))[flag]
        assert width == max(w for w in range(1, 1000)
                            if (w * w + 8 * w + 1) * bytes_per_parameter <= SRAM_BUDGET_BYTES)
    save("width_limits.json", boundaries)

    benchmarks = json.loads(Path("artifacts/team_25/mission3/benchmark_summaries.json").read_text())
    reference_rate = 256 * 240_000_000 / 432
    assumed_rate = 100_000_000
    estimate = dict(kind="estimate, not an on-device measurement", precision="float32",
                    kernel="Native dense layers built from ESP-DSP-style optimized float32 dot products",
                    source_proxy="dsps_dotprod_f32, length 256, ESP32S3 O2: 432 cycles",
                    assumed_cpu_hz=240_000_000, reference_proxy_macs_per_second=reference_rate,
                    assumed_macs_per_second=assumed_rate, macs=352,
                    arithmetic_only_estimate_us=352 / assumed_rate * 1_000_000,
                    arithmetic_only_estimate_ms=352 / assumed_rate * 1000,
                    caveat="100 MMAC/s is an explicit round-down assumption from the 142.22 MMAC/s long-vector reference, not a measured rate for these short layers. 3.52 us is an optimistic MAC-work estimate, not a full prediction guarantee. Length-5/16 tails, per-call setup, bias, ReLU, normalisation, memory access, wake-up, allocation, runtime and I/O add work; full board latency is unmeasured. One MAC is one weight multiply-accumulate, not one FLOP. No dual-core multiplier assumed.",
                    measured_codespace_public_inference_p50_ms=benchmarks["public_keepalive"]["percentiles_excluding_first"]["inference"]["p50"],
                    measured_codespace_local_inference_p50_ms=benchmarks["codespace_localhost"]["percentiles_excluding_first"]["inference"]["p50"],
                    measured_public_keepalive_rtt_p50_ms=benchmarks["public_keepalive"]["percentiles_excluding_first"]["round_trip"]["p50"])
    save("on_device_estimate.json", estimate)
    save("sources.json", SOURCES)
    save("calculations.json", dict(selected=memory, width_table=table, boundaries=boundaries,
                                    estimate=estimate, sources=SOURCES))
    if args.url:
        import requests
        endpoint = args.url.rstrip("/")
        health_response = requests.get(endpoint + "/health", timeout=30, allow_redirects=False)
        info_response = requests.get(endpoint + "/model_info", timeout=30, allow_redirects=False)
        assert health_response.status_code == info_response.status_code == 200
        assert info_response.json() == selected
        assert health_response.json() == {"status": "ok", "team": 25, "params": 385, "val_mae": 5.31}
        save("live_model_info.json", dict(url=endpoint, health=health_response.json(),
                                          model_info=info_response.json(), status=200))
    print("SELECTED:", json.dumps(selected))
    print("ACTIVATIONS: peak 32 elements = 128 float32 bytes or 32 hypothetical int8 bytes")
    print("WIDTH TABLE:", json.dumps(table))
    print("EXACT WIDTH LIMITS:", json.dumps(boundaries))
    print("ON-DEVICE ESTIMATE:", json.dumps(estimate))
    print("PASS: same network-counting helper, independent closed-form and exhaustive boundary checks, unchanged selected artifacts.")


if __name__ == "__main__":
    main()

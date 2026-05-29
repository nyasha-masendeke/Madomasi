"""Keras callback that samples CPU%, RAM%, and SoC temperature during training
and writes a per-epoch JSON next to history files for Excel graphing.

Samples on a 1 s background thread so brief spikes between epoch boundaries
are not missed, then aggregates per epoch (mean + max for cpu/temp, mean +
peak for ram). Exposes `output_dir` so train_head's _callback_output_dir
treats it as the canonical Training<N> directory — history_<stage>.json and
resources_<stage>.json land side by side.

Schema (lists aligned with history_<stage>.json epoch indices):
    {
      "stage": "head",
      "epoch": [0, 1, ...],
      "epoch_seconds":  [...],
      "cpu_percent_mean": [...], "cpu_percent_max": [...],
      "ram_percent_mean": [...], "ram_percent_peak": [...],
      "ram_used_gb_mean": [...], "ram_used_gb_peak": [...],
      "temp_c_mean":     [...], "temp_c_max":      [...],
      "samples_per_epoch": [...]
    }
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import psutil
import tensorflow as tf

from components.system_dashboard import _read_temp


class ResourceLogger(tf.keras.callbacks.Callback):
    def __init__(self, output_dir: str | Path, stage: str = "head", sample_interval: float = 1.0):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.stage = stage
        self.sample_interval = sample_interval

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._epoch_samples: list[dict] = []

        self._data = {
            "stage": stage,
            "epoch": [],
            "epoch_seconds": [],
            "cpu_percent_mean": [], "cpu_percent_max": [],
            "ram_percent_mean": [], "ram_percent_peak": [],
            "ram_used_gb_mean": [], "ram_used_gb_peak": [],
            "temp_c_mean": [], "temp_c_max": [],
            "samples_per_epoch": [],
        }
        self._t0: float = 0.0

    def _sample_once(self) -> dict:
        vm = psutil.virtual_memory()
        return {
            "cpu": psutil.cpu_percent(interval=None),
            "ram_pct": vm.percent,
            "ram_gb": vm.used / 1e9,
            "temp": _read_temp(),
        }

    def _sampler_loop(self) -> None:
        psutil.cpu_percent(interval=None)  # prime delta counter
        while not self._stop.is_set():
            s = self._sample_once()
            with self._lock:
                self._epoch_samples.append(s)
            self._stop.wait(self.sample_interval)

    def on_train_begin(self, logs=None):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._stop.clear()
        self._thread = threading.Thread(target=self._sampler_loop, daemon=True)
        self._thread.start()

    def on_epoch_begin(self, epoch, logs=None):
        with self._lock:
            self._epoch_samples.clear()
        self._t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self._t0
        with self._lock:
            samples = list(self._epoch_samples)
            self._epoch_samples.clear()

        def _stats(key):
            xs = [s[key] for s in samples if s.get(key) is not None]
            if not xs:
                return (float("nan"), float("nan"))
            return (sum(xs) / len(xs), max(xs))

        cpu_mean, cpu_max = _stats("cpu")
        ram_mean, ram_max = _stats("ram_pct")
        gb_mean,  gb_max  = _stats("ram_gb")
        t_mean,   t_max   = _stats("temp")

        d = self._data
        d["epoch"].append(int(epoch))
        d["epoch_seconds"].append(round(elapsed, 3))
        d["cpu_percent_mean"].append(round(cpu_mean, 2))
        d["cpu_percent_max"].append(round(cpu_max, 2))
        d["ram_percent_mean"].append(round(ram_mean, 2))
        d["ram_percent_peak"].append(round(ram_max, 2))
        d["ram_used_gb_mean"].append(round(gb_mean, 3))
        d["ram_used_gb_peak"].append(round(gb_max, 3))
        d["temp_c_mean"].append(round(t_mean, 2) if t_mean == t_mean else None)
        d["temp_c_max"].append(round(t_max, 2) if t_max == t_max else None)
        d["samples_per_epoch"].append(len(samples))

        self._flush()

    def on_train_end(self, logs=None):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._flush()

    def _flush(self) -> None:
        out = self.output_dir / f"resources_{self.stage}.json"
        out.write_text(json.dumps(self._data))

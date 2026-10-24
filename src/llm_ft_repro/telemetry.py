"""NVML observes the entire physical device, not only PyTorch's allocator."""
from __future__ import annotations

import threading
import time


class NvmlSampler:
    def __init__(self, device, interval):
        self.device, self.interval = device, interval
        self.samples, self.errors = [], []
        self.phase = "warmup"
        self.stop_event = threading.Event()

    def start(self):
        import pynvml as nv
        self.nv = nv
        nv.nvmlInit()
        self.handle = (nv.nvmlDeviceGetHandleByUUID(self.device) if self.device.startswith("GPU-")
                       else nv.nvmlDeviceGetHandleByIndex(int(self.device)))
        self.name = nv.nvmlDeviceGetName(self.handle)
        self.uuid = nv.nvmlDeviceGetUUID(self.handle)
        self.driver = nv.nvmlSystemGetDriverVersion()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def _loop(self):
        nv = self.nv
        fields = {
            "used_bytes": lambda: nv.nvmlDeviceGetMemoryInfo(self.handle).used,
            "sm_clock_mhz": lambda: nv.nvmlDeviceGetClockInfo(self.handle, nv.NVML_CLOCK_SM),
            "power_watts": lambda: nv.nvmlDeviceGetPowerUsage(self.handle) / 1000,
            "temperature_c": lambda: nv.nvmlDeviceGetTemperature(self.handle, nv.NVML_TEMPERATURE_GPU),
            "throttle_reasons": lambda: nv.nvmlDeviceGetCurrentClocksThrottleReasons(self.handle),
        }
        while not self.stop_event.is_set():
            row = {"monotonic_s": time.monotonic(), "phase": self.phase}
            for key, call in fields.items():
                try:
                    row[key] = call()
                except Exception as exc:
                    row[key] = None
                    error = f"{key}: {exc}"
                    if error not in self.errors:
                        self.errors.append(error)
            self.samples.append(row)
            self.stop_event.wait(self.interval)

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=5)
        self.nv.nvmlShutdown()
        return {"device": str(self.name), "uuid": str(self.uuid), "driver": str(self.driver),
                "samples": self.samples, "errors": self.errors,
                "paging_observed": None,
                "paging_note": "NVML does not measure unified-memory migration; use an external profiler."}

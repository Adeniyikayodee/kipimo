import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from kipimo.perception.model import INT8


class OnnxPerception:
    """int8 UNet on CPU. Latency is measured on every call, never assumed."""

    def __init__(self, path: Path = INT8, threads: int = 1):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self.size_mb = path.stat().st_size / 1e6
        self.latency_ms: list[float] = []

    def __call__(self, img: np.ndarray) -> np.ndarray:
        x = img.transpose(2, 0, 1)[None].astype(np.float32)
        t = time.perf_counter()
        out = self.sess.run(None, {"image": x})[0][0]
        self.latency_ms.append((time.perf_counter() - t) * 1000)
        return out

    def stats(self) -> dict:
        lat = np.array(self.latency_ms[-500:] or [0.0])
        return {
            "model_mb": round(self.size_mb, 3),
            "p50_ms": round(float(np.percentile(lat, 50)), 2),
            "p95_ms": round(float(np.percentile(lat, 95)), 2),
            "n": len(self.latency_ms),
        }

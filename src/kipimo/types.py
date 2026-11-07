from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

Crop = Literal["maize", "rice", "sorghum"]


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class Parcel:
    id: str
    cells: npt.NDArray[np.int32]
    holder: str
    sum_insured: float
    crop: Crop
    planted_day: int


@dataclass(frozen=True)
class Observation:
    t: float
    pose: Pose
    cell_ids: npt.NDArray[np.int32]
    water_frac: npt.NDArray[np.float32]
    crop_loss: npt.NDArray[np.float32]
    quality: float


@dataclass(frozen=True)
class Objective:
    parcel_ids: list[str]
    target_confidence: float
    max_distance_m: float
    reason: str


@dataclass(frozen=True)
class ParcelBelief:
    parcel_id: str
    loss_mean: float
    loss_ci95: tuple[float, float]
    confidence: float
    cells_observed: int
    cells_total: int

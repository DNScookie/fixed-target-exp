from dataclasses import dataclass, field
from typing import List

import numpy as np

from generator.detectors import Detector
from generator.models import MagneticField


@dataclass
class ExperimentSetup:
    bfield: MagneticField
    detectors: List[Detector] = field(default_factory=list)
    seed: int = 42

    @classmethod
    def from_dict(cls, obj: dict):
        return cls(
            bfield=MagneticField.from_dict(obj["bfield"]),
            detectors=[Detector.from_dict(det) for det in obj["detectors"]],
            seed=obj["seed"],
        )

    def reset(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        for i, det in enumerate(self.detectors):
            det.reset(seed + i)

    def add_detector(self, detector: Detector):
        detector.layer = len(self.detectors)
        detector.seed = self.seed + detector.layer
        self.detectors.append(detector)

    def to_dict(self):
        return {
            "bfield": self.bfield.to_dict(),
            "detectors": [det.to_dict() for det in self.detectors],
            "seed": self.seed,
        }

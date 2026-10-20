import abc
from dataclasses import dataclass, field
from typing import ClassVar, List, Tuple

import numpy as np

from generator.models import Particle, Hit

DETECTOR_REGISTRY: dict[str, type["Detector"]] = {}


def register_detector(cls):
    DETECTOR_REGISTRY[cls.type_id] = cls
    return cls


@dataclass
class Detector(abc.ABC):
    type_id: ClassVar[
        str
    ]  # формат типа 'typename_v3' (версии пишутся для обратной совместимости, если будем писать новые реализации)
    name: str
    z: float
    size_x: float
    size_y: float
    center_x: float = 0
    center_y: float = 0
    layer: int = 0
    seed: int = 42

    # внутренние буферы для текущего события
    raw_hits_buffer: List[Tuple[float, float, int, int]] = field(
        default_factory=list
    )  # (x, y, pid, hit_number)
    true_hits_buffer: List[Tuple[float, float, int, int]] = field(
        default_factory=list
    )  # (x, y, pid, hit_number)
    fake_hits_buffer: List[Tuple[float, float, str]] = field(
        default_factory=list
    )  # (x, y, reason)

    @classmethod
    def from_dict(cls, obj: dict):
        type_id = obj["type_id"]
        if type_id not in DETECTOR_REGISTRY:
            raise ValueError(f"Unknown detector type_id: {type_id}")

        det_cls = DETECTOR_REGISTRY[type_id]
        return det_cls._from_dict_impl(obj)

    @classmethod
    @abc.abstractmethod
    def _from_dict_impl(cls, obj: dict):
        """Создание конкретного детектора"""
        pass

    def reset(self, seed: int = 42):
        self.raw_hits_buffer.clear()
        self.true_hits_buffer.clear()
        self.fake_hits_buffer.clear()
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    # геометрия
    def area(self):
        return self.size_x * self.size_y

    def register(self, hit, particle: Particle, hit_number):
        x = hit[0]
        y = hit[1]
        if not (
            (-self.size_x / 2 - 1e-6 <= x - self.center_x <= self.size_x / 2 + 1e-6)
            and (-self.size_y / 2 - 1e-6 <= y - self.center_y <= self.size_y / 2 + 1e-6)
        ):
            return
        self.raw_hits_buffer.append((hit[0], hit[1], particle.id, hit_number))
        # print(f'Particle {particle.id} hits detector {self.layer}')

    @abc.abstractmethod
    def register_hits(self):
        """Зарегистрировать истинные хиты и добавить ложные. Зависит от типа детектора."""
        pass

    def finalize_hits(self, event_id: int, hit_id_start: int):
        hits = []
        hit_id = hit_id_start

        if not hasattr(self, "rng"):
            self.rng = np.random.default_rng(self.seed)

        self.register_hits()

        # истинные
        for x, y, pid, hit_number in self.true_hits_buffer:
            hits.append(
                Hit(
                    event_id=event_id,
                    hit_id=hit_id,
                    detector=self.name,
                    layer=self.layer,
                    x=x,
                    y=y,
                    z=self.z,
                    hit_number=hit_number,
                    is_fake=False,
                    true_particle_id=pid,
                )
            )
            hit_id += 1

        # шумовые
        for x, y, reason in self.fake_hits_buffer:
            hits.append(
                Hit(
                    event_id=event_id,
                    hit_id=hit_id,
                    detector=self.name,
                    layer=self.layer,
                    x=x,
                    y=y,
                    z=self.z,
                    hit_number=-1,
                    is_fake=True,
                    true_particle_id=None,
                    reason=reason,
                )
            )
            hit_id += 1

        # очистить состояние
        self.raw_hits_buffer.clear()
        self.true_hits_buffer.clear()
        self.fake_hits_buffer.clear()

        return hits

    def to_dict(self):
        return {
            "type_id": self.type_id,
            "name": self.name,
            "z": self.z,
            "size_x": self.size_x,
            "size_y": self.size_y,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "layer": self.layer,
            "seed": self.seed,
        }


@dataclass
class NaiveDetector(Detector):
    type_id: ClassVar[str] = "naive_v1"
    efficiency: float = 0.99
    noise_rate: float = 0.4

    @classmethod
    def _from_dict_impl(cls, obj: dict):
        return cls(
            name=obj["name"],
            z=obj["z"],
            size_x=obj["size_x"],
            size_y=obj["size_y"],
            center_x=obj["center_x"],
            center_y=obj["center_y"],
            layer=obj["layer"],
            seed=obj["seed"],
            efficiency=obj["efficiency"],
            noise_rate=obj["noise_rate"],
        )

    def register_hits(self):
        for x, y, pid, hit_number in self.raw_hits_buffer:
            if self.rng.random() < self.efficiency:
                self.true_hits_buffer.append((x, y, pid, hit_number))

        n_noise = self.rng.poisson(self.noise_rate * len(self.raw_hits_buffer))
        x_noise = self.rng.uniform(self.center_x - self.size_x / 2, self.center_x + self.size_x / 2, n_noise)
        y_noise = self.rng.uniform(self.center_y - self.size_y / 2, self.center_y + self.size_y / 2, n_noise)
        for i in range(n_noise):
            self.fake_hits_buffer.append((x_noise[i], y_noise[i], "noise"))

    def to_dict(self):
        return super().to_dict() | {
            "efficiency": self.efficiency,
            "noise_rate": self.noise_rate,
        }


register_detector(NaiveDetector)


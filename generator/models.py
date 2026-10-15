import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Vertex:
    x: float
    y: float
    z: float

    @classmethod
    def from_dict(cls, obj: dict):
        return cls(obj["x"], obj["y"], obj["z"])

    def to_dict(self):
        return {"x": self.x, "y": self.y, "z": self.z}

    def to_nparray(self):
        return np.array([self.x, self.y, self.z])


@dataclass
class Particle:
    id: int
    name: str
    charge: int
    mass: float
    momentum: np.ndarray
    energy: float
    theta: float
    phi: float
    color: str = "blue"

    @classmethod
    def from_dict(cls, obj: dict):
        return cls(
            obj["id"],
            obj["name"],
            obj["charge"],
            obj["mass"],
            np.array(obj["momentum"]),
            obj["energy"],
            obj["theta"],
            obj["phi"],
            obj["color"],
        )

    def __str__(self):
        return (
            f"{self.id:2d}: {self.name:3s} q={self.charge:+d} "
            + f"p={np.linalg.norm(self.momentum):.3f} GeV/c θ={self.theta:.1f}° φ={self.phi:.1f}°"
        )

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "charge": self.charge,
            "mass": self.mass,
            "momentum": self.momentum.tolist(),
            "energy": self.energy,
            "theta": self.theta,
            "phi": self.phi,
            "color": self.color,
        }


@dataclass
class Event:
    vertex: Vertex
    particles: list

    @classmethod
    def from_dict(cls, obj: dict):
        return cls(
            Vertex.from_dict(obj["vertex"]),
            [Particle.from_dict(p) for p in obj["particles"]],
        )

    def to_dict(self):
        return {
            "vertex": self.vertex.to_dict(),
            "particles": [p.to_dict() for p in self.particles],
        }


class ParticleGenerator:
    def __init__(self, config, seed=42):
        self.cfg = config
        self._normalize_particle_probs()
        self.rng = np.random.default_rng(seed)

    def _normalize_particle_probs(self):
        types = self.cfg["particle_types"]
        total = sum(types[name]["rate"] for name in types)
        self.particle_probs = {name: types[name]["rate"] / total for name in types}

    def _choose_particle_type(self):
        names, probs = zip(*self.particle_probs.items())
        return self.rng.choice(names, p=probs)

    def generate_vertex(self):
        sx, sy, sz = self.cfg["vertex_sigma"]
        return Vertex(
            self.rng.normal(0, sx), self.rng.normal(0, sy), self.rng.normal(0, sz)
        )

    def _sample_theta(self):
        eta0 = self.cfg.get("eta0", 1.5)
        sigma = self.cfg.get("eta_sigma", 0.8)

        eta = self.rng.normal(eta0, sigma)

        # клип (важно!)
        eta = np.clip(eta,
                      self.cfg.get("eta_min", -0.5),
                      self.cfg.get("eta_max", 3.0))

        theta = 2 * np.arctan(np.exp(-eta))
        return theta

    def generate_particle(self, pid):
        name = self._choose_particle_type()
        info = self.cfg["particle_types"][name]

        # Импульс (в ГэВ/с)
        p = self.rng.uniform(*self.cfg["momentum_range"])

        # Анизотропное распределение по θ
        theta = self._sample_theta()

        # Равномерное распределение по φ
        phi = np.radians(self.rng.uniform(*self.cfg["phi_range"]))

        # Проекции импульса
        px = p * math.sin(theta) * math.cos(phi)
        py = p * math.sin(theta) * math.sin(phi)
        pz = p * math.cos(theta)

        # Энергия частицы
        E = math.sqrt(p**2 + info["mass"] ** 2)

        return Particle(
            id=pid,
            name=name,
            charge=info["charge"],
            mass=info["mass"],
            momentum=np.array([px, py, pz]),
            energy=E,
            theta=math.degrees(theta),
            phi=math.degrees(phi),
            color=info["color"],
        )

    def generate_event(self):
        vertex = self.generate_vertex()
        particles = [self.generate_particle(i) for i in range(self.cfg["n_particles"])]
        return Event(vertex, particles)


@dataclass
class Hit:
    event_id: int
    hit_id: int
    detector: str
    layer: int
    x: float
    y: float
    z: float
    hit_number: int = -1  # номер хита для конкретной частицы
    is_fake: bool = False
    true_particle_id: Optional[int] = None
    reason: Optional[str] = None

    @classmethod
    def from_dict(cls, obj: dict):
        return cls(
            obj["event_id"],
            obj["hit_id"],
            obj["detector"],
            obj["layer"],
            obj["x"],
            obj["y"],
            obj["z"],
            obj["hit_number"],
            obj["is_fake"],
            obj["true_particle_id"],
            obj["reason"],
        )

    def __str__(self):
        return (
            f"hit {self.hit_id:03d} det={self.detector:8s} "
            + f"layer={self.layer:2d} z={self.z:0.2f} m x={self.x:+0.4f} m "
            + f"y={self.y:+0.4f} m fake={self.is_fake} "
            + f"pid={self.true_particle_id}"
        )

    def to_dict(self):
        return {
            "event_id": self.event_id,
            "hit_id": self.hit_id,
            "detector": self.detector,
            "layer": self.layer,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "hit_number": self.hit_number,
            "is_fake": self.is_fake,
            "true_particle_id": self.true_particle_id,
            "reason": self.reason,
        }


# --- Magnetic field (uniform inside region) ---
@dataclass
class MagneticField:
    B_vec: np.ndarray  # 3-vector in Tesla
    z_start: float
    z_end: float

    @classmethod
    def from_dict(cls, obj: dict):
        return cls(np.array(obj["B_vec"]), obj["z_start"], obj["z_end"])

    def B_at(self, z: float):
        if self.z_start <= z <= self.z_end:
            return self.B_vec.copy()
        else:
            return np.array([0.0, 0.0, 0.0])

    def to_dict(self):
        return {
            "B_vec": self.B_vec.tolist(),
            "z_start": self.z_start,
            "z_end": self.z_end,
        }

import numpy as np
from tqdm import tqdm

from generator.detectors import NaiveDetector
from generator.experiment import ExperimentSetup
from generator.models import MagneticField, ParticleGenerator
from generator.simulation import simulate_hits


def example_generator_config():
    return {
        "n_particles": 50,
        "momentum_range": [0.1, 2.0],  # ГэВ/с
        "eta0": 3.0,
        "eta_sigma": 0.6,
        "eta_min": 1.5,
        "eta_max": 4.0,
        "phi_range": [0, 360],
        "vertex_sigma": [0.01, 0.01, 0],  # м
        "particle_types": {
            "pi+": {"mass": 0.13957, "charge": +1, "rate": 0.4, "color": "red"},
            "pi-": {"mass": 0.13957, "charge": -1, "rate": 0.4, "color": "blue"},
            "K+": {"mass": 0.49368, "charge": +1, "rate": 0.05, "color": "green"},
            "K-": {"mass": 0.49368, "charge": -1, "rate": 0.05, "color": "orange"},
            "p": {"mass": 0.93827, "charge": +1, "rate": 0.07, "color": "black"},
            "e-": {"mass": 0.000511, "charge": -1, "rate": 0.03, "color": "yellow"},
        },
    }


def build_naive_detectors(n_layers=7, seed=42):
    exp_setup = ExperimentSetup(
        bfield=MagneticField(B_vec=np.array([0.0, 0.0, 0.8]), z_start=0.70, z_end=1.60)
    )
    for i in range(n_layers):
        exp_setup.add_detector(
            NaiveDetector(
                name=f"det{i}",
                z=(i + 1) * 1.60 / n_layers,
                size_x=0.5,
                size_y=0.5,
                seed=seed + i,
            )
        )
    return exp_setup


def generate_events_and_hits(
    n_events: int,
    generator_config: dict,
    experiment_setup: ExperimentSetup,
    seed=42,
):
    events = []
    hit_lists = []

    particle_generator = ParticleGenerator(config=generator_config, seed=seed)
    experiment_setup.reset(seed=seed)

    for i in tqdm(range(n_events), desc="Generating events"):
        event = particle_generator.generate_event()
        hits = simulate_hits(event, experiment_setup=experiment_setup, event_id=i)
        np.random.default_rng(seed + i).shuffle(hits)

        events.append(event)
        hit_lists.append(hits)

    return events, hit_lists


def generate_to_dict(
    n_events: int,
    generator_config: dict,
    experiment_setup: ExperimentSetup,
    seed: int = 42,
):
    events, hit_lists = generate_events_and_hits(
        n_events=n_events,
        generator_config=generator_config,
        experiment_setup=experiment_setup,
        seed=seed,
    )
    return {
        "n_events": n_events,
        "generator_config": generator_config,
        "experiment_setup": experiment_setup.to_dict(),
        "seed": seed,
        "events": [e.to_dict() for e in events],
        "hits": [[h.to_dict() for h in hits] for hits in hit_lists],
    }

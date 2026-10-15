import math
from typing import List

import numpy as np

from generator.experiment import ExperimentSetup
from generator.models import Event, Hit

# constants
C_LIGHT = 299792458.0  # m/s
CONV_0_3 = 0.3  # factor relating p[GeV/c], B[T], R[m]: p_T = 0.3 * q * B * R


def _rodrigues_rotate(vec: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotate vector `vec` around unit `axis` by `angle` radians (Rodrigues' formula)."""
    axis = axis / np.linalg.norm(axis)
    return (
        vec * math.cos(angle)
        + np.cross(axis, vec) * math.sin(angle)
        + axis * np.dot(axis, vec) * (1.0 - math.cos(angle))
    )


def _interpolate_crossing(
    prev_pos: np.ndarray, pos: np.ndarray, z_plane: float
) -> np.ndarray:
    """Linear interpolation between prev_pos and pos to find point at z = z_plane.
    Assumes prev_pos[2] != pos[2] and z_plane between them."""
    z0 = prev_pos[2]
    z1 = pos[2]
    t = (z_plane - z0) / (z1 - z0)
    return prev_pos + t * (pos - prev_pos)


def simulate_hits(
    event: Event,
    experiment_setup: ExperimentSetup,
    event_id: int = 0,
    ds: float = 1e-3,
    max_steps: int = 200000,
) -> List[Hit]:
    """
    Simulate hits for a single event.
    - event: Event (vertex + particles)
    - detectors: list of Detector objects (with z positions)
    - bfield: MagneticField object with B_at(z)
    - event_id: integer event id
    - ds: path-step in meters (default 1 mm)
    - max_steps: safety cap
    Returns: list of Hit objects (true + fake)
    """
    hits: List[Hit] = []
    hit_counter = 0

    bfield = experiment_setup.bfield
    detectors = experiment_setup.detectors

    # Precompute sorted detector z-values and map z->detector list (support multiple per-z)
    dets_by_z = {}
    for det in detectors:
        dets_by_z.setdefault(det.z, []).append(det)
    z_planes = sorted(dets_by_z.keys())

    z_min = min(z_planes) - 0.1
    z_max = max(z_planes) + 0.1

    for particle in event.particles:
        # initial state
        pos = np.array(
            [event.vertex.x, event.vertex.y, event.vertex.z], dtype=float
        )  # meters
        p = np.array(particle.momentum, dtype=float)  # GeV/c
        p_mag = np.linalg.norm(p)
        if p_mag == 0:
            continue
        p_unit = p / p_mag

        q = float(particle.charge)
        step_count = 0
        hit_count = 0

        prev_pos = pos.copy()

        # We will step forward until we exit z range or exceed steps
        while step_count < max_steps:
            step_count += 1

            # Determine local B (at current z)
            B_vec = bfield.B_at(pos[2])
            B_mag = np.linalg.norm(B_vec)

            # compute p_par and p_perp wrt B
            if B_mag > 0:
                B_unit = B_vec / B_mag
                p_par = np.dot(p, B_unit)
                p_perp_sq = max(0.0, p_mag**2 - p_par**2)
                p_perp = math.sqrt(p_perp_sq) if p_perp_sq > 0 else 0.0
            else:
                B_unit = None
                p_perp = 0.0

            # move along direction by ds (approx path length ds)
            # but in curved region we will rotate momentum first by angle dtheta then advance
            if B_mag > 0 and p_perp > 1e-12:
                # curvature kappa = q * 0.3 * B / p_perp (1/m)
                kappa = q * CONV_0_3 * B_mag / p_perp  # 1/m
                # angle to rotate for path length ds:
                dtheta = kappa * ds
                # rotate momentum vector around B_unit by dtheta
                p = _rodrigues_rotate(p, B_unit, dtheta)
                p_mag = np.linalg.norm(p)
                if p_mag == 0:
                    break
                p_unit = p / p_mag
            else:
                # no B or effectively parallel => straight propagation, p unchanged
                pass

            # advance position by ds along p_unit
            pos = pos + p_unit * ds

            # If we moved outside z bounds and are going further away, stop
            # Note: we continue a bit beyond z_max to ensure last detector crossing
            if (pos[2] < z_min and p_unit[2] <= 0) or (
                pos[2] > z_max and p_unit[2] >= 0
            ):
                break

            # For each detector plane that lies between prev_pos.z and pos.z, register an intersection
            for z_plane in z_planes:
                z0 = prev_pos[2]
                z1 = pos[2]
                # check crossing (strictly between to avoid double counting if step lands exactly on plane)
                if (z0 <= z_plane <= z1) or (z1 <= z_plane <= z0):
                    # ignore trivial same-z
                    if abs(z1 - z0) < 1e-12:
                        continue
                    # interpolate hit position at plane z_plane
                    hit_pos = _interpolate_crossing(prev_pos, pos, z_plane)
                    # for each detector at this z (support stack)
                    for det in dets_by_z[z_plane]:
                        det.register(hit_pos, particle, hit_count)
                        hit_count += 1

            prev_pos = pos.copy()

        # end propagation for this particle
        if step_count >= max_steps:
            print(f"warning: max_steps reached propagating {particle}")

    for det in detectors:
        hits.extend(det.finalize_hits(event_id, hit_counter))
        hit_counter += len(hits)

    # final: return hits (unsorted). Up to caller to sort/filter as needed.
    return hits

"""Exact forward propagation for a uniform longitudinal field.

Separate from the general stepper. No material interactions or detector smearing.
"""
import numpy as np

def advance_z(pos, momentum, charge, z_next, bz):
    """Exact helix segment in B=(0,0,bz), lengths m and momentum GeV/c.

    Uses dp/dt=q v×B. The sign of rotation is negative for positive charge.
    pz is positive for the forward model used here.
    """
    pos = np.asarray(pos, dtype=float).copy()
    p = np.asarray(momentum, dtype=float).copy()
    dz = z_next - pos[2]
    assert p[2] > 0 and dz >= -1e-12
    k = -0.299792458 * charge * bz / p[2]
    if abs(k) < 1e-12:
        pos[:2] += p[:2] / p[2] * dz
    else:
        a = k * dz
        si, co = np.sin(a), np.cos(a)
        px, py = p[:2].copy()
        pos[0] += (px * si + py * (co - 1)) / (p[2] * k)
        pos[1] += (px * (1 - co) + py * si) / (p[2] * k)
        p[0], p[1] = px * co - py * si, px * si + py * co
    pos[2] = z_next
    return pos, p



def simulate_hits_analytic(event, experiment_setup, event_id=0):
    """Register crossings using exact Bz helices and straight field-free segments.

    Supports forward particles (pz>0), planar detectors and one field interval.
    Other field directions are rejected rather than silently approximated.
    Detector response and buffers are shared with the stepwise simulator.
    """
    field = experiment_setup.bfield
    if not np.allclose(field.B_vec[:2], 0):
        raise ValueError("Analytic propagation requires B parallel to z")
    if field.z_end < field.z_start:
        raise ValueError("Invalid field interval")
    if any(d.raw_hits_buffer or d.true_hits_buffer or d.fake_hits_buffer
           for d in experiment_setup.detectors):
        raise ValueError("Detector buffers must be empty at event start")
    detectors = sorted(experiment_setup.detectors, key=lambda d: d.z)
    for particle in event.particles:
        pos = event.vertex.to_nparray().astype(float)
        momentum = np.asarray(particle.momentum, dtype=float).copy()
        if momentum[2] <= 0:
            raise ValueError("Analytic propagation requires pz>0")
        for number, detector in enumerate(detectors):
            if detector.z < pos[2]:
                continue
            stops = sorted({float(detector.z), *[float(b) for b in
                (field.z_start, field.z_end) if pos[2] < b < detector.z]})
            for stop in stops:
                middle = (pos[2] + stop) / 2
                bz = field.B_vec[2] if field.z_start <= middle <= field.z_end else 0.
                pos, momentum = advance_z(pos, momentum, particle.charge, stop, bz)
            detector.register(pos, particle, number)
    hits = []
    for detector in experiment_setup.detectors:
        hits.extend(detector.finalize_hits(event_id, len(hits)))
    return hits

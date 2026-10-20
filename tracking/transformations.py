from collections import defaultdict

import numpy as np

from generator.models import Hit

from collections import defaultdict
from generator.models import Hit


def hits_to_tracks(hits: list[Hit]) -> list[list[int]]:
    tracks = defaultdict(list)

    for i, hit in enumerate(hits):
        if not hit.is_fake and hit.true_particle_id is not None:
            tracks[hit.true_particle_id].append(i)

    return list(tracks.values())


def to_points_and_labels(
    hits: list[Hit], seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    Преобразует список Hit в:
      - массив точек shape (N, 3)
      - массив меток PID shape (N,)
    с согласованной случайной перестановкой.
    """
    rng = np.random.default_rng(seed)

    points = np.array([(h.x, h.y, h.z) for h in hits], dtype=np.float32)

    labels = [h.true_particle_id if not h.is_fake else -1 for h in hits]
    labels = np.array(labels, dtype=np.int32)

    perm = rng.permutation(len(hits))
    return points[perm], labels[perm]


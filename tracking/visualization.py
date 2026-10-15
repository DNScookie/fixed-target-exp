import matplotlib.pyplot as plt
import numpy as np

from generator.models import Hit


def visualize_tracks_2d(
    hits: list[Hit],
    tracks: list[list[int]],
    normal: np.ndarray = np.array([0.0, 0.0, 1.0]),
) -> None:
    """Plot tracks projected onto a plane with the given normal.

    Each track contains indices into `hits`.
    """
    n = np.asarray(normal, dtype=float)

    if n.shape != (3,):
        raise ValueError("Normal must be a 3D vector")

    norm = np.linalg.norm(n)
    if norm == 0:
        raise ValueError("Normal vector must be non-zero")
    n /= norm

    # Construct an orthonormal basis for the projection plane.
    reference = np.eye(3)[np.argmin(np.abs(n))]
    u = np.cross(n, reference)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)

    if not hits:
        print("No hits to visualize")
        return

    points = np.array([(h.x, h.y, h.z) for h in hits], dtype=float)
    origin = points.mean(axis=0)

    projected = np.column_stack((
        (points - origin) @ u,
        (points - origin) @ v,
    ))

    fig, ax = plt.subplots(figsize=(8, 8))

    for track_idx, indices in enumerate(tracks):
        valid_indices = [
            i for i in indices
            if isinstance(i, (int, np.integer)) and 0 <= i < len(hits)
        ]

        if not valid_indices:
            continue

        track_points = projected[valid_indices]

        ax.plot(
            track_points[:, 0],
            track_points[:, 1],
            marker="o",
            markersize=4,
            linewidth=1,
            alpha=0.7,
            label=f"Track {track_idx}",
        )

    ax.set_title(f"2D Track Projection, normal={n}")
    ax.set_xlabel("u [m]")
    ax.set_ylabel("v [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True)

    if tracks:
        ax.legend()

    plt.show()

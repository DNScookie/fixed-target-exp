from typing import List

import matplotlib.pyplot as plt
import numpy as np

from generator.models import Event


def visualize_3d(event: Event, hits: List, draw_tracks=True):
    # Sort hits
    hit_list = hits

    # True hits
    x_true = [h.x for h in hit_list if not h.is_fake]
    y_true = [h.y for h in hit_list if not h.is_fake]
    z_true = [h.z for h in hit_list if not h.is_fake]

    # Fake hits
    x_fake = [h.x for h in hit_list if h.is_fake]
    y_fake = [h.y for h in hit_list if h.is_fake]
    z_fake = [h.z for h in hit_list if h.is_fake]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(z_true, x_true, y_true, label="true hits", s=20, marker=".", c="blue")
    ax.scatter(z_fake, x_fake, y_fake, label="fake hits", s=20, marker=".", c="gray")

    # построим линии треков
    # Group true hits by particle ID
    particle_tracks_hits = {}
    for h in hit_list:
        if not h.is_fake and h.true_particle_id is not None:
            if h.true_particle_id not in particle_tracks_hits:
                particle_tracks_hits[h.true_particle_id] = []
            particle_tracks_hits[h.true_particle_id].append(h)

    # Plot tracks by connecting the hits for each particle
    for pid, hits_for_particle in particle_tracks_hits.items():
        # Sort hits by z-coordinate to draw lines in correct order
        hits_for_particle_sorted = sorted(hits_for_particle, key=lambda h: h.hit_number)

        track_z = []
        track_x = []
        track_y = []

        # Add the event vertex as the starting point of the track
        # All particles originate from the same event vertex
        track_z.append(event.vertex.z)
        track_x.append(event.vertex.x)
        track_y.append(event.vertex.y)

        # Add all sorted hits
        for h in hits_for_particle_sorted:
            track_z.append(h.z)
            track_x.append(h.x)
            track_y.append(h.y)

        # Plot the track
        ax.plot(
            track_z,
            track_x,
            track_y,
            marker="o",
            color=event.particles[pid].color,
            linestyle="-",
            linewidth=1 if draw_tracks else 0,
            markersize=0,
            alpha=0.4,
        )

    ax.set_xlabel("z [m]")
    ax.set_ylabel("x [m]")
    ax.set_zlabel("y [m]")

    # ax.set_title("Hit map (3D)")

    ax.legend()
    plt.show()


def visualize_2d(event: Event, hits: List, normal: np.ndarray = [0, 0, 1], draw_tracks=True, draw_true_hits=True):
    """
    2D projection of graph (nodes and edges) onto a plane with given normal.
    """
    n = np.asarray(normal, dtype=float)
    n_norm = np.linalg.norm(n)
    if n_norm == 0:
        raise ValueError("Normal vector must be non-zero")
    n = n / n_norm

    if abs(n[0]) < 0.9:
        a = np.array([1.0, 0.0, 0.0])
    else:
        a = np.array([0.0, 1.0, 0.0])

    u = np.cross(n, a)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)

    node_coords = np.array([(h.x, h.y, h.z) for h in hits])
    r0 = event.vertex.to_nparray()

    def project(point):
        r = np.asarray(point) - r0
        return np.dot(u, r), np.dot(v, r)

    plt.figure(figsize=(8, 8))

    # --- Project nodes ---
    node_pids = np.array([h.true_particle_id if not h.is_fake else -1 for h in hits])
    x2d_true_nodes, y2d_true_nodes = [], []
    x2d_fake_nodes, y2d_fake_nodes = [], []

    for i, h in enumerate(hits):
        x2d, y2d = project(node_coords[i])
        if node_pids[i] != -1:
            x2d_true_nodes.append(x2d)
            y2d_true_nodes.append(y2d)
        else:
            x2d_fake_nodes.append(x2d)
            y2d_fake_nodes.append(y2d)

    if draw_true_hits:
        plt.scatter(x2d_true_nodes, y2d_true_nodes, s=20, marker='o', c='blue', label='True Hits')
        plt.scatter(x2d_fake_nodes, y2d_fake_nodes, s=20, marker='.', c='gray', label='Fake Hits')
    else:
        plt.scatter(x2d_true_nodes + x2d_fake_nodes,
                    y2d_true_nodes + y2d_fake_nodes,
                    s=20, marker='.', c='gray', label='Hits')

    # --- tracks ---
    particle_tracks_hits = {}
    for h in hits:
        if not h.is_fake and h.true_particle_id is not None:
            particle_tracks_hits.setdefault(h.true_particle_id, []).append(h)

    for pid, hits_for_particle in particle_tracks_hits.items():
        hits_sorted = sorted(hits_for_particle, key=lambda h: h.hit_number)

        xs, ys = [], []

        # vertex
        vx, vy = project(r0)
        xs.append(vx)
        ys.append(vy)

        for h in hits_sorted:
            x2d, y2d = project([h.x, h.y, h.z])
            xs.append(x2d)
            ys.append(y2d)

        plt.plot(
            xs,
            ys,
            marker="o",
            markersize=0,
            linewidth=1 if draw_tracks else 0,
            alpha=0.5,
            color=event.particles[pid].color,
        )

    plt.xlabel("u [m]")
    plt.ylabel("v [m]")
    plt.title(f"2D projection {normal}")
    plt.axis("equal")
    plt.legend()
    plt.grid(True)
    plt.show()

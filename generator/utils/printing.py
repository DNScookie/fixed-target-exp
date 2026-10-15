from typing import List

from generator.models import Event


def print_particles(event: Event, limit=20):
    print(f"Vertex: ({event.vertex.x:.3f}, {event.vertex.y:.3f}, {event.vertex.z:.3f})")
    for p in event.particles[:limit]:
        print(p)


def print_hits(hits: List, limit=20):
    print(f"Total hits: {len(hits)} (true + fake)")
    for h in hits[:limit]:
        print(h)

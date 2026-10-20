from generator.generation import example_generator_config, build_naive_detectors, generate_to_dict
from generator.models import ParticleGenerator
from generator.simulation import simulate_hits
from generator.utils.fileio import to_file, from_file
from generator.utils.printing import print_particles, print_hits
from generator.utils.visualization import visualize_3d, visualize_2d
from tracking.transformations import hits_to_tracks
from tracking.visualization import visualize_tracks_2d

generator_config = example_generator_config()
particle_generator = ParticleGenerator(config=generator_config)
event = particle_generator.generate_event()
print_particles(event)

exp_setup = build_naive_detectors(n_layers=7)
hits = simulate_hits(event, experiment_setup=exp_setup, event_id=0)
print_hits(hits)
visualize_3d(event, hits)
visualize_2d(event, hits)

sim_dict = generate_to_dict(
    n_events=5,
    generator_config=generator_config,
    experiment_setup=exp_setup,
    seed=42,
)

filename = "data/simulated_data.json"
to_file(sim_dict, filename)
loaded_dict = from_file(filename)

tracks = hits_to_tracks(hits)
visualize_tracks_2d(hits, tracks, normal=[1, 0, 0])


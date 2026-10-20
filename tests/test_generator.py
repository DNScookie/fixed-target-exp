import unittest
import numpy as np
from generator.models import Event, Vertex, Particle, MagneticField
from generator.experiment import ExperimentSetup
from generator.detectors import NaiveDetector
from generator.generation import build_naive_detectors
from generator.analytic import advance_z, simulate_hits_analytic
from generator.simulation import simulate_hits


def particle(momentum=(.1, 0., 1.)):
    return Particle(0, 'p', 1, .938, np.array(momentum), 1.4, 0., 0.)


class GeneratorTests(unittest.TestCase):
    def test_noise_region_includes_detector_center(self):
        d = NaiveDetector('shifted', 1., .5, .3, center_x=2., center_y=-1., noise_rate=20.)
        for i in range(20):
            d.register(np.array([2., -1., 1.]), particle(), i)
        hits = d.finalize_hits(0, 0)
        self.assertGreater(sum(h.is_fake for h in hits), 0)
        self.assertTrue(all(abs(h.x-2.) <= .25 and abs(h.y+1.) <= .15 for h in hits))
        self.assertAlmostEqual(d.area(), .15)

    def test_rng_advances_and_reset_reproduces(self):
        d = NaiveDetector('d', 1., .5, .5, noise_rate=10.)
        def response():
            for i in range(5):
                d.register(np.array([0., 0., 1.]), particle(), i)
            return [(h.x, h.y, h.is_fake) for h in d.finalize_hits(0, 0)]
        d.reset(71); first=response(); second=response()
        self.assertNotEqual(first, second)
        d.reset(71); self.assertEqual(first, response())

    def test_seed_and_contiguous_ids(self):
        setup=build_naive_detectors(seed=99)
        self.assertEqual([d.seed for d in setup.detectors], list(range(99,106)))
        event=Event(Vertex(0.,0.,0.), [particle()])
        hits=simulate_hits_analytic(event,setup)
        self.assertEqual([h.hit_id for h in hits], list(range(len(hits))))

    def test_stepper_converges_to_helix_with_correct_sign(self):
        def setup():
            return ExperimentSetup(MagneticField(np.array([0.,0.,.8]), -.1, 2.),
                [NaiveDetector('d', 1., 10., 10., efficiency=1., noise_rate=0.)])
        event=Event(Vertex(0.,0.,0.), [particle()])
        exact=simulate_hits_analytic(event,setup())[0]
        errors=[]
        for ds in [.02, .005]:
            hits=simulate_hits(event,setup(),ds=ds)
            self.assertEqual(len(hits),1)
            h=hits[0]
            errors.append(np.linalg.norm([h.x-exact.x,h.y-exact.y]))
            self.assertLess(h.y,0.)
        self.assertLess(errors[1], errors[0]/2)
        self.assertLess(errors[1], 1e-4)

    def test_field_boundary_segments(self):
        setup=ExperimentSetup(MagneticField(np.array([0.,0.,.8]),.7,1.6),
            [NaiveDetector('d',1.,10.,10.,efficiency=1.,noise_rate=0.)])
        p=particle(); event=Event(Vertex(0.,0.,0.),[p])
        hit=simulate_hits_analytic(event,setup)[0]
        a,pa=advance_z(np.zeros(3),p.momentum,1,.7,0.)
        b,pb=advance_z(a,pa,1,1.,.8)
        self.assertTrue(np.allclose([hit.x,hit.y,hit.z],b))
        self.assertAlmostEqual(np.linalg.norm(pb),np.linalg.norm(p.momentum))


if __name__=='__main__':
    unittest.main()

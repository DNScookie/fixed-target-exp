"""Smoke tests for the interchangeable PyG convolution operators."""
import unittest
import torch
from torch_geometric.data import Data
from experiments.pareto.study import TrackNet


class TestTrackNetArchitectures(unittest.TestCase):
    def test_edge_and_node_forward_backward(self):
        torch.manual_seed(13)
        data = Data(
            x=torch.randn(6, 3),
            edge_index=torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]]))
        for name in ('gcn', 'sage', 'gat', 'edgeconv', 'gatv2',
                     'transformer', 'gine', 'resgated'):
            for node_task in (False, True):
                with self.subTest(architecture=name, node_task=node_task):
                    net = TrackNet(architecture=name, hidden=16, layers=2,
                                   heads=2, node_task=node_task)
                    out = net(data)
                    self.assertEqual(tuple(out.shape),
                                     (data.num_nodes if node_task else data.num_edges,))
                    self.assertTrue(torch.isfinite(out).all())
                    out.sum().backward()
                    self.assertTrue(any(p.grad is not None for p in net.parameters()))

    def test_unknown_architecture(self):
        with self.assertRaises(ValueError):
            TrackNet(architecture='unknown')


if __name__ == '__main__':
    unittest.main()

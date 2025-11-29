import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from causal_compression.continuity_field import ContinuityField


class TestContinuityField(unittest.TestCase):
    def setUp(self):
        self.dim = 3
        self.cf = ContinuityField(embedding_dim=self.dim, k_neighbors=3)

    def test_add_anchor(self):
        vec = np.array([1.0, 0.0, 0.0])
        self.cf.add_anchor(vec)
        self.assertEqual(len(self.cf.anchors), 1)
        np.testing.assert_array_equal(self.cf._anchor_matrix[0], vec)

    def test_projection_on_manifold(self):
        # Create a simple 2D plane in 3D: z = 0
        p1 = np.array([1.0, 1.0, 0.0])
        p2 = np.array([2.0, 1.0, 0.0])
        p3 = np.array([1.0, 2.0, 0.0])

        self.cf.add_anchor(p1)
        self.cf.add_anchor(p2)
        self.cf.add_anchor(p3)

        # Query a point ON the plane
        query = np.array([1.5, 1.5, 0.0])
        projection, residual = self.cf.compute_projection(query)

        # Residual should be near zero
        self.assertAlmostEqual(np.linalg.norm(residual), 0.0)
        # Projection should be the point itself
        np.testing.assert_array_almost_equal(projection, query)

    def test_projection_off_manifold(self):
        # Create a simple 2D plane in 3D: z = 0
        p1 = np.array([1.0, 1.0, 0.0])
        p2 = np.array([2.0, 1.0, 0.0])
        p3 = np.array([1.0, 2.0, 0.0])

        self.cf.add_anchor(p1)
        self.cf.add_anchor(p2)
        self.cf.add_anchor(p3)

        # Query a point OFF the plane (z=1)
        query = np.array([1.5, 1.5, 1.0])
        projection, residual = self.cf.compute_projection(query)

        # Residual should be approx [0, 0, 1]
        expected_residual = np.array([0.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(residual, expected_residual)

        # Drift metric should be 1.0
        drift = self.cf.get_drift_metric(query)
        self.assertAlmostEqual(drift, 1.0)


if __name__ == "__main__":
    unittest.main()

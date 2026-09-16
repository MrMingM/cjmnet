import unittest
import numpy as np
from . import codec as c


class PacketTests(unittest.TestCase):
    def test_roundtrip_bytes_and_abstention(self):
        grid, ids, z = (5, 7), np.array([0, 17, 34]), 4
        query = c.pack_query(ids, grid)
        np.testing.assert_array_equal(c.unpack_query(query, grid), ids)
        values = np.random.default_rng(3).random((3, 8, 8, z, 4))
        values[0] = 0
        packet = c.pack_evidence(ids, values, grid, z)
        decoded = c.unpack_evidence(packet, grid, z, ids)
        np.testing.assert_allclose(decoded, np.rint(values*255)/255, atol=1e-7)
        self.assertFalse(decoded[0].any())
        self.assertEqual(len(query)+len(packet), 32+len(ids)*(8+64*z*4))

    def test_budget_edges(self):
        for budget in (0, 1, 100, 1000, 262144):
            for peers in range(6):
                n = c.query_count(budget, peers, .125, 32, 100, 4)
                actual = peers*(32+n*(8+1024)) if n else 0
                self.assertLessEqual(actual, int(budget*.125))
                self.assertLessEqual(n, 32)

    def test_malformed_and_unrequested(self):
        ids, grid = np.array([1]), (3, 4)
        packet = c.pack_evidence(ids, np.zeros((1, 8, 8, 4, 4)), grid, 4)
        for bad in (b'', packet[:-1], packet+b'!'):
            with self.assertRaises(ValueError):
                c.unpack_evidence(bad, grid, 4, ids)
        with self.assertRaises(ValueError):
            c.unpack_evidence(packet, grid, 4, [2])
        with self.assertRaises(ValueError):
            c.pack_query([1, 1], grid)
        with self.assertRaises(ValueError):
            c.unpack_query(c.pack_query(ids, grid), (4, 3))


if __name__ == '__main__':
    unittest.main()

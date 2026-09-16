"""CPU-only checks that can run in the local Windows environment."""
import unittest
import numpy as np
from . import codec


class CodecTests(unittest.TestCase):
    def test_request_roundtrip_and_length(self):
        a = np.array([[0., .5, 1.], [.2, .75, .33]])
        packet = codec.pack_request(a)
        self.assertEqual(len(packet), codec.request_bytes(2, 3))
        np.testing.assert_allclose(codec.unpack_request(packet), a, atol=1/255)
        with self.assertRaises(ValueError):
            codec.unpack_request(packet[:-1])

    def test_multiscale_roundtrip_and_missing_blocks(self):
        levels = [np.arange(c*2*p*3*p, dtype=np.float32).reshape(c, 2*p, 3*p)
                  for c, p in [(2, 4), (3, 2), (4, 1)]]
        ids = [0, 5]
        for width in (2, 4):
            packet = codec.pack_response(levels, ids, (2, 3), width)
            cost = codec.block_bytes([2, 3, 4], [4, 2, 1], width)
            self.assertEqual(len(packet), 16 + len(ids)*cost)
            decoded, recovered = codec.unpack_response(packet, [a.shape for a in levels], (2, 3))
            np.testing.assert_equal(recovered, ids)
            for a, b in zip(levels, decoded):
                p = a.shape[1]//2
                expected = np.zeros_like(a)
                expected[:, :p, :p] = a[:, :p, :p]
                expected[:, p:, 2*p:] = a[:, p:, 2*p:]
                np.testing.assert_equal(b, expected)

    def test_budget_edges(self):
        cost = codec.block_bytes([64, 128, 256], [4, 2, 1])
        self.assertEqual(cost, 7172)
        for peers in range(5):
            for budget in [0, 1, 5000, 20000, 262144, 100000000]:
                active, quotas = codec.allocate(budget, peers, 25, 88, cost)
                total = peers*(codec.request_bytes(25, 88)+16)+sum(quotas)*cost if active else 0
                self.assertLessEqual(total, budget)
                self.assertTrue(all(0 <= q <= 2200 for q in quotas))
                self.assertEqual(len(quotas), peers)

    def test_empty_and_bad_responses(self):
        levels = [np.ones((2, 2, 3), dtype=np.float32)]
        packet = codec.pack_response(levels, [], (2, 3))
        decoded, _ = codec.unpack_response(packet, [levels[0].shape], (2, 3))
        self.assertEqual(float(decoded[0].sum()), 0)
        with self.assertRaises(ValueError):
            codec.pack_response(levels, [1, 1], (2, 3))
        with self.assertRaises(ValueError):
            codec.unpack_response(packet+b'x', [levels[0].shape], (2, 3))


if __name__ == '__main__':
    unittest.main()

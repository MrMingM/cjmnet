import unittest
import numpy as np

from lossless_comm.codec import pack_tensors, unpack_tensors


class TestLosslessCodec(unittest.TestCase):
    def roundtrip(self, arrays, backend="zlib"):
        packet, stat = pack_tensors(arrays, backend=backend, level=6)
        decoded = unpack_tensors(packet, [x.shape for x in arrays])
        self.assertEqual(len(arrays), len(decoded))
        for a, b in zip(arrays, decoded):
            self.assertEqual(a.shape, b.shape)
            self.assertTrue(
                np.array_equal(
                    np.ascontiguousarray(a).view(np.uint32),
                    np.ascontiguousarray(b).view(np.uint32),
                )
            )
        return stat

    def test_dense_random(self):
        rng = np.random.default_rng(7)
        a = rng.standard_normal((64, 40, 140), dtype=np.float32)
        self.roundtrip([a])

    def test_relu_sparse(self):
        rng = np.random.default_rng(8)
        a = rng.standard_normal((64, 40, 140), dtype=np.float32)
        a[a < 0] = 0
        stat = self.roundtrip([a])
        self.assertGreater(stat["tensors"][0]["zero_fraction"], 0.45)

    def test_three_scales(self):
        rng = np.random.default_rng(9)
        arrays = [
            np.maximum(rng.standard_normal((64, 80, 280), dtype=np.float32), 0),
            np.maximum(rng.standard_normal((128, 40, 140), dtype=np.float32), 0),
            np.maximum(rng.standard_normal((256, 20, 70), dtype=np.float32), 0),
        ]
        self.roundtrip(arrays)

    def test_negative_zero_is_preserved(self):
        a = np.array([0.0, -0.0, 1.0, -2.0], dtype=np.float32)
        self.roundtrip([a])


if __name__ == "__main__":
    unittest.main()

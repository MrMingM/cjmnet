from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import torch
from gspr_communication import codec, runtime as rt
from .core import drop_mask
from .joint_candidates import improving, candidates, replay, packet_bytes


class JointCandidateTests(unittest.TestCase):
    def test_candidates_include_beneficial_and_reject_tradeoffs(self):
        rows = []
        for x, label, recovered, lost, fp in [(0, 'beneficial', 1, 0, 0),
            (1, 'harmful', 0, 0, 0), (2, 'neutral', 0, 0, -1), (3, 'harmful', 1, 1, 0)]:
            rows.append({'sender_index': 1, 'blocks_yx': [[0, x]], 'removed_native_blocks': 1,
                'class': label, 'recovered_after_drop_iou70': recovered,
                'lost_after_drop_iou70': lost, 'fp_change_after_drop': fp})
        delta, improve = candidates(rows, 4, [(1, i) for i in range(4)])
        self.assertEqual(delta, [(1, 1), (1, 3)])
        self.assertEqual(improve, [(1, 0), (1, 2)])
        with self.assertRaises(ValueError):
            candidates(rows[:-1], 4, [(1, i) for i in range(4)])
        with self.assertRaises(ValueError):
            candidates(rows+rows[:1], 4, [(1, i) for i in range(4)])

    def test_replay_uses_saved_sender_and_counts_actual_packets(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            model = SimpleNamespace(grid=(2, 2), value_bytes=4,
                cost=codec.block_bytes([1, 1, 1], [4, 2, 1]))
            levels = [torch.randn(2, 1, side, side) for side in (8, 4, 2)]
            packet = codec.pack_response([x[1].numpy() for x in levels], [0, 3], model.grid)
            request = codec.pack_request(torch.ones(2, 2).numpy())
            (directory/'sender_1.bin').write_bytes(packet)
            (directory/'request.bin').write_bytes(request)
            meta = {'packet_sha256': {p.name: rt.sha256(p) for p in directory.glob('*.bin')},
                'total_bytes': len(packet)+len(request), 'selected_sender_blocks': [[1, 0], [1, 3]]}
            changed = [x.clone() for x in levels]
            for x in changed:
                x[1] = 10000
            msg = replay(model, changed, directory, meta)
            decoded, _ = codec.unpack_response(packet, [tuple(x.shape[1:]) for x in levels], model.grid)
            for x, old, source in zip(msg.levels, decoded, levels):
                torch.testing.assert_close(x[1], torch.from_numpy(old), atol=0, rtol=0)
                torch.testing.assert_close(x[0], source[0], atol=0, rtol=0)
            mask = drop_mask(msg.mask, [(1, 0)])
            self.assertEqual(packet_bytes(model, msg, mask), meta['total_bytes']-model.cost)
            (directory/'sender_1.bin').write_bytes(packet+b'x')
            with self.assertRaises(ValueError):
                replay(model, levels, directory, meta)

    def test_rule_matches_opportunity_check(self):
        for recovered in (0, 1):
            for lost in (0, 1):
                for fp in (-1, 0, 1):
                    expected = lost == 0 and recovered >= 0 and fp <= 0 and (recovered > 0 or fp < 0)
                    self.assertEqual(improving({'recovered_after_drop_iou70': recovered,
                        'lost_after_drop_iou70': lost, 'fp_change_after_drop': fp}), expected)


if __name__ == '__main__':
    unittest.main()

import unittest
import torch
from .harm_diagnostic import deletion_orders, filter_mask, candidates, empty_stats, merge_stats
from .test_evidence import toy_model


class HarmTests(unittest.TestCase):
    def test_equal_counts_and_ego_untouched(self):
        obs = torch.zeros(3, 13, 2, 4)
        obs[1, 3, 0] = 1
        obs[2, 3] = 1
        obs[1:, 0] = torch.arange(8).reshape(2, 4)/8
        obs[1:, 1] = torch.arange(8).reshape(2, 4)/8
        full = torch.ones(3, 1, 2, 4)
        orders = deletion_orders(obs, 7)
        for fraction in (.01, .05, .1, .5):
            masks = [filter_mask(full, orders, s, fraction) for s in ('low_r', 'high_u', 'random')]
            self.assertEqual(len({float(m.sum()) for m in masks}), 1)
            for mask in masks:
                self.assertTrue(mask[0].all())
                self.assertTrue(mask[1, 0, 1].all())  # no-point regions excluded
        self.assertEqual(orders[0]['low_r'][0], 0)
        self.assertEqual(orders[0]['high_u'][0], 3)

    def test_seed_deduplication_and_empty_sender(self):
        obs = torch.rand(3, 13, 3, 4)
        obs[:, 3] = 1
        obs[2, 3] = 0
        a, b = deletion_orders(obs, 3), deletion_orders(obs, 3)
        torch.testing.assert_close(a[0]['random'], b[0]['random'])
        rows = candidates(a)
        self.assertEqual(len(rows), len({(p, i) for p, i, _ in rows}))
        self.assertTrue(all(p == 1 for p, _, _ in rows))
        self.assertLessEqual(len(rows), 6)

    def test_single_removal_resets_full_context_and_matches_wire(self):
        model, encoded = toy_model()
        full = torch.ones_like(model.empty_masks(encoded))
        for block in (0, 3):
            mask = full.clone()
            mask[1].flatten()[block] = 0
            self.assertEqual(int((full-mask).sum()), 1)
            fast, _ = model.detect(encoded, mask, False)
            wire, _ = model.detect(encoded, mask, True)
            for key in fast:
                torch.testing.assert_close(fast[key], wire[key])
        self.assertTrue(full.all())

    def test_ap_accumulation_does_not_alias_frames(self):
        total, frame = empty_stats(), empty_stats()
        frame[.7].update(tp=[1], fp=[0], gt=2, score=[.8])
        merge_stats(total, frame)
        total[.7]['tp'][0] = 99
        self.assertEqual(frame[.7]['tp'], [1])
        self.assertEqual(total[.7]['gt'], 2)


if __name__ == '__main__':
    unittest.main()

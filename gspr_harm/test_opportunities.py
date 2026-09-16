import unittest
from .check_opportunities import summarize


class OpportunityTests(unittest.TestCase):
    def row(self, label, recovered, lost, fp):
        return dict(scene='scene_a', sample_index=1, sender_id='a', blocks_yx=[[0, 0]],
            delta_send_minus_drop={'total_loss': 0.}, **{'class': label},
            recovered_after_drop_iou70=recovered, lost_after_drop_iou70=lost, fp_change_after_drop=fp)

    def test_missed_improvement_and_tradeoffs(self):
        rows = [self.row('beneficial', 1, 0, 0), self.row('neutral', 0, 0, -1),
                self.row('harmful', 2, 1, 0), self.row('harmful', 1, 0, 1)]
        result = summarize(rows, {'beneficial': 1, 'neutral': 1, 'harmful': 2})
        self.assertEqual(result['improving_single_deletions'], 2)
        self.assertEqual(result['missed_fraction'], 1.)
        self.assertEqual(result['scenes_with_missed_improvements'], 1)

    def test_no_opportunities_does_not_claim_no_ap_headroom(self):
        result = summarize([self.row('harmful', 0, 0, 0)], {'harmful': 1, 'beneficial': 0, 'neutral': 0})
        self.assertIsNone(result['missed_fraction'])
        self.assertIn('不能', result['interpretation'])


if __name__ == '__main__':
    unittest.main()

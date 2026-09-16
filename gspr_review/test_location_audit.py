import unittest
from .audit_location import differences


class LocationAuditTests(unittest.TestCase):
    def row(self):
        return dict(sample_index=7, total_bytes=256, eligible_points=8, message_sha256='abc')

    def test_equal(self):
        self.assertTrue(differences([self.row()], [self.row()])['matches'])

    def test_hash_only_is_not_silently_accepted(self):
        changed = dict(self.row(), message_sha256='xyz')
        result = differences([self.row()], [changed])
        self.assertFalse(result['matches'])
        self.assertEqual(result['mismatch_counts']['message_sha256'], 1)
        self.assertEqual(result['mismatch_counts']['eligible_points'], 0)

    def test_structural_differences_and_missing_hash(self):
        changed = dict(self.row(), eligible_points=9, total_bytes=255)
        del changed['message_sha256']
        result = differences([self.row()], [changed])
        self.assertFalse(result['matches'])
        self.assertEqual(set(result['first_examples'][0]['differences']), {'eligible_points', 'total_bytes', 'message_sha256'})

    def test_frame_count_mismatch(self):
        self.assertFalse(differences([self.row()], [])['matches'])


if __name__ == '__main__':
    unittest.main()

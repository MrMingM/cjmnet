import tempfile
import unittest
from pathlib import Path
from .packet_replay import PacketReplay


class PacketReplayTests(unittest.TestCase):
    def test_exact_bytes_and_candidate_discrepancy(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'frame.json'
            record = PacketReplay(path, 'record', 'input')
            self.assertEqual(record.exchange('query', b'abc'), b'abc')
            record.finish()
            replay = PacketReplay(path, 'replay', 'input')
            self.assertEqual(replay.exchange('query', b'abd'), b'abc')
            replay.finish()
            self.assertEqual(replay.differences, {'query': 1})
            with self.assertRaises(ValueError):
                PacketReplay(path, 'record', 'input')
            with self.assertRaises(ValueError):
                PacketReplay(path, 'replay', 'different input')

    def test_sequence_and_missing_packets_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'frame.json'
            record = PacketReplay(path, 'record', 'input')
            record.exchange('query', b'abc')
            record.finish()
            replay = PacketReplay(path, 'replay', 'input')
            with self.assertRaises(ValueError):
                replay.exchange('response', b'abc')
            with self.assertRaises(ValueError):
                replay.finish()


if __name__ == '__main__':
    unittest.main()

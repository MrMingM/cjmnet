"""Offline paired evaluation: canonical wire bytes, guarded by input fingerprints."""
import base64
import hashlib
import json
from pathlib import Path


def fingerprint(data):
    digest = hashlib.sha256()
    values = dict(data['processed_lidar'], record_len=data['record_len'])
    for key in sorted(values):
        value = values[key]
        if not hasattr(value, 'detach'):
            continue
        a = value.detach().cpu().contiguous().numpy()
        digest.update(key.encode())
        digest.update(str((a.shape, a.dtype)).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


class PacketReplay:
    def __init__(self, path, mode, input_hash):
        self.path, self.mode, self.input_hash = Path(path), mode, input_hash
        self.index, self.entries, self.differences = 0, [], {}
        if mode == 'replay':
            stored = json.loads(self.path.read_text(encoding='utf-8'))
            if stored['input_sha256'] != input_hash:
                raise ValueError('Packet replay input mismatch: '+str(self.path))
            self.entries = stored['packets']
        elif mode != 'record' or self.path.exists():
            raise ValueError('Recording requires a fresh packet file')

    def exchange(self, kind, candidate):
        if self.mode == 'record':
            self.entries.append({'kind': kind, 'bytes': base64.b64encode(candidate).decode('ascii')})
            result = candidate
        else:
            if self.index >= len(self.entries) or self.entries[self.index]['kind'] != kind:
                raise ValueError('Packet replay sequence differs at '+kind)
            result = base64.b64decode(self.entries[self.index]['bytes'], validate=True)
            if candidate != result:
                self.differences[kind] = self.differences.get(kind, 0)+1
        self.index += 1
        return result

    def finish(self):
        if self.index != len(self.entries):
            raise ValueError('Not all recorded messages were consumed')
        if self.mode == 'record':
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open('x', encoding='utf-8') as stream:
                json.dump({'input_sha256': self.input_hash, 'packets': self.entries}, stream)

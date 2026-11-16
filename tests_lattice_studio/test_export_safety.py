"""Small real STL writes with injected disk and publication failures."""
import errno
import os
import struct
import tempfile
import unittest
from collections import namedtuple
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import trimesh

from latticefill.mesh_memory import export_stl_batched


DiskUsage = namedtuple('DiskUsage', 'total used free')


class FaultyStream:
    """Keep actual file I/O, failing only the requested storage operation."""
    def __init__(self, stream, fault):
        self.stream = stream
        self.fault = fault

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def __enter__(self):
        self.stream.__enter__()
        return self

    def __exit__(self, *args):
        result = self.stream.__exit__(*args)
        if self.fault == 'close' and args[0] is None:
            raise OSError(errno.ENOSPC, 'injected close failure')
        return result

    def write(self, data):
        if self.stream.tell() >= 84:
            if self.fault == 'enospc':
                self.stream.write(data[:7])
                raise OSError(errno.ENOSPC, 'injected disk exhaustion')
            if self.fault == 'short':
                return self.stream.write(data[:-1])
            if self.fault == 'silent_short':
                self.stream.write(data[:-1])
                return len(data)
            if self.fault == 'cancel':
                raise KeyboardInterrupt('injected cancellation')
        return self.stream.write(data)

    def flush(self):
        if self.fault == 'flush':
            raise OSError(errno.ENOSPC, 'injected flush failure')
        return self.stream.flush()


@contextmanager
def failing_writes(fault):
    real_open, real_fdopen = open, os.fdopen

    def wrap_open(*args, **kwargs):
        return FaultyStream(real_open(*args, **kwargs), fault)

    def wrap_fdopen(*args, **kwargs):
        return FaultyStream(real_fdopen(*args, **kwargs), fault)

    with patch('builtins.open', side_effect=wrap_open), \
            patch('os.fdopen', side_effect=wrap_fdopen):
        yield


class ExportSafetyTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix='stl export safety ')
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.path = self.root / 'result with spaces.stl'
        self.previous = b'previous complete result'
        self.path.write_bytes(self.previous)
        self.unrelated = self.root / 'result with spaces.stl.other.partial'
        self.unrelated.write_bytes(b'unrelated partial must survive')
        self.mesh = trimesh.creation.box([2, 4, 6])
        self.space = patch('shutil.disk_usage', return_value=DiskUsage(10**12, 0, 10**12))
        self.space.start()
        self.addCleanup(self.space.stop)

    def assert_failure_preserved_files(self, existing=True):
        if existing:
            self.assertEqual(self.path.read_bytes(), self.previous)
        else:
            self.assertFalse(self.path.exists())
        self.assertEqual(self.unrelated.read_bytes(), b'unrelated partial must survive')
        expected = {self.unrelated, self.path} if existing else {self.unrelated}
        self.assertEqual(set(self.root.iterdir()), expected)

    def test_preflight_uses_destination_disk_and_keeps_reserve(self):
        # A 12-face box requires exactly 684 bytes, plus a 64 MiB minimum reserve.
        checked = []

        def disk_usage(destination):
            checked.append(Path(destination).resolve())
            return DiskUsage(10**12, 0, 684 + 64 * 1024**2 - 1)

        with patch('shutil.disk_usage', side_effect=disk_usage):
            with self.assertRaises(OSError) as caught:
                export_stl_batched(self.mesh, self.path)
        self.assertEqual(caught.exception.errno, errno.ENOSPC)
        self.assertEqual(checked, [self.root.resolve()])
        self.assertIn('684', str(caught.exception))
        self.assert_failure_preserved_files()

    def test_preflight_reserve_boundary_allows_complete_export(self):
        with patch('shutil.disk_usage', return_value=DiskUsage(10**12, 0, 684 + 64 * 1024**2)):
            export_stl_batched(self.mesh, str(self.path))
        self.assertEqual(self.path.stat().st_size, 684)

    def test_large_export_checks_size_scaled_reserve_without_allocating_mesh(self):
        class LargeFaces:
            def __len__(self):
                return 28_000_000

            def __getitem__(self, key):
                raise AssertionError('preflight must reject before accessing triangles')

        # 1,400,000,084-byte STL fits, but does not leave its 5% reserve.
        with patch('shutil.disk_usage', return_value=DiskUsage(10**12, 0, 1_470_000_088)):
            with self.assertRaises(OSError) as caught:
                export_stl_batched(SimpleNamespace(faces=LargeFaces(), vertices=np.empty((0, 3))), self.path)
        self.assertEqual(caught.exception.errno, errno.ENOSPC)
        self.assertIn('1400000084', str(caught.exception))
        self.assert_failure_preserved_files()

    def test_unknown_free_space_fails_before_touching_final(self):
        with patch('shutil.disk_usage', side_effect=OSError('cannot inspect disk')):
            with self.assertRaisesRegex(OSError, 'cannot inspect disk'):
                export_stl_batched(self.mesh, self.path)
        self.assert_failure_preserved_files()

    def test_mid_write_disk_exhaustion_preserves_existing_or_absent_final(self):
        for existing in (True, False):
            with self.subTest(existing=existing):
                if not existing:
                    self.path.unlink()
                with failing_writes('enospc'):
                    with self.assertRaises(OSError) as caught:
                        export_stl_batched(self.mesh, self.path)
                self.assertEqual(caught.exception.errno, errno.ENOSPC)
                self.assert_failure_preserved_files(existing)

    def test_short_write_never_publishes_incomplete_stl(self):
        for fault in ('short', 'silent_short'):
            with self.subTest(fault=fault):
                with failing_writes(fault):
                    with self.assertRaises(OSError):
                        export_stl_batched(self.mesh, self.path)
                self.assert_failure_preserved_files()

    def test_flush_failure_preserves_final(self):
        with failing_writes('flush'):
            with self.assertRaises(OSError):
                export_stl_batched(self.mesh, self.path)
        self.assert_failure_preserved_files()

    def test_close_failure_preserves_final(self):
        with failing_writes('close'):
            with self.assertRaises(OSError):
                export_stl_batched(self.mesh, self.path)
        self.assert_failure_preserved_files()

    def test_sync_failure_preserves_final(self):
        with patch('os.fsync', side_effect=OSError(errno.ENOSPC, 'sync failed')):
            with self.assertRaises(OSError):
                export_stl_batched(self.mesh, self.path)
        self.assert_failure_preserved_files()

    def test_cancellation_cleans_only_its_own_partial(self):
        with failing_writes('cancel'):
            with self.assertRaises(KeyboardInterrupt):
                export_stl_batched(self.mesh, self.path)
        self.assert_failure_preserved_files()

    def test_publication_failure_preserves_final(self):
        with patch('os.replace', side_effect=PermissionError('destination is locked')):
            with self.assertRaisesRegex(PermissionError, 'destination is locked'):
                export_stl_batched(self.mesh, self.path)
        self.assert_failure_preserved_files()

    def test_roundtrip_is_batched_and_final_changes_only_at_atomic_publication(self):
        original_vertices = self.mesh.vertices.copy()
        original_faces = self.mesh.faces.copy()
        source_vertices = self.mesh.vertices
        batch_lengths = []
        published = []
        real_replace = os.replace
        shift = np.array([10., -20., 30.])
        owner = self

        class BoundedVertices:
            def __getitem__(self, faces):
                owner.assertEqual(owner.path.read_bytes(), owner.previous)
                owner.assertLessEqual(len(faces), 5)
                batch_lengths.append(len(faces))
                return source_vertices[faces]

        def publish(source, destination):
            source, destination = Path(source), Path(destination)
            self.assertEqual(destination.resolve(), self.path.resolve())
            self.assertEqual(source.parent.resolve(), self.root.resolve())
            self.assertNotEqual(source, self.path)
            self.assertNotEqual(source, self.unrelated)
            self.assertEqual(self.path.read_bytes(), self.previous)
            data = source.read_bytes()
            self.assertEqual(len(data), 684)
            self.assertEqual(struct.unpack('<I', data[80:84]), (12,))
            published.append(source)
            real_replace(source, destination)

        bounded_mesh = SimpleNamespace(faces=self.mesh.faces, vertices=BoundedVertices())
        with patch('latticefill.mesh_memory.FACE_BATCH', 5), patch('os.replace', side_effect=publish):
            export_stl_batched(bounded_mesh, self.path, shift)
        self.assertEqual(batch_lengths, [5, 5, 2])
        self.assertEqual(len(published), 1)
        self.assertFalse(published[0].exists())
        self.assertEqual(set(self.root.iterdir()), {self.path, self.unrelated})
        actual = trimesh.load(self.path, force='mesh')
        self.assertTrue(actual.is_volume)
        self.assertAlmostEqual(actual.volume, 48.)
        np.testing.assert_allclose(actual.bounds, [[9., -22., 27.], [11., -18., 33.]])
        records = np.frombuffer(self.path.read_bytes()[84:], dtype=np.dtype([
            ('normal', '<f4', (3,)), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')]))
        np.testing.assert_allclose(records['normal'], self.mesh.face_normals)
        np.testing.assert_array_equal(records['attribute'], np.zeros(12))
        np.testing.assert_array_equal(self.mesh.vertices, original_vertices)
        np.testing.assert_array_equal(self.mesh.faces, original_faces)


if __name__ == '__main__':
    unittest.main()

import errno
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from latticefill.studio_jobs import compute
from tests_lattice_studio.test_surface_stitch import split_box


class ExportCheckpointTests(unittest.TestCase):
    def test_export_failure_keeps_geometry_and_statistics_without_done_result(self):
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder);body,_=split_box()
            request=dict(action='generate',model_dir=str(directory/'source'),body=0,
                config=dict(lattice='cubic',cell=4,wall=1.2,pitch=.4,axis='z',scope='body',
                            surface_mode='volume',quality='final'))
            with patch('latticefill.studio_jobs.read_body',return_value=body),patch('latticefill.mesh_memory.export_stl_batched',side_effect=OSError(errno.ENOSPC,'test full disk')):
                with self.assertRaises(OSError):compute(directory,request)
            checkpoint=json.loads((directory/'export_checkpoint.json').read_text(encoding='utf-8'))
            self.assertGreater(checkpoint['stats']['faces'],0)
            self.assertEqual(checkpoint['bytes'],84+50*checkpoint['stats']['faces'])
            self.assertTrue((directory/'display.glb').is_file())
            self.assertFalse((directory/'result.stl').exists())
            self.assertFalse((directory/'design.json').exists())
            state=json.loads((directory/'status.json').read_text(encoding='utf-8'))
            self.assertNotEqual(state['state'],'done')


if __name__=='__main__':unittest.main()

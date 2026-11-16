import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import trimesh
from latticefill.mesh_memory import export_stl_batched
from latticefill.mesh_quality import export_display
from latticefill.studio_jobs import atomic_json
from tests_lattice_studio.validate_atlas_output import main


class SavedOutputAuditTests(unittest.TestCase):
    def test_audits_original_large_layout_with_world_translation(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);output=root/'audit';mesh=trimesh.creation.box([2,4,6])
            centre=np.array([10.,-20.,1800.])
            with patch('latticefill.mesh_quality.DIRECT_DISPLAY_FACE_THRESHOLD',1):
                quality=export_display(mesh,root/'display.glb',1,.25,centre)
            export_stl_batched(mesh,root/'result.stl',centre)
            atomic_json(root/'status.json',dict(state='done'))
            atomic_json(root/'design.json',dict(bounds=(mesh.bounds+centre).tolist(),
                display_quality=quality,stats=dict(faces=12,volume=48.,source_trimmed=True,components=1)))
            with patch.object(sys,'argv',['audit',str(root),str(output)]),patch('tests_lattice_studio.validate_atlas_output.render'):
                main()
            report=json.loads((output/'final_audit.json').read_text())
            self.assertEqual(report['faces'],12)
            self.assertEqual(report['bytes'],684)
            self.assertLess(report['max_stl_glb_coordinate_error_mm'],1e-5)
            self.assertAlmostEqual(report['stl_signed_volume_mm3'],48.)


if __name__=='__main__':unittest.main()

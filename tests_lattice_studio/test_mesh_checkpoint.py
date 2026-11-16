import tempfile
import unittest
import gc
from pathlib import Path
import numpy as np
import trimesh
from latticefill.core import FillParams


class MeshCheckpointTests(unittest.TestCase):
    def test_roundtrip_and_reject_changed_parameters(self):
        from latticefill.mesh_checkpoint import save_pretrim,load_pretrim
        with tempfile.TemporaryDirectory() as folder:
            p=FillParams(work_dir=folder,cell=7.3,wall=1.2)
            p.stats={'memory_mode':'chunked','polygon_vertices':17}
            mesh=trimesh.creation.box([2,4,6])
            save_pretrim(mesh,p,.25,.18)
            restored,pitch,density=load_pretrim(Path(folder),p)
            np.testing.assert_array_equal(restored.faces,mesh.faces)
            np.testing.assert_array_equal(restored.vertices,mesh.vertices)
            self.assertEqual((pitch,density),(.25,.18))
            self.assertEqual(p.stats['polygon_vertices'],17)
            p.cell=8
            with self.assertRaisesRegex(ValueError,'参数'):load_pretrim(Path(folder),p)
            del restored;gc.collect()

    def test_partial_checkpoint_is_not_accepted(self):
        from latticefill.mesh_checkpoint import load_pretrim
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder)/'pretrim';directory.mkdir()
            np.save(directory/'vertices.npy',np.zeros((3,3)))
            with self.assertRaisesRegex(ValueError,'检查点'):load_pretrim(Path(folder),FillParams())

    def test_finishing_can_resume_without_smoothing_twice(self):
        from unittest.mock import patch
        from latticefill.mesh_checkpoint import load_pretrim
        from latticefill.core import fill_solid,_finish_mesh
        from tests_lattice_studio.test_atlas_surface import params
        from tests_lattice_studio.test_surface_stitch import split_box
        with tempfile.TemporaryDirectory() as folder:
            body,surface=split_box();p=params();p.memory_mode='chunked';p.smooth=2;p.work_dir=folder
            protected=body.selected_mesh([f['id'] for f in body.face_info if f['id'] not in surface.active])
            with patch('latticefill.mesh_checkpoint.CHECKPOINT_FACE_MIN',1),patch('latticefill.surface.trim_to_source',side_effect=RuntimeError('test interrupted trim')):
                with self.assertRaisesRegex(RuntimeError,'interrupted'):
                    fill_solid(body.mesh,p,preserve_faces=protected,surface_field=surface)
            restored,pitch,density=load_pretrim(Path(folder),p)
            with patch('latticefill.mesh_memory.smooth_taubin_batched',side_effect=AssertionError('must not smooth twice')):
                out=_finish_mesh(restored,body.mesh,p,pitch,density,lambda *a:None,pretrim_ready=True)
            self.assertTrue(out.is_volume);self.assertTrue(p.stats['source_trimmed'])
            del restored,out;gc.collect()


if __name__=='__main__':unittest.main()

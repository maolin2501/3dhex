import json
import gc
import tempfile
import unittest
from pathlib import Path
import numpy as np
import trimesh
from latticefill.core import FillParams,fill_solid
from latticefill.studio_jobs import validate_design,estimate


def config(**extra):
    value=dict(lattice='cubic',cell=4,wall=1,shell=.8,pitch=.25,axis='z',scope='body',
               surface_mode='volume',quality='final',preserve_faces=[0,1])
    value.update(extra);return value


class FaceThicknessTests(unittest.TestCase):
    def test_normalizes_only_selected_overrides_and_defaults_old_requests(self):
        cfg=validate_design(config(preserve_thicknesses={'0':'3'}))
        self.assertEqual(cfg['preserve_thicknesses'],{'0':3.0})
        self.assertEqual(cfg['shell'],.8)
        self.assertEqual(validate_design(config()).get('preserve_thicknesses'),{})

    def test_rejects_wrong_faces_and_invalid_dimensions(self):
        for override in ({'2':3},{'00':3},{'0':float('nan')},{'0':float('inf')},
                         {'0':0},{'0':-1},{'0':51},{'0':True},[3]):
            with self.subTest(override=override),self.assertRaises(ValueError):
                validate_design(config(preserve_thicknesses=override))

    def test_resolution_and_scratch_estimates_include_local_thickness(self):
        thin=validate_design(config(preserve_thicknesses={'0':.3}))
        self.assertFalse(estimate([[0,0,0],[10,10,10]],thin)['feature_resolved'])
        base=validate_design(config(memory_mode='chunked'))
        thick=validate_design(config(memory_mode='chunked',preserve_thicknesses={'0':3}))
        bounds=[[0,0,0],[100,100,100]]
        self.assertGreater(estimate(bounds,thick)['scratch_gb'],estimate(bounds,base)['scratch_gb'])

    def test_independent_thicknesses_in_dense_and_chunked_geometry(self):
        mesh=trimesh.creation.box([10,10,10]);mesh.apply_translation([5,5,5])
        top=mesh.submesh([np.flatnonzero(mesh.face_normals[:,2]>.9)],append=True)
        bottom=mesh.submesh([np.flatnonzero(mesh.face_normals[:,2]<-.9)],append=True)
        for mode in ('dense','chunked'):
            with self.subTest(mode=mode):
                p=FillParams(lattice='cubic',cell=4,wall=1,shell=.8,pitch=.25,memory_mode=mode)
                out=fill_solid(mesh,p,preserve_faces=trimesh.util.concatenate([top,bottom]),
                               preserve_regions=[(top,3.),(bottom,.8)])
                self.assertTrue(out.is_volume)
                np.testing.assert_array_equal(out.contains([[1.7,1.7,7.4],[1.7,1.7,6.5],
                    [1.7,1.7,.4],[1.7,1.7,1.5],[1.7,1.7,10.6]]),[True,False,True,False,False])

    def test_thick_band_not_hidden_by_nearer_thin_face(self):
        mesh=trimesh.creation.box([10,10,10]);mesh.apply_translation([5,5,5])
        top=mesh.submesh([np.flatnonzero(mesh.face_normals[:,2]>.9)],append=True)
        side=mesh.submesh([np.flatnonzero(mesh.face_normals[:,0]<-.9)],append=True)
        p=FillParams(lattice='cubic',cell=4,wall=.6,shell=.8,pitch=.2,memory_mode='chunked')
        out=fill_solid(mesh,p,preserve_regions=[(top,3.),(side,.8)])
        self.assertTrue(out.contains([[1.5,1.7,7.5]])[0])

    def test_checkpoint_rejects_changed_overrides_but_accepts_legacy_empty(self):
        from latticefill.mesh_checkpoint import save_pretrim,load_pretrim
        with tempfile.TemporaryDirectory() as folder:
            p=FillParams(work_dir=folder,preserve_thicknesses={'0':3.})
            save_pretrim(trimesh.creation.box(),p,.25,.2)
            p.preserve_thicknesses={'0':2.}
            with self.assertRaisesRegex(ValueError,'参数'):load_pretrim(folder,p)
            ready=Path(folder)/'pretrim/ready.json';meta=json.loads(ready.read_text(encoding='utf-8'))
            del meta['params']['preserve_thicknesses'];ready.write_text(json.dumps(meta),encoding='utf-8')
            p.preserve_thicknesses={};restored,_,_=load_pretrim(folder,p)
            self.assertEqual(len(restored.faces),12)
            del restored
            gc.collect()

    def test_group_scratch_cleanup_preserves_unrecognized_files(self):
        from latticefill.lowmem import cleanup_job_scratch
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);owned=root/'scratch-lowmem-groups';owned.mkdir()
            (owned/'preserve-0.bin').write_bytes(b'owned')
            (owned/'preserve-1.bin').write_bytes(b'owned')
            other=root/'scratch-lowmem-other';other.mkdir()
            (other/'preserve-not-generated.bin').write_bytes(b'keep')
            cleanup_job_scratch(root)
            self.assertFalse(owned.exists())
            self.assertTrue(other.exists())

    def test_compute_persists_overrides_and_changes_only_target_layer(self):
        from latticefill.cad_io import tessellate_shape,save_bodies
        from latticefill.studio_jobs import compute
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
        body=tessellate_shape(BRepPrimAPI_MakeBox(10,10,10).Shape())[0]
        top=next(f['id'] for f in body.face_info if body.selected_mesh([f['id']]).bounds[0,2]>9.99)
        bottom=next(f['id'] for f in body.face_info if body.selected_mesh([f['id']]).bounds[1,2]<.01)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'source';source.mkdir();save_bodies([body],source)
            job=root/'job';job.mkdir()
            cfg=config(preserve_faces=[top,bottom],preserve_thicknesses={str(top):3.},memory_mode='chunked')
            compute(job,dict(action='generate',model_dir=str(source),body=0,config=cfg))
            result=json.loads((job/'design.json').read_text(encoding='utf-8'))
            self.assertEqual(result['config']['preserve_thicknesses'],{str(top):3.})
            self.assertEqual(result['config']['shell'],.8)
            output=trimesh.load(job/'result.stl',force='mesh')
            np.testing.assert_array_equal(output.contains([[1.7,1.7,7.4],[1.7,1.7,1.5]]),[True,False])


if __name__=='__main__':unittest.main()

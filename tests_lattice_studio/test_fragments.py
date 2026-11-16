import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import trimesh
from latticefill.mesh_quality import clean_fragments, safe_display, export_display
from latticefill.surface import chart_uv, face_chart
from latticefill.studio_jobs import validate_design, estimate


class FragmentTests(unittest.TestCase):
    def test_display_preserves_thin_closed_parts_without_fixed_face_target(self):
        rods=[trimesh.creation.cylinder(radius=.15,height=8,sections=32) for _ in range(4)]
        for i,m in enumerate(rods): m.apply_translation([i,0,0])
        mesh=trimesh.util.concatenate(rods)
        with patch.object(trimesh.Trimesh,'simplify_quadric_decimation',side_effect=AssertionError('unsafe decimator')):
            display,quality=safe_display(mesh,.3,.1)
        self.assertTrue(display.is_watertight)
        self.assertTrue(display.is_winding_consistent)
        self.assertEqual(quality['components'],4)
        self.assertLessEqual(quality['volume_error'],.02)
        self.assertLessEqual(quality['tolerance_mm'],.025)

    def test_invalid_surface_is_not_published_as_repaired(self):
        m=trimesh.creation.box()
        m.update_faces(np.arange(11))
        with self.assertRaises(ValueError): safe_display(m,.6,.2)

    def test_large_display_retains_full_mesh(self):
        mesh=trimesh.creation.icosphere(subdivisions=2)
        with patch('latticefill.mesh_quality.DIRECT_DISPLAY_FACE_THRESHOLD',100):
            display,quality=safe_display(mesh,.6,.2)
        self.assertEqual(quality['method'],'original_large')
        self.assertEqual(len(display.faces),len(mesh.faces))
        self.assertTrue(display.is_volume)
        self.assertTrue(quality['topology_preserved'])
        self.assertEqual(quality['tolerance_mm'],0)
        self.assertAlmostEqual(display.volume,mesh.volume)

    def test_only_tiny_numerical_fragments_are_removed(self):
        m=trimesh.creation.box([5,5,5])
        chip=trimesh.creation.icosphere(subdivisions=1,radius=.02)
        chip.apply_translation([8,0,0])
        cleaned,info=clean_fragments(trimesh.util.concatenate([m,chip]),.6)
        self.assertEqual(info['removed_fragments'],1)
        self.assertEqual(info['components'],1)
        self.assertTrue(cleaned.is_volume)
        self.assertAlmostEqual(cleaned.volume,125)

    def test_large_total_fragment_loss_is_rejected(self):
        meshes=[trimesh.creation.box()]
        for i in range(10):
            chip=trimesh.creation.icosphere(subdivisions=1,radius=.1)
            chip.apply_translation([i+2,0,0]); meshes.append(chip)
        with self.assertRaisesRegex(ValueError,'碎片过多'):
            clean_fragments(trimesh.util.concatenate(meshes),.6)

    def test_exported_glb_has_local_coordinates_and_valid_topology(self):
        mesh=trimesh.creation.icosphere(subdivisions=3,radius=5)
        centre=np.array([100000.,200000.,1600.])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'display.glb'
            quality=export_display(mesh,path,.6,.2,centre)
            scene=trimesh.load_scene(path,process=False)
            for part in scene.geometry.values():
                self.assertTrue(part.is_volume)
                self.assertLess(np.abs(part.vertices).max(),6)
            np.testing.assert_allclose(scene.bounds,mesh.bounds+centre,atol=.06)
            self.assertTrue(quality['watertight'])

    def test_spindle_torus_inner_branch_uses_arc_length(self):
        R,r,mid=1145.83333333326,1210.83333333326,9.360352883649819
        chart=dict(kind='torus',origin=[0,0,0],u=[1,0,0],v=[0,1,0],axis=[0,0,1],
                   major_radius=R,minor_radius=r,v_mid=mid,radial_sign=-1,radius=abs(R+r*np.cos(mid)))
        u=np.linspace(.2,1.2,20); v=mid+np.linspace(-.025,.025,20)
        def point(u,v):
            rho=R+r*np.cos(v)
            return np.column_stack((rho*np.cos(u),rho*np.sin(u),r*np.sin(v)))
        actual=chart_uv(point(u,v),chart)
        np.testing.assert_allclose(actual[0],u*chart['radius'],atol=1e-8)
        np.testing.assert_allclose(actual[1],(v-mid)*r,atol=1e-8)
        for du,dv in [(1e-5,0),(0,1e-5)]:
            pts=point(u,v); other=point(u+du,v+dv)
            delta=np.column_stack(chart_uv(other,chart))-np.column_stack(actual)
            scale=np.linalg.norm(delta,axis=1)/np.linalg.norm(other-pts,axis=1)
            self.assertTrue(((scale>.85)&(scale<1.2)).all())

    def test_missing_torus_chart_cannot_silently_flatten(self):
        with self.assertRaisesRegex(ValueError,'环面缺少解析坐标'):
            face_chart(trimesh.creation.icosphere(),'Torus')

    def test_undersampled_preview_is_blocked_but_fine_final_is_allowed(self):
        cfg=validate_design(dict(lattice='kelvin',wall=.55,cell=4,pitch=.2,axis='z',scope='body'))
        bounds=[[0,0,0],[128.57,128.6,149.21]]
        coarse=estimate(bounds,cfg)
        self.assertTrue(coarse['allowed_by_count'])
        self.assertFalse(coarse['allowed'])
        self.assertIn('阻止生成碎片',coarse['blocked_reason'])
        fine=estimate(bounds,cfg|{'quality':'final'})
        self.assertTrue(fine['allowed'])
        self.assertEqual(fine['pitch'],.2)

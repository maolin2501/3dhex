import unittest
from unittest.mock import patch
import numpy as np
from latticefill.cad_io import tessellate_shape
from latticefill.core import FillParams, fill_solid
from latticefill.surface import PerforatedSurface
from latticefill.adaptive_surface import AdaptiveSurface
from latticefill.mesh_quality import clean_fragments
from tests_lattice_studio.test_surface_stitch import split_box


class AdaptiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.body, cls.surface=split_box()
        cls.p=FillParams(lattice='bcc',cell=7.3,wall=1.2,shell=.8,pitch=.2,
            surface_mode='perforated',surface_thickness=.8,surface_cell=6.4,
            surface_width=.7,pore_rounding=.35,junction_blend=.25,edge_width=1,
            seam_style='adaptive',surface_continuity='auto')
        cls.surface.prepare_edges(.2,cls.p)

    def test_no_seam_patch_or_fixed_chart_network(self):
        s=self.surface
        self.assertIsNone(s.stitches)
        self.assertIsNone(s.seam_tree)
        self.assertFalse(self.p.stats['fixed_face_patterns'])
        self.assertGreater(len(s.adaptive.sites),3)

    def test_same_field_on_both_sides_of_cad_label_switch(self):
        s=self.surface;s.pitch=.2;s.grid_order=np.arange(3);s.origin=np.array([3.,3.,9.65])
        shape=(45,45,1);lattice=np.full(shape,-100.);depth=np.full(shape,.35)
        fids=sorted(s.active)
        a=s.combine_nearest(lattice,depth,0,self.p,np.full(shape,fids[0]+1))
        b=s.combine_nearest(lattice,depth,0,self.p,np.full(shape,fids[1]+1))
        np.testing.assert_array_equal(a,b)
        self.assertTrue(np.any(a<-.1));self.assertTrue(np.any(a>.1))

    def test_relabelling_cad_faces_does_not_change_sites_or_pores(self):
        body,_=split_box()
        ids=sorted(self.surface.active)
        body.face_ids[body.face_ids==ids[1]]=ids[0]
        body.face_info=[f for f in body.face_info if f['id']!=ids[1]]
        s=PerforatedSurface(body,[ids[0]],chart_mode='adaptive')
        adaptive=AdaptiveSurface(s,self.p,.2)
        np.testing.assert_array_equal(adaptive.sites,self.surface.adaptive.sites)
        q=np.random.default_rng(14).uniform([0,0,9.2],[16,16,10],(500,3))
        np.testing.assert_array_equal(adaptive.field(q),self.surface.adaptive.field(q))

    def test_anchors_use_actual_shared_web_not_old_hex_charts(self):
        s=self.surface
        for _,target,curve in s.network.terminals:
            self.assertGreater(s.adaptive.support(target),.08)
            self.assertGreaterEqual(s.adaptive.source.depth(curve).min(),-1e-5)
            self.assertLessEqual(np.linalg.norm(np.diff(curve,axis=0),axis=1).max(),.10001)

    def test_reproducible_nonuniform_spacing_and_positive_holes(self):
        s=self.surface.adaptive
        other=AdaptiveSurface(self.surface,self.p,.2)
        np.testing.assert_array_equal(s.sites,other.sites)
        spacing=s.tree.query(s.sites,k=2)[0][:,1]
        self.assertGreater(np.ptp(spacing),.1)
        self.assertLess(s.field(s.sites).max(),-.1)

    def test_memory_guard_precedes_large_surface_sampling(self):
        with patch('latticefill.adaptive_surface.MAX_ADAPTIVE_SITES',1):
            with self.assertRaisesRegex(ValueError,'低内存预算'):
                AdaptiveSurface(self.surface,self.p,.2)

    def test_stl_audit_probes_cross_actual_cad_labels_in_web(self):
        from tests_lattice_studio.adaptive_seam_probes import seam_crossings
        paths,pairs=seam_crossings(self.surface,self.p)
        self.assertGreater(len(paths),0)
        self.assertTrue(all(a!=b for a,b in pairs))
        for _,_,curve in paths:
            self.assertGreater(self.surface.adaptive.field(curve).min(),.17)
            self.assertLessEqual(np.linalg.norm(np.diff(curve,axis=0),axis=1).max(),.11)

    def test_small_result_connected_and_clipped(self):
        body,s=split_box()
        p=FillParams(**(vars(self.p)|dict(stats={},smooth=2,memory_mode='chunked')))
        protected=body.selected_mesh([f['id'] for f in body.face_info if f['id'] not in s.active])
        out=fill_solid(body.mesh,p,preserve_faces=protected,surface_field=s)
        out,quality=clean_fragments(out,.7)
        self.assertTrue(out.is_volume);self.assertEqual(quality['components'],1)
        self.assertTrue(p.stats['source_trimmed'])

    def test_adaptive_mode_needs_no_projection_chart(self):
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeSphere
        body=tessellate_shape(BRepPrimAPI_MakeSphere(10).Shape())[0]
        s=PerforatedSurface(body,[f['id'] for f in body.face_info],chart_mode='adaptive')
        adaptive=AdaptiveSurface(s,self.p,.2)
        self.assertFalse(s.approximate)
        self.assertGreater(len(adaptive.sites),20)


if __name__=='__main__':unittest.main()

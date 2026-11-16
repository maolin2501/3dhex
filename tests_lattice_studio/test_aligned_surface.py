import unittest
from unittest.mock import patch
import numpy as np
from latticefill.core import FillParams,fill_solid
from latticefill.surface import PerforatedSurface
from latticefill.aligned_surface import AlignedSurface
from latticefill.segment_guard import SegmentGuard,segment_distance
from latticefill.mesh_quality import clean_fragments
from tests_lattice_studio.test_surface_stitch import split_box


class AlignedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.body,cls.surface=split_box()
        cls.p=FillParams(lattice='bcc',cell=7.3,wall=1.2,shell=.8,pitch=.2,
            surface_mode='perforated',surface_thickness=.8,surface_cell=6.4,
            surface_width=.7,pore_rounding=.35,junction_blend=.25,edge_width=1,
            seam_style='aligned',surface_continuity='auto')
        cls.surface.prepare_edges(.2,cls.p)

    def test_regular_shared_layout_without_cad_seam_rails(self):
        s=self.surface
        self.assertIsNone(s.stitches);self.assertIsNone(s.seam_tree)
        self.assertEqual(self.p.stats['aligned_layout'],'planar_triangular_rows')
        self.assertFalse(self.p.stats['fixed_face_patterns'])
        self.assertEqual(self.p.stats['seam_rail_length_mm'],0)
        self.assertLess(s.adaptive.field(s.adaptive.sites).max(),-.1)

    def test_relabelling_faces_does_not_change_sites_or_routes(self):
        body,_=split_box();ids=sorted(self.surface.active)
        body.face_ids[body.face_ids==ids[1]]=ids[0]
        s=PerforatedSurface(body,ids,chart_mode='aligned')
        p=FillParams(**(vars(self.p)|dict(stats={})))
        s.prepare_edges(.2,p)
        np.testing.assert_array_equal(s.adaptive.sites,self.surface.adaptive.sites)
        self.assertEqual(len(s.network.paths),len(self.surface.network.paths))
        for a,b in zip(s.network.paths,self.surface.network.paths):np.testing.assert_array_equal(a,b)

    def test_straight_paths_attached_and_inside(self):
        s=self.surface
        self.assertGreater(len(s.network.terminals),0)
        for _,target,path in s.network.terminals:
            self.assertGreater(s.adaptive.support(target),.16)
            self.assertGreater(s.adaptive.source.depth(path).min(),.1)
        for path in s.network.paths:
            direction=path[-1]-path[0];direction/=np.linalg.norm(direction)
            self.assertLess(np.linalg.norm(np.cross(path-path[0],direction),axis=1).max(),1e-10)
            self.assertLessEqual(np.linalg.norm(np.diff(path,axis=0),axis=1).max(),.10001)
        self.assertLessEqual(self.p.stats['straight_angle_deg'][-1],50.)

    def test_shared_phase_fit_never_increases_cost(self):
        self.assertLessEqual(self.p.stats['aligned_phase_cost_after'],self.p.stats['aligned_phase_cost_before']+1e-6)
        self.assertIn(self.p.stats['aligned_phase'][0],(0.,.5))

    def test_budget_and_collapsed_sites(self):
        with patch('latticefill.aligned_surface.MAX_ADAPTIVE_SITES',1):
            with self.assertRaisesRegex(ValueError,'低内存预算'):AlignedSurface(self.surface,self.p,.2)
        other=AlignedSurface(self.surface,self.p,.2)
        with self.assertRaisesRegex(ValueError,'取点'):other._set_sites(np.zeros((1,3)))
        with self.assertRaisesRegex(ValueError,'重合'):other._set_sites(np.zeros((3,3)))

    def test_axis_guided_sphere_is_deterministic(self):
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeSphere
        from latticefill.cad_io import tessellate_shape
        body=tessellate_shape(BRepPrimAPI_MakeSphere(15).Shape())[0]
        s=PerforatedSurface(body,[f['id'] for f in body.face_info],chart_mode='aligned')
        a=AlignedSurface(s,self.p,.2);b=AlignedSurface(s,self.p,.2)
        np.testing.assert_array_equal(a.sites,b.sites)
        self.assertEqual(a.layout,'meridian_ring_rows')
        self.assertGreater(np.ptp(a.row_repeats),0)
        self.assertFalse(s.approximate)

    def test_small_output_connected_and_clipped(self):
        body,s=split_box()
        p=FillParams(**(vars(self.p)|dict(stats={},smooth=2,memory_mode='chunked')))
        protected=body.selected_mesh([f['id'] for f in body.face_info if f['id'] not in s.active])
        out=fill_solid(body.mesh,p,preserve_faces=protected,surface_field=s)
        out,quality=clean_fragments(out,.7)
        self.assertTrue(out.is_volume);self.assertEqual(quality['components'],1)
        self.assertTrue(p.stats['source_trimmed'])

    def test_thin_web_scales_attachment_margin(self):
        body,s=split_box()
        p=FillParams(**(vars(self.p)|dict(stats={},surface_width=.3,pitch=.1)))
        s.prepare_edges(.1,p)
        self.assertAlmostEqual(s.adaptive.material_margin,.069)
        self.assertGreater(len(s.network.terminals),0)
        for _,target,_ in s.network.terminals:
            self.assertGreater(s.adaptive.support(target),s.adaptive.material_margin)


class SegmentTests(unittest.TestCase):
    def test_finite_crossings_parallel_and_skew(self):
        a,b=np.array([0.,0,0]),np.array([10.,0,0])
        self.assertAlmostEqual(segment_distance(a,b,np.array([5.,-5,0]),np.array([5.,5,0]))[0],0)
        self.assertAlmostEqual(segment_distance(a,b,np.array([5.,-5,2]),np.array([5.,5,2]))[0],2)
        self.assertAlmostEqual(segment_distance(a,b,np.array([2.,3,0]),np.array([4.,3,0]))[0],3)
        self.assertAlmostEqual(segment_distance(a,b,np.array([12.,0,0]),np.array([13.,0,0]))[0],2)

    def test_reject_middle_crossing_allow_intended_nodes(self):
        g=SegmentGuard(7.3,1.2)
        g.add(np.array([0.,0,0]),np.array([10.,0,0]))
        self.assertFalse(g.clear(np.array([5.,-5,0]),np.array([5.,5,0])))
        self.assertTrue(g.clear(np.array([0.,0,0]),np.array([5.,5,0])))
        self.assertTrue(g.clear(np.array([5.,-5,3]),np.array([5.,5,3])))


if __name__=='__main__':unittest.main()

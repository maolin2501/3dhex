import unittest
from unittest.mock import patch
import numpy as np
from latticefill.cad_io import tessellate_shape
from latticefill.core import FillParams, fill_solid
from latticefill.surface import PerforatedSurface
from latticefill.surface_stitch import seam_paths, material_runs
from latticefill.studio_jobs import validate_design
from latticefill.mesh_quality import clean_fragments


def split_box():
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    body = tessellate_shape(BRepPrimAPI_MakeBox(16,16,10).Shape())[0]
    top = next(f['id'] for f in body.face_info if body.selected_mesh([f['id']]).bounds[0,2] > 9.99)
    triangles = np.flatnonzero(body.face_ids == top)
    split_id = max(f['id'] for f in body.face_info)+1
    body.face_ids[triangles[1:]] = split_id
    body.face_info.append(dict(id=split_id, type='Plane'))
    surface = PerforatedSurface(body, [top,split_id])
    for fid, angle in ((top,0.), (split_id,.7)):
        surface.charts[fid] = dict(kind='plane', origin=[8,8,10],
            u=[np.cos(angle),np.sin(angle),0], v=[-np.sin(angle),np.cos(angle),0])
    return body, surface


class StitchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.body, cls.surface = split_box()
        cls.p = FillParams(lattice='bcc', cell=7.3, wall=1.2, shell=.8, pitch=.2,
            surface_thickness=.8, surface_cell=6.4, surface_width=.7, pore_rounding=.35,
            junction_blend=.25, edge_width=1, surface_continuity='auto',
            seam_style='stitch', surface_mode='perforated')
        cls.surface.prepare_edges(.2, cls.p)

    def test_no_design_seam_rail_but_real_rims_remain(self):
        s = self.surface
        self.assertIsNone(s.seam_tree)
        self.assertIsNotNone(s.edge_tree)
        self.assertEqual(self.p.stats['seam_rail_length_mm'], 0)
        self.assertGreater(self.p.stats['surface_core_connections'], 0)
        self.assertGreater(s.edge_tree.query([8,8,10])[0], 7.9)

    def test_connections_inside_attached_to_core_and_sampled_continuously(self):
        s = self.surface
        for (anchor,target), curve in zip(s.stitches.anchors, s.stitches.paths):
            self.assertGreater(s.anchor_sampler.support(anchor), .08)
            self.assertGreater(s.network.field(target[None])[0], .59)
            self.assertGreater(s.anchor_sampler.source.depth(curve).min(), 0.)
            self.assertLessEqual(np.linalg.norm(np.diff(curve,axis=0),axis=1).max(), s.stitches.step*1.001)
            self.assertGreater(s.stitches.field(curve).min(), .349)
            self.assertGreater(s.anchor_sampler.source.depth(target[None])[0], 1.39)

    def test_explicit_curve_survives_protected_nearest_label(self):
        s = self.surface; s.pitch=.2; s.grid_order=np.arange(3)
        point=s.stitches.paths[0][len(s.stitches.paths[0])//2]
        s.origin=point
        depth=s.anchor_sampler.source.depth(point[None])[0]
        nearest=np.full((1,1,1),next(fid+1 for fid in s.meshes if fid not in s.active))
        out=s.combine_nearest(np.full((1,1,1),-100.),np.full((1,1,1),depth),0,self.p,nearest)
        self.assertGreater(out.item(),.34)

    def test_chart_holes_not_filled_by_a_continuous_seam_bar(self):
        s=self.surface; s.pitch=.2; s.grid_order=np.arange(3)
        # Fine sample the artificial diagonal seam; some intervals must stay void.
        values=[]
        for t in np.linspace(2,14,121):
            s.origin=np.array([t,t,9.9])
            _,_,fid=s.anchor_sampler.source.closest(s.origin)
            nearest=np.full((1,1,1),fid+1)
            values.append(s.combine_nearest(np.full((1,1,1),-100.),np.full((1,1,1),.1),0,self.p,nearest).item())
        self.assertGreater(np.count_nonzero(np.asarray(values)<-.1),10)

    def test_open_closed_and_branching_seam_walks_visit_edges_once(self):
        vertices=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[2,0,0]])
        edges=np.array([[0,1],[1,2],[2,3],[3,0],[1,4]])
        paths=list(seam_paths(edges,np.tile([2,3],(len(edges),1)),vertices))
        self.assertEqual(sum(len(p)-1 for _,p in paths),len(edges))
        self.assertAlmostEqual(sum(np.linalg.norm(np.diff(p,axis=0),axis=1).sum() for _,p in paths),5.)
        runs=material_runs(np.array([1,1,-1,-1,1]),closed=True)
        self.assertEqual(len(runs),1)
        self.assertEqual(set(runs[0]),{0,1,4})

    def test_stitch_memory_budget(self):
        from latticefill.surface_stitch import SurfaceStitches
        with patch('latticefill.surface_stitch.MAX_STITCH_SAMPLES',1):
            with self.assertRaisesRegex(ValueError,'低内存采样预算'):
                SurfaceStitches(self.surface,self.p,.2)

    def test_default_and_invalid_mode(self):
        cfg=dict(lattice='bcc',axis='z',scope='body')
        self.assertEqual(validate_design(cfg)['seam_style'],'aligned')
        self.assertEqual(validate_design(cfg|dict(seam_style='rail'))['seam_style'],'rail')
        with self.assertRaises(ValueError): validate_design(cfg|dict(seam_style='unknown'))

    def test_generated_stitch_mesh_is_one_closed_component(self):
        body,surface=split_box()
        p=FillParams(**(vars(self.p)|dict(stats={},smooth=2,memory_mode='chunked')))
        protected=body.selected_mesh([f['id'] for f in body.face_info if f['id'] not in surface.active])
        out=fill_solid(body.mesh,p,preserve_faces=protected,surface_field=surface)
        out,quality=clean_fragments(out,.7)
        self.assertEqual(quality['components'],1)
        self.assertTrue(out.is_volume)
        self.assertTrue(p.stats['source_trimmed'])


if __name__ == '__main__': unittest.main()

import unittest
import numpy as np
from latticefill.cad_io import demo_bodies, tessellate_shape
from latticefill.core import FillParams, fill_solid
from latticefill.surface import PerforatedSurface, chart_uv, pattern_field
from latticefill.continuity import MeshProbe, chart_point, chart_scales, web_target
from latticefill.lattices import _bcc_dist
from latticefill.studio_jobs import validate_design
from latticefill.mesh_quality import clean_fragments


class ContinuityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.body = demo_bodies()[0]
        cls.surface = PerforatedSurface(cls.body, [f['id'] for f in cls.body.face_info])
        cls.params = FillParams(lattice='bcc', cell=7.3, wall=1.2, shell=0, pitch=.2,
            surface_thickness=.8, surface_cell=6.4, surface_width=.7, pore_rounding=.35,
            junction_blend=.25, edge_width=1, surface_continuity='auto', surface_mode='perforated')
        cls.surface.prepare_edges(.2, cls.params)

    def test_every_routed_terminal_attaches_to_sheet_material(self):
        surface, p = self.surface, self.params
        probe = MeshProbe(self.body.mesh, self.body.face_ids)
        self.assertGreater(len(surface.network.terminals), 100)
        for root, target, curve in surface.network.terminals:
            self.assertGreaterEqual(probe.depth(curve).min(), -1e-5)
            self.assertLessEqual(np.linalg.norm(np.diff(curve, axis=0), axis=1).max(), surface.network.step*1.01)
            _, _, fid = probe.closest(target)
            support = min(pattern_field(target[None], surface.charts[fid], 'hex', 6.4, .7, .35)[0],
                          p.surface_thickness-probe.depth(target[None])[0])
            rail = max(.7/2, .8*.75)-surface.seam_tree.query(target)[0]
            edge = 1-surface.edge_tree.query(target)[0]
            self.assertGreater(max(support, rail, edge), .025, (root, target, support))
            self.assertGreater(surface.network.field(curve).min(), .59)

    def test_deep_core_matches_analytic_bcc(self):
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
        body = tessellate_shape(BRepPrimAPI_MakeBox(24, 24, 24).Shape())[0]
        surface = PerforatedSurface(body, [f['id'] for f in body.face_info])
        surface.prepare_edges(.2, self.params)
        points = np.random.default_rng(71).uniform(5, 19, (1000, 3))
        expected = self.params.wall/2-_bcc_dist(*points.T, self.params.cell)
        np.testing.assert_allclose(surface.network.field(points), expected, atol=.012)

    def test_web_targets_are_material_for_all_patterns(self):
        chart = dict(kind='plane', origin=[0,0,0], u=[1,0,0], v=[0,1,0])
        points = np.random.default_rng(43).uniform(-15, 15, (100, 2))
        for pattern in ('hex', 'square', 'circle'):
            targets = np.array([web_target(uv, pattern, 6.4) for uv in points])
            phi = pattern_field(np.column_stack((targets, np.zeros(len(targets)))), chart, pattern, 6.4, .7, .35)
            self.assertGreater(phi.min(), .3)

    def test_chart_inverse_including_negative_torus_branch(self):
        charts = [dict(kind='cylinder', origin=[1,2,3], u=[1,0,0], v=[0,1,0], axis=[0,0,1], radius=12)]
        for sign, angle in ((1,.4), (-1, np.pi)):
            charts.append(dict(kind='torus', origin=[1,2,3], u=[1,0,0], v=[0,1,0], axis=[0,0,1],
                radius=abs(3+6*np.cos(angle)), major_radius=3, minor_radius=6,
                radial_sign=sign, v_mid=angle, v_periodic=True))
        for chart in charts:
            scales = chart_scales(chart, 'hex', 4)
            for uv in (np.array([.5,.2]), np.array([-.5,-.2])):
                point = chart_point(uv, chart, scales)
                restored = np.array(chart_uv(point[None], chart)).ravel()*scales
                np.testing.assert_allclose(restored, uv, atol=1e-9)

    def test_auto_mode_validation(self):
        cfg = dict(lattice='bcc', axis='z', scope='body')
        self.assertEqual(validate_design(cfg)['surface_continuity'], 'auto')
        with self.assertRaises(ValueError): validate_design(cfg|{'surface_continuity':'invalid'})

    def test_graph_memory_guards_fail_before_large_allocations(self):
        from unittest.mock import patch
        from latticefill.continuity import BoundaryBCC
        with patch('latticefill.continuity.MAX_GRAPH_CELLS', 1):
            with self.assertRaisesRegex(ValueError, '单元数量过多'):
                BoundaryBCC(self.surface, self.params, .2)

        with patch('latticefill.continuity.MAX_GRAPH_SAMPLES', 10):
            with self.assertRaisesRegex(ValueError, '低内存中心线预算'):
                BoundaryBCC(self.surface, self.params, .2)

    def test_tangent_cad_seam_joins_misphased_sheets(self):
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
        body = tessellate_shape(BRepPrimAPI_MakeBox(8,8,8).Shape())[0]
        top = next(f['id'] for f in body.face_info if body.selected_mesh([f['id']]).bounds[0,2] > 7.99)
        triangles = np.flatnonzero(body.face_ids == top)
        split_id = max(f['id'] for f in body.face_info)+1
        body.face_ids[triangles[1:]] = split_id
        body.face_info.append(dict(id=split_id, type='Plane'))
        surface = PerforatedSurface(body, [top,split_id])
        for fid, angle in ((top,0.), (split_id,.7)):
            surface.charts[fid] = dict(kind='plane', origin=[4,4,8],
                u=[np.cos(angle),np.sin(angle),0], v=[-np.sin(angle),np.cos(angle),0])
        p = FillParams(lattice='cubic', surface_continuity='auto', surface_cell=6.4, surface_width=.7)
        surface.prepare_edges(.2,p); surface.pitch=.2
        surface.grid_order=np.arange(3); surface.origin=np.array([4.,4.,7.8])
        shape=(1,1,1); depth=np.full(shape,.2); lattice=np.full(shape,-10.)
        for fid in (top,split_id):
            nearest=np.full(shape,fid+1)
            joined=surface.combine_nearest(lattice,depth,0,p,nearest)
            self.assertGreater(joined.item(),.2)
        surface.seam_tree=None
        self.assertLess(surface.combine_nearest(lattice,depth,0,p,nearest).item(),-1.)

    def test_small_generation_is_connected_and_clipped(self):
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
        body = tessellate_shape(BRepPrimAPI_MakeBox(12, 12, 8).Shape())[0]
        surface = PerforatedSurface(body, [f['id'] for f in body.face_info])
        p = FillParams(**{**vars(self.params), 'stats':{}, 'smooth':2, 'memory_mode':'chunked'})
        out = fill_solid(body.mesh, p, surface_field=surface)
        out, quality = clean_fragments(out, .7)
        self.assertTrue(out.is_volume)
        self.assertTrue(p.stats['source_trimmed'])
        self.assertEqual(quality['components'], 1)
        self.assertGreater(p.stats['routed_terminals'], 0)
        self.assertTrue(np.all(out.bounds[0] >= body.mesh.bounds[0]-1e-6))
        self.assertTrue(np.all(out.bounds[1] <= body.mesh.bounds[1]+1e-6))


if __name__ == '__main__': unittest.main()

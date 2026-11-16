import unittest
from unittest.mock import patch
import numpy as np
from latticefill.core import FillParams,fill_solid
from latticefill.surface import PerforatedSurface
from latticefill.mesh_quality import clean_fragments
from tests_lattice_studio.test_surface_stitch import split_box


class VertexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.body,cls.surface=split_box()
        cls.p=FillParams(lattice='bcc',cell=7.3,wall=1.2,shell=.8,pitch=.2,
            surface_mode='perforated',surface_thickness=.8,surface_cell=6.4,
            surface_width=.7,pore_rounding=.35,junction_blend=.25,edge_width=1,
            seam_style='vertex',surface_continuity='auto')
        cls.surface.prepare_edges(.2,cls.p)

    def test_true_three_cell_vertices(self):
        a=self.surface.adaptive
        self.assertGreater(len(a.vertices),0)
        for nodes in (a.vertices,a.outer_vertices,a.inner_vertices):
            for point,triple in zip(nodes,a.vertex_triples):
                if not np.isfinite(point).all():continue
                distances=np.linalg.norm(a.sites[triple]-point,axis=1)
                self.assertLess(np.ptp(distances),a.stats['vertex_tolerance_mm'])
                nearest=np.linalg.norm(a.sites-point,axis=1).min()
                self.assertAlmostEqual(nearest,distances.min(),places=4)
        with self.assertRaisesRegex(ValueError,'不是登记'):a.vertex_id(a.sites[0])

    def test_only_registered_vertices_and_straight_segments(self):
        a=self.surface.adaptive;n=self.surface.network
        self.assertGreater(len(n.terminals),0)
        for root,target,curve in n.terminals:
            index=a.vertex_id(target);port=a.inner_vertices[index]
            split=np.linalg.norm(curve-port,axis=1).argmin()
            self.assertTrue(a.core_safe(curve[:split+1]))
            self.assertTrue(a.contact_safe(curve[split:],[target]))
            self.assertLess(np.linalg.norm(np.cross(target-port,a.outer_vertices[index]-port)),1e-5)
        for path in n.paths:
            direction=path[-1]-path[0];direction/=np.linalg.norm(direction)
            self.assertLess(np.linalg.norm(np.cross(path-path[0],direction),axis=1).max(),1e-8)
        self.assertEqual(self.p.stats['nonvertex_terminals'],0)
        self.assertEqual(self.p.stats['vertex_edge_fallbacks'],0)
        self.assertGreater(self.p.stats['removed_old_outer_members'],0)

    def test_old_periodic_shafts_are_removed_not_only_hidden(self):
        n=self.surface.network
        points=np.vstack(n.replaced_paths)
        mask=n.field(points)<-.3
        self.assertGreater(mask.sum(),0)
        result=n.replace_periodic(np.full(mask.sum(),.6),points[mask])
        self.assertLess(result.max(),0)

    def test_no_old_members_touch_design_sheet(self):
        n=self.surface.network;a=self.surface.adaptive
        kept=len(n.paths)-2*len(n.terminals)
        self.assertGreater(kept,0)
        for path in n.paths[:kept]:self.assertTrue(a.contact_safe(path,[]))

    def test_budget_and_unsupported_modes_fail_explicitly(self):
        from latticefill.studio_jobs import validate_design
        from latticefill.vertex_surface import VertexSurface
        config=dict(lattice='bcc',axis='z',scope='body',seam_style='vertex')
        self.assertEqual(validate_design(config)['seam_style'],'vertex')
        for change in (dict(lattice='kelvin'),dict(surface_continuity='off')):
            with self.assertRaisesRegex(ValueError,'BCC'):validate_design(config|change)
            p=FillParams(**(vars(self.p)|change|dict(stats={})))
            with self.assertRaisesRegex(ValueError,'BCC'):fill_solid(self.body.mesh,p,surface_field=self.surface)
        with patch('latticefill.vertex_surface.MAX_VERTEX_TRIPLES',1):
            with self.assertRaisesRegex(ValueError,'低内存预算'):VertexSurface(self.surface,self.p,.2)._triples()

    def test_edge_midpoint_is_not_a_vertex_fallback(self):
        a=self.surface.adaptive
        # A point on a bisector can have positive sheet support without being
        # a triple junction. Registration, not material membership, decides.
        found=False
        for i in range(len(a.vertices)):
            for j in range(i):
                if len(set(a.vertex_triples[i])&set(a.vertex_triples[j]))!=2:continue
                midpoint=(a.vertices[i]+a.vertices[j])/2
                if a.support(midpoint)<=0:continue
                with self.assertRaisesRegex(ValueError,'不是登记'):a.vertex_id(midpoint)
                found=True;break
            if found:break
        self.assertTrue(found)

    def test_graph_budget_checked_while_building_and_empty_faces_rejected(self):
        body,s=split_box();p=FillParams(**(vars(self.p)|dict(stats={})))
        with patch('latticefill.continuity.MAX_GRAPH_SAMPLES',100):
            with self.assertRaisesRegex(ValueError,'低内存中心线预算'):s.prepare_edges(.2,p)
        empty=PerforatedSurface(body,[],chart_mode='vertex')
        with self.assertRaisesRegex(ValueError,'没有可开孔'):empty.prepare_edges(.2,p)

    def test_relabelling_faces_is_deterministic(self):
        body,_=split_box();ids=sorted(self.surface.active)
        body.face_ids[body.face_ids==ids[1]]=ids[0]
        s=PerforatedSurface(body,ids,chart_mode='vertex')
        p=FillParams(**(vars(self.p)|dict(stats={})));s.prepare_edges(.2,p)
        np.testing.assert_array_equal(s.adaptive.vertices,self.surface.adaptive.vertices)
        self.assertEqual(len(s.network.paths),len(self.surface.network.paths))
        for a,b in zip(s.network.paths,self.surface.network.paths):np.testing.assert_array_equal(a,b)

    def test_small_output_connected_and_clipped(self):
        for mode in ('chunked','dense'):
            with self.subTest(mode=mode):
                body,s=split_box()
                p=FillParams(**(vars(self.p)|dict(stats={},smooth=2,memory_mode=mode)))
                protected=body.selected_mesh([f['id'] for f in body.face_info if f['id'] not in s.active])
                out=fill_solid(body.mesh,p,preserve_faces=protected,surface_field=s)
                out,quality=clean_fragments(out,.7)
                self.assertTrue(out.is_volume);self.assertEqual(quality['components'],1)
                self.assertTrue(p.stats['source_trimmed'])


if __name__=='__main__':unittest.main()

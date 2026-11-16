"""Surface layout must cover non-revolution geometry before voxel generation."""
import unittest
import numpy as np
import trimesh
from latticefill.cad_io import CadBody
from latticefill.core import FillParams
from latticefill.surface import PerforatedSurface
from latticefill.aligned_surface import AlignedSurface
from tests_lattice_studio.test_surface_stitch import split_box


def params(pattern='hex'):
    return FillParams(lattice='bcc', cell=7.3, wall=1.2, pitch=.25,
        surface_mode='perforated', surface_thickness=.8, surface_cell=6.4,
        surface_width=1, pore_rounding=.2, junction_blend=.2,
        seam_style='atlas', surface_pattern=pattern, surface_continuity='auto')


def atlas(surface, p):
    # Existing implementation is the regression baseline until the atlas exists.
    try:
        from latticefill.atlas_surface import AtlasSurface
    except ImportError:
        return AlignedSurface(surface, p, p.pitch)
    return AtlasSurface(surface, p, p.pitch)


class AtlasTests(unittest.TestCase):
    def test_square_and_hex_are_different_regular_planar_layouts(self):
        _, s = split_box()
        h = atlas(s, params('hex')); q = atlas(s, params('square'))
        xy = np.array([[x,y,9.6] for x in np.linspace(2,14,35) for y in np.linspace(2,14,35)])
        self.assertFalse(np.allclose(h.field(xy), q.field(xy)))
        # Square centres on an interior row have a 6.4 mm period in both axes.
        d = q.tree.query(q.sites, k=2)[0][:,1]
        self.assertGreater(np.mean(abs(d-6.4)<1e-4), .7)

    def test_ellipsoid_and_flat_ends_have_full_surface_coverage(self):
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=1)
        mesh.vertices *= [35,18,27]
        body = CadBody('ellipsoid', mesh, np.zeros(len(mesh.faces),int), [dict(id=0,type='BSplineSurface')])
        s = PerforatedSurface(body, [0], chart_mode='atlas')
        a = atlas(s, params())
        distance = a.tree.query(mesh.triangles_center)[0]
        self.assertLess(float(np.mean(distance>6.4)), .01)
        self.assertLess(float(np.percentile(distance,95)), 5.5)

    def test_opposite_sheet_neighbours_do_not_make_fake_bisectors(self):
        from latticefill.atlas_surface import OrientedSiteTree
        sites=np.array([[0,0,0],[6,0,0],[0,6,0],[3,3,-1],[9,3,-1],[3,9,-1]],float)
        normals=np.array([[0,0,1]]*3+[[0,0,-1]]*3,float)
        tree=OrientedSiteTree(sites,normals,normal_at=lambda q:np.tile([0,0,1],(len(q),1)))
        _, ids=tree.query([[3,3,-.2]],k=3)
        self.assertTrue(np.all(ids<3))

    def test_cad_face_relabelling_does_not_move_grid(self):
        body,s=split_box(); first=atlas(s,params())
        ids=sorted(s.active); body.face_ids[body.face_ids==ids[1]]=ids[0]
        second=atlas(PerforatedSurface(body,ids,chart_mode='atlas'),params())
        np.testing.assert_allclose(first.sites,second.sites,atol=1e-9)

    def test_report_and_reject_uncovered_layout(self):
        _,s=split_box(); a=atlas(s,params())
        self.assertIn('atlas_coverage_p95_mm',a.stats)
        self.assertLess(a.stats['atlas_uncovered_fraction'],.01)
        a.sites=a.sites[:1]
        from latticefill.atlas_surface import OrientedSiteTree
        a.tree=OrientedSiteTree(a.sites,a.site_normals[:1],normal_at=a.face_probe.normals_at)
        with self.assertRaisesRegex(ValueError,'覆盖'):
            a.check_coverage()

    def test_atlas_routes_only_to_registered_polygon_vertices(self):
        _,s=split_box(); p=params(); s.prepare_edges(p.pitch,p)
        self.assertGreater(len(s.network.terminals),0)
        for _,target,path in s.network.terminals:
            self.assertGreaterEqual(s.adaptive.vertex_id(target),0)
            self.assertGreater(s.adaptive.source.depth(path).min(),0)
        self.assertEqual(p.stats['nonvertex_terminals'],0)

    def test_service_accepts_atlas_and_rejects_unsupported_combinations(self):
        from latticefill.studio_jobs import validate_design
        config=dict(lattice='bcc',axis='z',scope='body',seam_style='atlas')
        self.assertEqual(validate_design(config)['design_version'],12)
        for change in (dict(lattice='kelvin'),dict(surface_continuity='off'),dict(surface_pattern='circle')):
            with self.subTest(change=change),self.assertRaises(ValueError):
                validate_design(config|change)

    def test_small_hex_and_square_output_connected_and_clipped(self):
        from latticefill.core import fill_solid
        from latticefill.mesh_quality import clean_fragments
        for pattern in ('hex','square'):
            with self.subTest(pattern=pattern):
                body,s=split_box();p=params(pattern);p.memory_mode='chunked';p.smooth=2
                protected=body.selected_mesh([f['id'] for f in body.face_info if f['id'] not in s.active])
                out=fill_solid(body.mesh,p,preserve_faces=protected,surface_field=s)
                out,quality=clean_fragments(out,1.)
                self.assertTrue(out.is_volume)
                self.assertEqual(quality['components'],1)
                self.assertTrue(p.stats['source_trimmed'])

    def test_sheet_choice_does_not_switch_before_actual_midwall(self):
        from latticefill.atlas_surface import OrientedSiteTree
        sites=np.array([[0,0,0],[6,0,0],[0,6,0],[3,3,-1],[9,3,-1],[3,9,-1]],float)
        normals=np.array([[0,0,1]]*3+[[0,0,-1]]*3,float)
        tree=OrientedSiteTree(sites,normals,normal_at=lambda q:np.where(q[:,2,None]>-.5,[0,0,1],[0,0,-1]))
        points=np.array([[3,3,-47/130-1e-6],[3,3,-47/130+1e-6]])
        d,ids=tree.query(points,k=3)
        self.assertTrue(np.all(ids<3))
        np.testing.assert_allclose(d[0],d[1],atol=1e-5)

    def test_coverage_does_not_borrow_opposite_sheet_sites(self):
        vertices=np.array([[x,y,0] for x in (1,5) for y in (1,5)]+
                          [[x,y,-1] for x in (-10,10) for y in (-10,10)],float)
        mesh=trimesh.convex.convex_hull(vertices)
        body=CadBody('wedge',mesh,np.zeros(len(mesh.faces),int),[dict(id=0,type='BSplineSurface')])
        a=atlas(PerforatedSurface(body,[0],chart_mode='atlas'),params('square'))
        self.assertGreater(np.sum(a.site_normals[:,2]>.99),0)
        d,ids=a.tree.query(a.coverage_points,normals=a.coverage_normals)
        self.assertTrue(np.all(np.einsum('ij,ij->i',a.coverage_normals,a.site_normals[ids])>.35))

    def test_curved_registered_vertices_are_junctions_of_actual_field(self):
        mesh=trimesh.creation.icosphere(subdivisions=2);mesh.vertices*=[15,6,10]
        body=CadBody('curved',mesh,np.zeros(len(mesh.faces),int),[dict(id=0,type='BSplineSurface')])
        a=atlas(PerforatedSurface(body,[0],chart_mode='atlas'),params());a._build_vertices()
        self.assertGreater(len(a.vertices),0)
        for point,triple in zip(a.vertices,a.vertex_triples):
            d,ids=a.tree.query(point,k=a.neighbours)
            active=set(ids[d<=d.min()+a.stats['vertex_tolerance_mm']])
            self.assertTrue(set(triple).issubset(active),(point,triple,active))

    def test_source_normals_stable_on_surface_and_across_batch_sizes(self):
        from latticefill.continuity import MeshProbe
        mesh=trimesh.creation.icosphere(subdivisions=2);mesh.vertices*=[15,6,10]
        rng=np.random.default_rng(42);ids=rng.integers(len(mesh.faces),size=1000)
        uv=rng.random((1000,2));uv[uv.sum(axis=1)>1]=1-uv[uv.sum(axis=1)>1]
        t=mesh.triangles[ids];points=t[:,0]+uv[:,0,None]*(t[:,1]-t[:,0])+uv[:,1,None]*(t[:,2]-t[:,0])
        probe=MeshProbe(mesh);actual=probe.normals_at(points)
        np.testing.assert_allclose(actual,mesh.face_normals[ids],atol=1e-8)
        np.testing.assert_allclose(actual[:20],probe.normals_at(points[:20]),atol=1e-8)


if __name__=='__main__':unittest.main()

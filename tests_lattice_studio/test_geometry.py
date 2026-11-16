import itertools
import tempfile
import unittest
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from latticefill.cad_io import demo_bodies, save_bodies, read_body, tessellate_shape
from latticefill.core import FillParams, fill_solid
from latticefill.lattices import lattice_phi, _kelvin_dist, LATTICE_TYPES
from latticefill.studio_jobs import validate_design, estimate
from latticefill.surface import PerforatedSurface, pattern_field, chart_uv


class GeometryTests(unittest.TestCase):
    def test_viewer_exposes_all_structures(self):
        from html.parser import HTMLParser
        class Options(HTMLParser):
            in_lattices = False
            values = []
            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                if tag == 'select': self.in_lattices = attrs.get('id') == 'latticeType'
                if tag == 'option' and self.in_lattices: self.values.append(attrs['value'])
            def handle_endtag(self, tag):
                if tag == 'select': self.in_lattices = False
        parser = Options()
        parser.feed((Path(__file__).resolve().parents[1] / 'lattice_studio/index.html').read_text(encoding='utf-8'))
        self.assertEqual(set(parser.values), set(LATTICE_TYPES))
        self.assertEqual(len(parser.values), len(LATTICE_TYPES))

    def test_cad_faces_roundtrip(self):
        body = demo_bodies()[0]
        self.assertEqual(len(body.face_info), 6)
        self.assertTrue(body.mesh.is_watertight)
        self.assertEqual(len(body.face_ids), len(body.mesh.faces))
        selected = body.selected_mesh([0])
        self.assertEqual(len(selected.faces), body.face_info[0]['triangles'])
        with tempfile.TemporaryDirectory() as temp:
            save_bodies([body], Path(temp))
            restored = read_body(Path(temp), 0)
            np.testing.assert_array_equal(body.face_ids, restored.face_ids)
            np.testing.assert_allclose(body.mesh.bounds, restored.mesh.bounds)
        with self.assertRaises(ValueError):
            body.selected_mesh([9999])

    def test_kelvin_matches_explicit_edge_network(self):
        # Truncated octahedron edge length sqrt(2); centre period 4.
        vertices = np.unique([p for signs in itertools.product([-1, 1], repeat=3)
                              for p in itertools.permutations(np.array([0, 1, 2]) * signs)], axis=0)
        edges = [(a,b) for i,a in enumerate(vertices) for b in vertices[i+1:]
                 if abs(np.linalg.norm(a-b)-np.sqrt(2)) < 1e-8]
        self.assertEqual(len(edges), 36)
        rng = np.random.default_rng(482)
        pts = rng.uniform(-2, 2, (150,3))
        expected = np.full(len(pts), np.inf)
        centres = [np.zeros(3)] + [np.array(v)*2 for v in itertools.product([-1,1], repeat=3)]
        centres += [np.array(v) for v in [(4,0,0),(-4,0,0),(0,4,0),(0,-4,0),(0,0,4),(0,0,-4)]]
        for centre in centres:
            for a,b in edges:
                v=b-a
                t=np.clip(((pts-centre-a)*v).sum(axis=1)/(v*v).sum(),0,1)
                expected=np.minimum(expected,np.linalg.norm(pts-centre-a-t[:,None]*v,axis=1))
        actual=_kelvin_dist(*pts.T,4)
        np.testing.assert_allclose(actual,expected,atol=1e-6)
        np.testing.assert_allclose(_kelvin_dist(*(pts+4).T,4),actual,atol=1e-6)
        self.assertAlmostEqual(float(_kelvin_dist(2.,.5,.5,4)),0.)

    def test_all_lattices_small_mesh(self):
        mesh = trimesh.creation.box([8,8,8])
        for kind in LATTICE_TYPES:
            with self.subTest(kind=kind):
                p = FillParams(lattice=kind, cell=4, wall=.9, shell=0, pitch=.3, smooth=0)
                out = fill_solid(mesh, p)
                self.assertGreater(len(out.faces), 100)
                self.assertTrue(out.is_watertight)
                self.assertTrue(0 < p.stats['density'] < .95)

    def test_selected_face_limits_design_region(self):
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
        body = tessellate_shape(BRepPrimAPI_MakeBox(10,10,10).Shape())[0]
        top_id = next(fid for fid in range(6) if body.selected_mesh([fid]).bounds[0,2] > 9.99)
        top=body.selected_mesh([top_id])
        p=FillParams(lattice='cubic',cell=4,wall=1,shell=0,pitch=.25,region_depth=3)
        out=fill_solid(body.mesh,p,design_faces=top)
        self.assertTrue(out.contains([[1.7,1.7,2]])[0])  # original bottom remains solid
        self.assertFalse(out.contains([[1.7,1.7,9]])[0]) # selected top layer is hollow
        self.assertTrue(out.is_watertight)
        p2=FillParams(lattice='cubic',cell=4,wall=1,shell=1,pitch=.25)
        protected=fill_solid(body.mesh,p2,preserve_faces=top)
        self.assertTrue(protected.contains([[1.7,1.7,9.6]])[0])
        self.assertFalse(protected.contains([[1.7,1.7,.3]])[0]) # bottom not accidentally shelled

    def test_unselected_faces_are_perforated_without_changing_outline(self):
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
        body = tessellate_shape(BRepPrimAPI_MakeBox(10,10,10).Shape())[0]
        top_id = next(fid for fid in range(6) if body.selected_mesh([fid]).bounds[0,2] > 9.99)
        top = body.selected_mesh([top_id])
        params = dict(lattice='cubic', cell=4, wall=1, shell=0, pitch=.25,
                      region_depth=3, unselected='perforate')
        out = fill_solid(body.mesh, FillParams(**params), design_faces=top)
        # Bottom is not selected and is outside the design-depth band, but holes
        # must reach it; matching rods still end at the original bottom outline.
        self.assertFalse(out.contains([[1.7,1.7,.15]])[0])
        self.assertTrue(out.contains([[.15,.15,.15]])[0])
        self.assertFalse(out.contains([[.15,.15,-.7]])[0])
        self.assertTrue(out.is_watertight)
        all_body = fill_solid(body.mesh, FillParams(**params))
        np.testing.assert_allclose(out.vertices, all_body.vertices)
        np.testing.assert_array_equal(out.faces, all_body.faces)
        # Explicitly preserved faces take priority over continuous perforation.
        protected = fill_solid(body.mesh, FillParams(**(params | {'shell':1})),
                               preserve_faces=top)
        self.assertTrue(protected.contains([[1.7,1.7,9.6]])[0])
        self.assertFalse(protected.contains([[1.7,1.7,.15]])[0])

    def test_unselected_curved_face_has_holes_and_retains_radius(self):
        body = demo_bodies()[0]
        bottom_id = next(f['id'] for f in body.face_info
                         if np.max(np.abs(body.selected_mesh([f['id']]).vertices[:,2])) < 1e-6)
        selected = body.selected_mesh([bottom_id])
        p = FillParams(lattice='cubic', cell=4, wall=1.2, shell=0, pitch=.3,
                       region_depth=2, unselected='perforate')
        out = fill_solid(body.mesh, p, design_faces=selected)
        angles = np.linspace(.3,3.7,130)
        x, y, z = 23.6*np.cos(angles), 23.6*np.sin(angles), np.full(130,16.)
        phi = lattice_phi('cubic',x,y,z,4,1.2,.3)
        holes = phi < -.4
        material = phi > .4
        self.assertGreater(np.count_nonzero(holes),10)
        self.assertGreater(np.count_nonzero(material),5)
        points = np.column_stack((x,y,z))
        self.assertFalse(out.contains(points[holes]).any())
        self.assertTrue(out.contains(points[material]).all())
        outside = np.column_stack((25*np.cos(angles),25*np.sin(angles),z))
        self.assertFalse(out.contains(outside).any())
        self.assertTrue(out.is_watertight)

    def test_validation_and_memory_guard(self):
        default=dict(lattice='hex3d',cell=4,wall=.6,scope='body',axis='z')
        cfg=validate_design(default)
        self.assertEqual(cfg['unselected'],'perforate')
        self.assertEqual(cfg['design_version'],12)
        self.assertEqual(cfg['surface_mode'],'perforated')
        with self.assertRaises(ValueError):validate_design(default|{'surface_mode':'invalid'})
        with self.assertRaises(ValueError):validate_design(default|{'surface_pattern':'invalid'})
        with self.assertRaises(ValueError):validate_design(default|{'surface_thickness':0})
        self.assertEqual(validate_design(default|{'unselected':'solid'})['unselected'],'solid')
        with self.assertRaises(ValueError):validate_design(default|{'unselected':'invalid'})
        with self.assertRaises(ValueError):validate_design(default|{'wall':float('nan')})
        with self.assertRaises(ValueError):validate_design(default|{'scope':'faces'})
        with self.assertRaises(ValueError):validate_design(default|{'wall':5})
        with self.assertRaises(ValueError):validate_design(default|{'design_faces':[1], 'preserve_faces':[1]})
        e=estimate([[0,0,0],[300,100,350]],cfg)
        self.assertLessEqual(e['voxels'],e['limit'])
        self.assertGreater(e['pitch'],cfg['pitch'])
        cfg['quality']='final'
        self.assertFalse(estimate([[0,0,0],[350,100,350]],cfg)['allowed'])

    def test_perforated_surface_is_a_curved_sheet_not_just_rods(self):
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
        from OCP.gp import gp_Ax2, gp_Pnt, gp_Dir
        ax=gp_Ax2(gp_Pnt(0,0,0),gp_Dir(0,0,1))
        shape=BRepAlgoAPI_Cut(BRepPrimAPI_MakeCylinder(ax,10,12,2.5).Shape(),
                             BRepPrimAPI_MakeCylinder(ax,7,12,2.5).Shape()).Shape()
        body=tessellate_shape(shape,tolerance=.03)[0]
        outer=next(f['id'] for f in body.face_info if f.get('chart',{}).get('radius')==10)
        inner=next(f['id'] for f in body.face_info if f.get('chart',{}).get('radius')==7)
        field=PerforatedSurface(body,{f['id'] for f in body.face_info}-{inner})
        p=FillParams(lattice='kelvin',cell=4,wall=.8,shell=.7,pitch=.2,smooth=2,
                     boundary_blur=.6,unselected='perforate',surface_mode='perforated')
        out=fill_solid(body.mesh,p,preserve_faces=body.selected_mesh([inner]),surface_field=field)
        self.assertTrue(out.is_volume)
        self.assertTrue(p.stats['source_trimmed'])
        theta,z=np.meshgrid(np.linspace(.3,2.2,30),np.linspace(2,10,20))
        points=np.column_stack((9.8*np.cos(theta.ravel()),9.8*np.sin(theta.ravel()),z.ravel()))
        mask=pattern_field(points,field.charts[outer],'hex',4,.8)
        self.assertGreater(np.count_nonzero(mask>.25),30)
        self.assertGreater(np.count_nonzero(mask<-.5),30)
        # Holes are carved through the face band, even where a volume rod would
        # otherwise have met the surface. Webs remain on the original curve.
        self.assertTrue(out.contains(points[mask>.25]).all())
        self.assertFalse(out.contains(points[mask<-.5]).any())
        self.assertTrue(out.contains(points[mask<-.5]*[7.2/9.8,7.2/9.8,1]).all())
        self.assertLessEqual(np.max(np.linalg.norm(out.vertices[:,:2],axis=1)),10.00001)
        # Hundreds of face triangles must coincide with the original surface,
        # not merely stay somewhere inside its bounding box.
        centres=out.triangles_center[::max(1,len(out.faces)//2500)]
        _,dist,_=trimesh.proximity.closest_point(body.mesh,centres)
        self.assertGreater(np.count_nonzero(dist<1e-5),200)

    def test_surface_patterns_and_cylinder_seam(self):
        chart=dict(kind='cylinder',origin=[0,0,0],u=[1,0,0],v=[0,1,0],axis=[0,0,1],radius=10)
        for pattern in ('hex','square','circle'):
            points=np.array([[-10,1e-8,3],[-10,-1e-8,3]])
            phi=pattern_field(points,chart,pattern,4,.6)
            self.assertAlmostEqual(float(phi[0]),float(phi[1]),places=6)

    def test_sheet_patterns_respect_local_selection_and_solid_region(self):
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
        body=tessellate_shape(BRepPrimAPI_MakeBox(8,8,8).Shape())[0]
        top=next(f['id'] for f in body.face_info if body.selected_mesh([f['id']]).bounds[0,2]>7.99)
        field=PerforatedSurface(body,[top])
        volumes=[]
        for pattern in ('hex','square','circle'):
            p=FillParams(lattice='cubic',cell=4,wall=.8,shell=0,pitch=.2,region_depth=2,
                         surface_mode='perforated',surface_pattern=pattern)
            out=fill_solid(body.mesh,p,design_faces=body.selected_mesh([top]),surface_field=field)
            self.assertTrue(out.is_volume)
            self.assertTrue(p.stats['source_trimmed'])
            self.assertTrue(out.contains([[1.5,1.5,1],[5,5,1]]).all())
            x,y=np.meshgrid(np.linspace(.5,7.5,25),np.linspace(.5,7.5,25))
            pts=np.column_stack((x.ravel(),y.ravel(),np.full(x.size,7.9)))
            phi=pattern_field(pts,field.charts[top],pattern,4,.8)
            self.assertFalse(out.contains(pts[phi<-.5]).any())
            self.assertTrue(out.contains(pts[phi>.3]).all())
            volumes.append(out.volume)
        self.assertEqual(len(set(round(v,1) for v in volumes)),3)

    def test_non_volume_source_does_not_claim_exact_trim(self):
        from latticefill.surface import trim_to_source
        box=trimesh.creation.box()
        source=box.copy()
        source.update_faces(np.arange(len(source.faces)-1))
        _,trimmed=trim_to_source(box,source)
        self.assertFalse(trimmed)


if __name__ == '__main__':
    unittest.main()

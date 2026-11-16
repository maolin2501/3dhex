import unittest
import numpy as np
from latticefill.core import FillParams, fill_solid
from latticefill.cad_io import demo_bodies
from latticefill.surface import rounded_hex_field, pattern_field, smooth_union, PerforatedSurface
from latticefill.studio_jobs import validate_design, estimate
from latticefill.mesh_quality import clean_fragments


class ReferenceStyleTests(unittest.TestCase):
    def test_rounded_hex_keeps_flat_web_width_and_period(self):
        x=np.array([0.,2.85,3.2]);y=np.zeros(3)
        phi=rounded_hex_field(x,y,6.4,.7,.35)
        self.assertLess(phi[0],0)
        self.assertAlmostEqual(phi[1],0,places=6)
        self.assertAlmostEqual(phi[2],.35,places=6)
        np.testing.assert_allclose(phi,rounded_hex_field(x+6.4,y,6.4,.7,.35),atol=1e-7)
        y=np.array([(6.4-.7)/np.sqrt(3)])
        self.assertGreater(rounded_hex_field(np.zeros(1),y,6.4,.7,.35)[0],0)

    def test_rounding_keeps_square_pattern_phase(self):
        chart=dict(kind='plane',origin=[0,0,0],u=[1,0,0],v=[0,1,0])
        pts=np.array([[0,0,0],[2,2,0],[.5,2,0]])
        rounded=pattern_field(pts,chart,'square',4,1,.25)
        plain=pattern_field(pts,chart,'square',4,1)
        self.assertTrue((rounded[:2]*plain[:2]>0).all())
        self.assertAlmostEqual(rounded[2],0)

    def test_bounded_join_leaves_distant_fields_unchanged(self):
        a=np.array([-2.,0.,2.]);b=np.array([1.,0.,-2.])
        c=smooth_union(a,b,.2)
        np.testing.assert_allclose(c,[1,.1,2])
        np.testing.assert_array_equal(smooth_union(a,b,0),np.maximum(a,b))

    def test_surface_web_participates_in_sampling_guard(self):
        cfg=validate_design(dict(lattice='bcc',cell=7.3,wall=1.2,axis='z',scope='body',
                                 surface_cell=6.4,surface_width=.7,pitch=.3,quality='final'))
        self.assertFalse(estimate([[0,0,0],[20,20,20]],cfg)['allowed'])
        self.assertTrue(estimate([[0,0,0],[20,20,20]],cfg|{'pitch':.2})['allowed'])
        with self.assertRaises(ValueError):validate_design(cfg|{'surface_width':7})
        with self.assertRaises(ValueError):validate_design(cfg|{'junction_blend':.5})

    def test_reference_style_generates_closed_connected_curved_sheet(self):
        body=demo_bodies()[0]
        field=PerforatedSurface(body,{f['id'] for f in body.face_info})
        p=FillParams(lattice='bcc',cell=7.3,wall=1.2,shell=0,pitch=.2,smooth=2,
            surface_mode='perforated',surface_cell=6.4,surface_width=.7,
            surface_thickness=.8,pore_rounding=.35,junction_blend=.25,edge_width=1)
        out=fill_solid(body.mesh,p,surface_field=field)
        out,quality=clean_fragments(out,.7)
        self.assertTrue(out.is_volume)
        self.assertTrue(p.stats['source_trimmed'])
        self.assertEqual(quality['components'],1)
        self.assertIsNotNone(field.edge_tree)
        self.assertEqual(p.stats['surface_cell'],6.4)
        self.assertEqual(p.stats['surface_width'],.7)
        self.assertTrue(np.all(out.bounds[0]>=body.mesh.bounds[0]-1e-5))
        self.assertTrue(np.all(out.bounds[1]<=body.mesh.bounds[1]+1e-5))


if __name__=='__main__':unittest.main()

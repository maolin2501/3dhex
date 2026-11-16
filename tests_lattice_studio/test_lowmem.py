import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import trimesh
from scipy import ndimage
from latticefill.lowmem import Scratch,grid_spec,rasterize,SolidStencil,SeedField,marching_blocks,cleanup_job_scratch
from latticefill.core import FillParams,fill_solid
from latticefill.cad_io import demo_bodies
from latticefill.surface import PerforatedSurface


class LowMemoryTests(unittest.TestCase):
    def test_batched_connectivity_matches_manifold_components(self):
        from latticefill.mesh_quality import clean_fragments
        outer=trimesh.creation.box([4,4,4]);inner=trimesh.creation.box([2,2,2])
        hollow=trimesh.boolean.difference([outer,inner],engine='manifold')
        chip=trimesh.creation.icosphere(subdivisions=1,radius=.02);chip.apply_translation([8,0,0])
        mesh=trimesh.util.concatenate([hollow,chip])
        expected,info=clean_fragments(mesh,.6)
        with patch('latticefill.mesh_quality.DIRECT_DISPLAY_FACE_THRESHOLD',1):
            actual,stats=clean_fragments(mesh,.6)
        self.assertEqual(stats['components'],info['components'])
        self.assertEqual(stats['removed_fragments'],info['removed_fragments'])
        self.assertAlmostEqual(stats['removed_volume'],info['removed_volume'])
        self.assertAlmostEqual(stats['volume'],56)
        self.assertTrue(actual.is_volume)
        self.assertAlmostEqual(actual.volume,expected.volume)

    def test_low_memory_volume_subtracts_internal_cavities(self):
        from latticefill.mesh_quality import clean_fragments
        outer=trimesh.creation.box([4,4,4]);inner=trimesh.creation.box([2,2,2])
        mesh=trimesh.boolean.difference([outer,inner],engine='manifold')
        out,stats=clean_fragments(mesh,.6)
        self.assertAlmostEqual(stats['volume'],56)
        self.assertAlmostEqual(out.volume,56)

    def test_batched_smoothing_matches_existing_taubin(self):
        from latticefill.mesh_memory import smooth_taubin_batched
        for original in (trimesh.creation.icosphere(subdivisions=2),trimesh.creation.box([9,7,6])):
            expected=original.copy();actual=original.copy()
            trimesh.smoothing.filter_taubin(expected,iterations=5)
            with patch('trimesh.smoothing.laplacian_calculation',side_effect=AssertionError('full adjacency')):
                smooth_taubin_batched(actual,5)
            np.testing.assert_allclose(actual.vertices,expected.vertices,atol=1e-12)
            np.testing.assert_array_equal(actual.faces,expected.faces)

    def test_streamed_stl_preserves_geometry_and_normals(self):
        from latticefill.mesh_memory import export_stl_batched
        mesh=trimesh.creation.icosphere(subdivisions=2)
        shift=np.array([100.,-50.,1618.])
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'result.stl'
            with patch('latticefill.mesh_memory.FACE_BATCH',17):
                export_stl_batched(mesh,p,shift)
            self.assertEqual(p.stat().st_size,84+50*len(mesh.faces))
            actual=trimesh.load(p,force='mesh')
            self.assertTrue(actual.is_volume)
            np.testing.assert_allclose(actual.bounds,mesh.bounds+shift,atol=.0002)
            # STL is float32; the world offset quantizes small local features.
            legacy=mesh.copy();legacy.apply_translation(shift)
            legacy_path=Path(folder)/'legacy.stl';legacy.export(legacy_path)
            expected=trimesh.load(legacy_path,force='mesh')
            self.assertAlmostEqual(actual.volume,expected.volume,places=10)
            self.assertLess(abs(actual.volume/mesh.volume-1),2e-5)

    def test_stencil_classifies_closed_box_and_hollow_shell(self):
        meshes=[trimesh.creation.box([4,6,8]),demo_bodies()[0].mesh]
        for mesh in meshes:
            shape,origin,order=grid_spec(mesh.bounds,.4)
            stencil=SolidStencil(mesh,shape,origin,.4,order)
            actual=np.concatenate([stencil.inside(k,min(k+7,shape[2])) for k in range(0,shape[2],7)],axis=2)
            rng=np.random.default_rng(87);ijk=rng.integers(0,shape,(300,3))
            points=(origin+ijk*.4)[:,np.argsort(order)]
            _,d,_=trimesh.proximity.closest_point(mesh,points)
            valid=d>.05
            np.testing.assert_array_equal(actual[tuple(ijk[valid].T)],mesh.contains(points[valid]))

    def test_extremely_long_triangle_rows_stay_bounded(self):
        from latticefill.lowmem import projected_samples
        count=0
        for u,v in projected_samples([0,0],[1,150000]):
            self.assertLessEqual(len(u),65536)
            count+=len(v)
        self.assertEqual(count,2*150001)

    def test_raster_has_all_six_box_faces_without_subdivision(self):
        mesh=trimesh.creation.box([4,6,8]);shape,origin,order=grid_spec(mesh.bounds,.2)
        seeds=np.zeros(shape,np.uint16)
        with patch('trimesh.remesh.subdivide_to_size',side_effect=AssertionError('whole subdivision')):
            rasterize(mesh,seeds,origin,.2,order,9)
        self.assertGreater(np.count_nonzero(seeds),1000)
        for axis in range(3):
            self.assertTrue(np.any(np.take(seeds,3,axis=axis)))
            self.assertTrue(np.any(np.take(seeds,shape[axis]-4,axis=axis)))

    def test_capped_slab_distances_match_global_edt_and_tree_fallback(self):
        seeds=np.zeros((21,23,27),np.uint16);seeds[4,6,7]=3;seeds[16,18,20]=8
        full=ndimage.distance_transform_edt(seeds==0,sampling=.2)
        for tree in (False,True):
            with patch('latticefill.lowmem.EDT_BLOCK_POINTS',0 if tree else 1000000):
                field=SeedField(seeds,.2)
                for k in range(0,27,4):
                    d,labels=field.query(k,min(k+4,27),1.2,True)
                    np.testing.assert_allclose(d,np.minimum(full[:,:,k:k+4],1.2),atol=1e-6)
                    self.assertTrue(set(np.unique(labels))<={0,3,8})

    def test_xy_tiled_edt_matches_distances_and_cad_labels_without_tree(self):
        seeds=np.zeros((61,59,37),np.uint16)
        seeds[4,6,7]=3;seeds[30,29,18]=6;seeds[55,50,30]=8
        full,indices=ndimage.distance_transform_edt(seeds==0,sampling=.2,return_indices=True)
        expected_labels=seeds[tuple(indices)]
        with patch('latticefill.lowmem.EDT_BLOCK_POINTS',20000):
            field=SeedField(seeds,.2)
            with patch.object(field,'_tree',side_effect=AssertionError('unnecessary KD fallback')):
                for k in range(0,37,5):
                    stop=min(k+5,37)
                    d,labels=field.query(k,stop,1.2,True)
                    expected=np.minimum(full[:,:,k:stop],1.2)
                    np.testing.assert_allclose(d,expected,atol=1e-6)
                    valid=full[:,:,k:stop]<1.2
                    np.testing.assert_array_equal(labels[valid],expected_labels[:,:,k:stop][valid])
                    self.assertFalse(np.any(labels[~valid]))
                    bare,_=field.query(k,stop,1.2,False)
                    np.testing.assert_array_equal(d,bare)

    def test_block_marching_is_closed_and_independent_of_block_size(self):
        xyz=np.meshgrid(*([np.linspace(-5,5,41)]*3),indexing='ij')
        phi=(3-np.sqrt(sum(v*v for v in xyz))).astype(np.float32)
        results=[marching_blocks(phi,.25,np.array([-5.]*3),np.array([1,0,2]),n) for n in (3,7,40)]
        for m in results:
            self.assertTrue(m.is_watertight)
            self.assertTrue(m.is_winding_consistent)
            self.assertEqual(len(m.faces),len(results[-1].faces))
            self.assertAlmostEqual(abs(m.volume),abs(results[-1].volume),places=4)

    def test_scratch_cleanup_even_on_exception(self):
        with tempfile.TemporaryDirectory() as parent:
            with self.assertRaisesRegex(RuntimeError,'test'):
                with Scratch(parent) as scratch:
                    scratch.array('test.bin',(20,20,20),np.float32)[:]=1
                    raise RuntimeError('test')
            self.assertEqual(list(Path(parent).iterdir()),[])

    def test_chunked_volume_is_close_to_dense_and_does_not_subdivide(self):
        mesh=trimesh.creation.box([12,10,8])
        p=FillParams(lattice='bcc',cell=4,wall=1.2,shell=0,pitch=.2,memory_mode='dense')
        dense=fill_solid(mesh,p)
        with patch('trimesh.remesh.subdivide_to_size',side_effect=AssertionError('whole subdivision')):
            q=FillParams(lattice='bcc',cell=4,wall=1.2,shell=0,pitch=.2,memory_mode='chunked')
            result=fill_solid(mesh,q)
        self.assertTrue(result.is_volume)
        self.assertLess(abs(result.volume/dense.volume-1),.12)
        self.assertEqual(q.stats['memory_mode'],'chunked')

    def test_cancel_cleanup_preserves_non_scratch_and_unknown_files(self):
        with tempfile.TemporaryDirectory() as parent:
            root=Path(parent);scratch=root/'scratch-lowmem-test';scratch.mkdir()
            (scratch/'field.bin').write_bytes(b'field')
            other=root/'scratch-lowmem-unrecognized';other.mkdir()
            (other/'user.txt').write_text('keep',encoding='utf-8')
            (root/'result.stl').write_bytes(b'keep')
            cleanup_job_scratch(root)
            self.assertFalse(scratch.exists())
            self.assertTrue((other/'user.txt').exists())
            self.assertTrue((root/'result.stl').exists())

    def test_all_lattice_families_support_chunked_axes(self):
        from latticefill.lattices import LATTICE_TYPES
        mesh=trimesh.creation.box([9,7,6])
        for kind in LATTICE_TYPES:
            with self.subTest(kind=kind):
                p=FillParams(lattice=kind,cell=4,wall=1.2,shell=0,pitch=.3,
                             memory_mode='chunked',axis='y')
                out=fill_solid(mesh,p)
                self.assertTrue(out.is_volume)

    def test_preserved_face_and_local_design_in_chunked_mode(self):
        mesh=trimesh.creation.box([12,10,8])
        top=mesh.submesh([np.flatnonzero(mesh.face_normals[:,2]>.9)],append=True)
        p=FillParams(lattice='bcc',cell=4,wall=1,shell=.8,pitch=.25,
                     memory_mode='chunked',unselected='solid',region_depth=2)
        out=fill_solid(mesh,p,design_faces=top,preserve_faces=top)
        self.assertTrue(out.is_volume)
        self.assertTrue(out.contains([[1.3,.4,-2],[1.3,.4,3.7]]).all())


if __name__=='__main__':unittest.main()

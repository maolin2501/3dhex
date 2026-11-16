"""Independent serialized-file and sampled shape audit of a reduced STL."""
import gc
import json
import sys
from pathlib import Path
import numpy as np
import trimesh
import vtk
from vtk.util.numpy_support import numpy_to_vtk,numpy_to_vtkIdTypeArray,vtk_to_numpy
from .inspect_reference import read_triangles
from .validate_atlas_output import render


def load(path):
    scene=trimesh.load_scene(path,process=False)
    if len(scene.graph.nodes_geometry)!=1:raise ValueError('Unexpected geometry count')
    transform,name=scene.graph[next(iter(scene.graph.nodes_geometry))]
    return scene.geometry[name],transform


def samples(mesh,seed):
    rng=np.random.default_rng(seed)
    indices=rng.integers(0,len(mesh.faces),100000)
    tri=mesh.vertices[mesh.faces[indices]]
    bary=rng.dirichlet([1,1,1],len(indices))
    return np.einsum('ni,nij->nj',bary,tri)


def distances(mesh,query):
    points=vtk.vtkPoints();points.SetData(numpy_to_vtk(np.asarray(mesh.vertices),deep=False))
    cells=vtk.vtkCellArray()
    cells.SetData(numpy_to_vtkIdTypeArray(np.arange(len(mesh.faces)+1,dtype=np.int64)*3,deep=False),
        numpy_to_vtkIdTypeArray(np.asarray(mesh.faces).ravel(),deep=False))
    poly=vtk.vtkPolyData();poly.SetPoints(points);poly.SetPolys(cells)
    distance=vtk.vtkImplicitPolyDataDistance();distance.SetInput(poly)
    query_points=vtk.vtkPoints();query_points.SetData(numpy_to_vtk(np.ascontiguousarray(query),deep=False))
    query_poly=vtk.vtkPolyData();query_poly.SetPoints(query_points)
    evaluate=vtk.vtkSampleImplicitFunctionFilter()
    evaluate.SetInputData(query_poly);evaluate.SetImplicitFunction(distance)
    evaluate.ComputeGradientsOff();evaluate.Update()
    result=np.abs(vtk_to_numpy(evaluate.GetOutput().GetPointData().GetScalars())).astype(float)
    return dict(samples=len(result),max_mm=float(result.max()),p99_mm=float(np.quantile(result,.99)),
                mean_mm=float(result.mean()))


def main():
    folder=Path(sys.argv[1]);report=json.loads((folder/'simplification.json').read_text())
    source=Path(report['source_job'])
    mesh,transform=load(folder/'display.glb')
    original,source_transform=load(source/'display.glb')
    np.testing.assert_allclose(transform,source_transform,atol=1e-10)
    original_samples=samples(original,20261110)
    reduced_samples=samples(mesh,20261111)
    print('Measuring simplified-to-original surface deviation',flush=True)
    reverse=distances(original,reduced_samples)
    del original
    gc.collect()
    print(json.dumps(dict(reduced_to_original=reverse)),flush=True)
    forward=distances(mesh,original_samples)
    gc.collect()
    print(json.dumps(dict(original_to_reduced=forward)),flush=True)
    if max(reverse['max_mm'],forward['max_mm'])>report['tolerance_mm']+.0002:
        raise ValueError('Sampled deviation exceeds the requested tolerance')
    tris=read_triangles(folder/'result.stl')
    if len(tris)!=len(mesh.faces) or len(tris)!=report['after']['faces']:
        raise ValueError('STL/GLB face count mismatch')
    maximum=0.;volume=0.;bounds=np.array([[np.inf]*3,[-np.inf]*3])
    for k in range(0,len(tris),65536):
        actual=tris[k:k+65536].astype(float)
        expected=mesh.vertices[mesh.faces[k:k+65536]]@transform[:3,:3].T+transform[:3,3]
        maximum=max(maximum,float(np.abs(actual-expected).max()))
        if not np.isfinite(actual).all():raise ValueError('Nonfinite STL coordinates')
        bounds[0]=np.minimum(bounds[0],actual.min(axis=(0,1)))
        bounds[1]=np.maximum(bounds[1],actual.max(axis=(0,1)))
        actual-=transform[:3,3]
        volume+=np.einsum('ij,ij->i',actual[:,0],np.cross(actual[:,1],actual[:,2])).sum()/6
    if maximum>1.5e-4:raise ValueError('Serialized STL/GLB discrepancy')
    volume_error=abs(volume-report['after']['volume_mm3'])/report['after']['volume_mm3']
    if volume_error>1e-4:raise ValueError('STL signed volume mismatch')
    result=dict(reduced_to_original=reverse,original_to_reduced=forward,
                stl_glb_max_difference_mm=maximum,stl_signed_volume_mm3=volume,
                stl_volume_relative_error=volume_error,bounds=bounds.tolist(),
                note='Bidirectional random surface sampling, not an exhaustive Hausdorff proof.')
    (folder/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)
    del tris
    render(mesh,folder)


if __name__=='__main__':main()

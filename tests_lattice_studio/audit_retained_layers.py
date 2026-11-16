"""Sample actual saved output inside the four requested source face layers."""
import json
import sys
from pathlib import Path
import numpy as np
import trimesh
import vtk
from vtk.util.numpy_support import numpy_to_vtk,numpy_to_vtkIdTypeArray
from latticefill.continuity import MeshProbe


def main():
    job=Path(sys.argv[1]);request=json.loads((job/'request.json').read_text(encoding='utf-8'))
    design=json.loads((job/'design.json').read_text(encoding='utf-8'))
    expected={'0':3.,'6':3.,'7':3.,'13':3.}
    if design['config']['preserve_thicknesses']!=expected or design['config']['shell']!=.8:
        raise ValueError('Unexpected retained-face thickness configuration')
    source=np.load(Path(request['model_dir'])/'body_0.npz')
    original=trimesh.Trimesh(source['vertices'],source['faces'],process=False)
    centre=original.bounds.mean(axis=0);original.apply_translation(-centre)
    probe=MeshProbe(original);rng=np.random.default_rng(20261110)
    batches=[];metadata=[]
    for fid in (0,6,7,13):
        indices=np.flatnonzero(source['face_ids']==fid)
        weights=original.area_faces[indices];weights/=weights.sum()
        choices=rng.choice(indices,1000,p=weights)
        bary=rng.dirichlet([1,1,1],len(choices))
        points=np.einsum('ni,nij->nj',bary,original.triangles[choices])
        normals=original.face_normals[choices]
        for depth in (.4,1.5,2.6,3.5):
            inward=points-normals*depth
            source_depth=probe.depth(inward)
            # These source patches are about 3 mm thick: the 3.5 mm probe
            # is outside the original solid and verifies that it is not expanded.
            valid=(source_depth>.3) if depth<=3 else (source_depth<-.3)
            if valid.sum()<100:raise ValueError(f'Face {fid}, depth {depth}: insufficient source probes')
            batches.append(inward[valid]);metadata.append((fid,depth,int(valid.sum())))
    del probe,original,source
    display_path=Path(sys.argv[3]) if len(sys.argv)>3 else job/'display.glb'
    scene=trimesh.load_scene(display_path,process=False)
    transform,name=scene.graph[next(iter(scene.graph.nodes_geometry))]
    mesh=scene.geometry[name]
    world_from_local=np.asarray(transform);local_from_source=np.linalg.inv(world_from_local)
    points=vtk.vtkPoints();points.SetData(numpy_to_vtk(np.ascontiguousarray(mesh.vertices),deep=False))
    cells=vtk.vtkCellArray();cells.SetData(numpy_to_vtkIdTypeArray(np.arange(len(mesh.faces)+1,dtype=np.int64)*3,deep=True),
        numpy_to_vtkIdTypeArray(np.asarray(mesh.faces).ravel(),deep=False))
    poly=vtk.vtkPolyData();poly.SetPoints(points);poly.SetPolys(cells)
    distance=vtk.vtkImplicitPolyDataDistance();distance.SetInput(poly)
    results=[]
    for batch,(fid,depth,count) in zip(batches,metadata):
        world=batch+centre
        local=world@local_from_source[:3,:3].T+local_from_source[:3,3]
        signed=np.array([-distance.EvaluateFunction(point) for point in local])
        fraction=float(np.mean(signed>0))
        item=dict(face_id=fid,label=f'面 {fid+1:03d}',depth_mm=depth,source_samples=count,
                  source_probe_side='inside' if depth<=3 else 'outside',
                  output_material_fraction=fraction,min_output_depth_mm=float(signed.min()))
        results.append(item);print(json.dumps(item,ensure_ascii=False),flush=True)
        if depth<=2.6 and fraction<.995:raise ValueError(f'Face {fid}: requested 3 mm layer has unexpected voids')
        if depth>3 and fraction>.005:raise ValueError(f'Face {fid}: material extends outside the source')
    output=Path(sys.argv[2]);output.mkdir(parents=True,exist_ok=True)
    (output/'retained_layer_audit.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()

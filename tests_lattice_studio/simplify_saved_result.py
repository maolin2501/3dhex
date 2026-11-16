"""One-off, separately saved tolerance simplification of a verified result."""
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import manifold3d as manifold
import numpy as np
import trimesh

from latticefill.mesh_memory import export_stl_batched
from latticefill.mesh_quality import as_manifold, as_trimesh


def describe(solid):
    return dict(faces=solid.num_tri(), vertices=solid.num_vert(),
                euler=solid.num_vert()-solid.num_edge()+solid.num_tri(),
                volume_mm3=solid.volume(), bounds=list(solid.bounding_box()))


def topology_reduce(mesh,tolerance):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk,numpy_to_vtkIdTypeArray,vtk_to_numpy
    points=vtk.vtkPoints()
    points.SetData(numpy_to_vtk(np.asarray(mesh.vertices),deep=False))
    cells=vtk.vtkCellArray()
    cells.SetData(numpy_to_vtkIdTypeArray(np.arange(len(mesh.faces)+1,dtype=np.int64)*3,deep=False),
                  numpy_to_vtkIdTypeArray(np.asarray(mesh.faces).ravel(),deep=False))
    poly=vtk.vtkPolyData();poly.SetPoints(points);poly.SetPolys(cells)
    reduce=vtk.vtkDecimatePro();reduce.SetInputData(poly)
    reduce.SetTargetReduction(.9)
    reduce.PreserveTopologyOn();reduce.SplittingOff();reduce.PreSplitMeshOff()
    reduce.BoundaryVertexDeletionOff();reduce.AccumulateErrorOn()
    reduce.SetErrorIsAbsolute(1);reduce.SetAbsoluteError(tolerance)
    reduce.SetOutputPointsPrecision(vtk.vtkAlgorithm.DOUBLE_PRECISION)
    last=[-1]
    def progress(obj,event):
        step=int(obj.GetProgress()*20)
        if step>last[0]:
            print(f'Topology-preserving reduction {step*5}%',flush=True);last[0]=step
    reduce.AddObserver(vtk.vtkCommand.ProgressEvent,progress)
    reduce.Update()
    output=reduce.GetOutput()
    return trimesh.Trimesh(np.array(vtk_to_numpy(output.GetPoints().GetData())),
        np.array(vtk_to_numpy(output.GetPolys().GetConnectivityArray())).reshape(-1,3),process=False)


def main():
    job=Path(sys.argv[1]).resolve()
    destination=Path(sys.argv[2]).resolve()
    tolerance=float(sys.argv[3])
    method=sys.argv[4] if len(sys.argv)>4 else 'manifold'
    if method not in ('manifold','vtk'):raise ValueError('Unknown method')
    if not np.isfinite(tolerance) or not 0<tolerance<=.05:
        raise ValueError('This conservative export requires 0 < tolerance <= 0.05 mm')
    if destination.exists():
        raise FileExistsError('Use a new output directory; existing results are never overwritten')
    design=json.loads((job/'design.json').read_text(encoding='utf-8'))
    status=json.loads((job/'status.json').read_text(encoding='utf-8'))
    if status['state']!='done':raise ValueError('Source generation is not complete')
    started=time.monotonic()
    print('Loading verified indexed geometry corresponding to the original STL',flush=True)
    scene=trimesh.load_scene(job/'display.glb',process=False)
    if len(scene.graph.nodes_geometry)!=1:raise ValueError('Expected one indexed geometry')
    transform,name=scene.graph[next(iter(scene.graph.nodes_geometry))]
    if not np.allclose(transform[:3,:3],np.eye(3)):
        raise ValueError('Expected translation-only placement')
    mesh=scene.geometry[name]
    if len(mesh.faces)!=design['stats']['faces']:
        raise ValueError('Source STL/display mesh counts disagree')
    with (job/'result.stl').open('rb') as stream:
        header=stream.read(84)
        if int.from_bytes(header[80:84],'little')!=len(mesh.faces):
            raise ValueError('Source STL header disagrees')
        stream.seek(0);source_hash=hashlib.file_digest(stream,'sha256').hexdigest()
    solid=as_manifold(mesh)
    before=describe(solid)
    del scene,mesh
    gc.collect()
    print(json.dumps(dict(stage='simplifying',tolerance_mm=tolerance,before=before)),flush=True)
    if method=='vtk':
        normalized=as_trimesh(solid)
        del solid
        gc.collect()
        reduced=topology_reduce(normalized,tolerance)
        del normalized
        gc.collect()
        simplified=as_manifold(reduced)
        del reduced
    else:
        simplified=solid.simplify(tolerance)
        del solid
    if simplified.status()!=manifold.Error.NoError or simplified.is_empty():
        raise ValueError('Simplification did not produce a valid closed manifold')
    after=describe(simplified)
    relative_volume_error=abs(after['volume_mm3']-before['volume_mm3'])/before['volume_mm3']
    print(json.dumps(dict(stage='candidate',after=after,
                         relative_volume_error=relative_volume_error)),flush=True)
    if after['faces']>=before['faces']:raise ValueError('No reduction was achieved')
    if after['euler']!=before['euler']:raise ValueError('Topology changed')
    if relative_volume_error>.005:raise ValueError('Volume change exceeds 0.5%')
    gc.collect()
    print(json.dumps(dict(stage='checking_components',after=after,
                         relative_volume_error=relative_volume_error)),flush=True)
    parts=simplified.decompose()
    if len(parts)!=design['stats']['components']:
        raise ValueError('Connected component count changed')
    components=[dict(volume_mm3=p.volume(),faces=p.num_tri(),genus=p.genus()) for p in parts]
    del parts
    reduced=as_trimesh(simplified)
    del simplified
    gc.collect()
    destination.mkdir(parents=True,exist_ok=False)
    print('Exporting simplified STL and indexed preview',flush=True)
    export_stl_batched(reduced,destination/'result.stl',translation=transform[:3,3])
    preview=trimesh.Scene()
    preview.add_geometry(reduced,transform=transform)
    preview.export(destination/'display.glb')
    del preview,reduced
    gc.collect()
    # Validate actual serialized connectivity, not merely the in-memory result.
    saved=trimesh.load_scene(destination/'display.glb',process=False)
    saved_mesh=next(iter(saved.geometry.values()))
    check=as_manifold(saved_mesh)
    if check.num_tri()!=after['faces'] or check.num_vert()-check.num_edge()+check.num_tri()!=after['euler']:
        raise ValueError('Serialized topology changed')
    del check,saved,saved_mesh
    gc.collect()
    with (destination/'result.stl').open('rb') as stream:
        final_hash=hashlib.file_digest(stream,'sha256').hexdigest()
    report=dict(source_job=str(job),source_stl_sha256=source_hash,
                method='vtk_preserve_topology' if method=='vtk' else 'manifold_tolerance',tolerance_mm=tolerance,
                before=before,after=after,components=components,
                relative_volume_error=relative_volume_error,
                face_reduction_fraction=1-after['faces']/before['faces'],
                source_bytes=(job/'result.stl').stat().st_size,
                simplified_bytes=(destination/'result.stl').stat().st_size,
                sha256=final_hash,translation=transform[:3,3].tolist(),
                elapsed_seconds=time.monotonic()-started,
                retained_face_settings=design['config'].get('preserve_thicknesses',{}),
                note='Original isolated components are preserved; not repaired or removed.')
    (destination/'simplification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()

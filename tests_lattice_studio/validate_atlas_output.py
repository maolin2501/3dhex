"""Bounded saved-STL/GLB agreement audit and indexed offscreen final views."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import trimesh
from .inspect_reference import read_triangles


def main():
    job=Path(sys.argv[1]).resolve()
    status=json.loads((job/'status.json').read_text(encoding='utf-8'))
    if status['state']!='done':raise ValueError('Full generation is not complete')
    design=json.loads((job/'design.json').read_text(encoding='utf-8'))
    scene=trimesh.load_scene(job/'display.glb',process=False)
    if len(scene.graph.nodes_geometry)!=1:raise ValueError('Expected one output geometry')
    transform,name=scene.graph[next(iter(scene.graph.nodes_geometry))]
    mesh=scene.geometry[name]
    triangles=read_triangles(job/'result.stl')
    if len(triangles)!=len(mesh.faces) or len(triangles)!=design['stats']['faces']:
        raise ValueError('Saved face counts disagree')
    bounds=np.array([[np.inf]*3,[-np.inf]*3]);volume=0.;max_error=0.
    centre=np.asarray(design['bounds']).mean(axis=0)
    for i in range(0,len(triangles),65536):
        actual=triangles[i:i+65536].astype(float)
        expected=mesh.vertices[mesh.faces[i:i+65536]]@transform[:3,:3].T+transform[:3,3]
        max_error=max(max_error,float(np.max(np.abs(actual-expected))))
        if not np.isfinite(actual).all():raise ValueError('Nonfinite STL coordinates')
        bounds[0]=np.minimum(bounds[0],actual.min(axis=(0,1)))
        bounds[1]=np.maximum(bounds[1],actual.max(axis=(0,1)))
        actual-=centre
        volume+=np.einsum('ij,ij->i',actual[:,0],np.cross(actual[:,1],actual[:,2])).sum()/6
    if max_error>1.5e-4:raise ValueError(f'STL/GLB discrepancy {max_error}')
    relative=abs(volume-design['stats']['volume'])/design['stats']['volume']
    if relative>1e-4:raise ValueError(f'STL volume discrepancy {relative}')
    with (job/'result.stl').open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
    audit=dict(faces=len(triangles),bytes=(job/'result.stl').stat().st_size,
        max_stl_glb_coordinate_error_mm=max_error,stl_signed_volume_mm3=float(volume),
        relative_volume_error=float(relative),bounds=bounds.tolist(),sha256=sha,
        source_trimmed=design['stats']['source_trimmed'],components=design['stats']['components'],
        validated_serialized_topology=design['display_quality']['watertight'])
    output=Path(sys.argv[2]) if len(sys.argv)>2 else Path('reference_meshes/new_model/atlas_20261110')
    output.mkdir(exist_ok=True,parents=True)
    (output/'final_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit),flush=True)
    del triangles,scene
    render(mesh,output)


def render(mesh,output):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk,numpy_to_vtkIdTypeArray
    vertices=np.ascontiguousarray(mesh.vertices,dtype=np.float32)
    faces=mesh.faces.view(np.ndarray)
    points=vtk.vtkPoints();points.SetData(numpy_to_vtk(vertices,deep=False))
    cells=vtk.vtkCellArray()
    cells.SetData(numpy_to_vtkIdTypeArray(np.arange(len(faces)+1,dtype=np.int64)*3,deep=False),
                  numpy_to_vtkIdTypeArray(faces.ravel(),deep=False))
    poly=vtk.vtkPolyData();poly.SetPoints(points);poly.SetPolys(cells)
    mapper=vtk.vtkPolyDataMapper();mapper.SetInputData(poly)
    actor=vtk.vtkActor();actor.SetMapper(mapper)
    actor.GetProperty().SetColor(.29,.56,.50);actor.GetProperty().SetAmbient(.25)
    ren=vtk.vtkRenderer();ren.SetBackground(.95,.97,.96);ren.AddActor(actor)
    window=vtk.vtkRenderWindow();window.SetOffScreenRendering(1);window.SetSize(1300,1150);window.AddRenderer(ren)
    camera=ren.GetActiveCamera();camera.SetViewUp(0,0,1);camera.ParallelProjectionOn()
    for name,direction in [('final_front',[0,-1,.2]),('final_outer',[-1,1.8,1.1]),('final_side',[2,1,.4])]:
        camera.SetFocalPoint(0,0,0);camera.SetPosition(*np.asarray(direction)*300)
        ren.ResetCamera();ren.ResetCameraClippingRange();window.Render()
        capture=vtk.vtkWindowToImageFilter();capture.SetInput(window);capture.Update()
        writer=vtk.vtkPNGWriter();writer.SetFileName(str(output/f'{name}.png'))
        writer.SetInputConnection(capture.GetOutputPort());writer.Write()
        print('rendered',name,flush=True)
    window.Finalize()


if __name__=='__main__':main()

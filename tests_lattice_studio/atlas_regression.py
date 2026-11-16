"""Small surface-only diagnostic for the user-selected model, no volume job."""
import json
from pathlib import Path
import time
import numpy as np
import trimesh
from latticefill.cad_io import read_body
from latticefill.core import FillParams
from latticefill.surface import PerforatedSurface
from latticefill.atlas_surface import AtlasSurface

ROOT=Path(__file__).resolve().parent.parent
SOURCE=ROOT/'.lattice-studio/3ba81f8b946509f94ec80dcb'
OUT=ROOT/'reference_meshes/new_model/atlas_20261110'


def load():
    data=json.loads((SOURCE/'design.json').read_text(encoding='utf-8'));cfg=data['config']|{'seam_style':'atlas'}
    body=read_body(ROOT/'.lattice-studio'/data['model_id'],0)
    active=set(f['id'] for f in body.face_info)-set(cfg['preserve_faces'])
    surface=PerforatedSurface(body,active,translation=-body.mesh.bounds.mean(axis=0),chart_mode='atlas')
    p=FillParams(**{k:v for k,v in cfg.items() if k in FillParams.__dataclass_fields__})
    return body,surface,p


def main():
    started=time.perf_counter();body,surface,p=load()
    a=AtlasSurface(surface,p,p.pitch)
    print(json.dumps(a.stats,ensure_ascii=False),flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'coverage.json').write_text(json.dumps(a.stats,ensure_ascii=False,indent=2),encoding='utf-8')
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk,numpy_to_vtkIdTypeArray
    # Refinement only for this diagnostic drawing; not the output algorithm.
    vertices,faces=trimesh.remesh.subdivide_to_size(a.mesh.vertices,a.mesh.faces,max_edge=.9,max_iter=12)
    print('surface view samples',len(vertices),'triangles',len(faces),flush=True)
    phi=np.empty(len(vertices),np.float32)
    for i in range(0,len(vertices),10000):phi[i:i+10000]=a.field(vertices[i:i+10000])
    points=vtk.vtkPoints();points.SetData(numpy_to_vtk(np.ascontiguousarray(vertices),deep=False))
    cells=vtk.vtkCellArray();cells.SetData(numpy_to_vtkIdTypeArray(np.arange(len(faces)+1,dtype=np.int64)*3,deep=True),numpy_to_vtkIdTypeArray(faces.ravel().astype(np.int64),deep=True))
    poly=vtk.vtkPolyData();poly.SetPoints(points);poly.SetPolys(cells);poly.GetPointData().SetScalars(numpy_to_vtk(phi,deep=True))
    clip=vtk.vtkClipPolyData();clip.SetInputData(poly);clip.SetValue(0);clip.Update()
    mapper=vtk.vtkPolyDataMapper();mapper.SetInputConnection(clip.GetOutputPort());mapper.ScalarVisibilityOff()
    actor=vtk.vtkActor();actor.SetMapper(mapper);actor.GetProperty().SetColor(.29,.58,.51);actor.GetProperty().SetAmbient(.3)
    ren=vtk.vtkRenderer();ren.SetBackground(.95,.97,.96);ren.AddActor(actor)
    window=vtk.vtkRenderWindow();window.SetOffScreenRendering(1);window.SetSize(1100,1000);window.AddRenderer(ren)
    camera=ren.GetActiveCamera();camera.SetViewUp(0,0,1);camera.ParallelProjectionOn()
    for name,direction in [('outer',[-1,1.8,1.1]),('inner',[1,-1.8,1.1]),('side',[2,1,.4])]:
        camera.SetFocalPoint(0,0,0);camera.SetPosition(*np.asarray(direction)*250);ren.ResetCamera();ren.ResetCameraClippingRange();window.Render()
        capture=vtk.vtkWindowToImageFilter();capture.SetInput(window);capture.Update()
        writer=vtk.vtkPNGWriter();writer.SetFileName(str(OUT/f'{name}.png'));writer.SetInputConnection(capture.GetOutputPort());writer.Write()
    window.Finalize();print('elapsed',time.perf_counter()-started,flush=True)


if __name__=='__main__':main()

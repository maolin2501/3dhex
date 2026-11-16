"""Read-only STL analysis and offscreen geometry renders for comparison."""
import argparse
import json
from pathlib import Path
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray


def read_triangles(path):
    dtype=np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attr','<u2')])
    n=int(np.fromfile(path,dtype='<u4',count=1,offset=80)[0])
    if path.stat().st_size!=84+n*50:
        raise ValueError('Expected binary STL')
    return np.memmap(path,dtype=dtype,mode='r',offset=84,shape=(n,))['vertices']


def render(tris, output, direction, focal=None, scale=None):
    pts=np.ascontiguousarray(tris.reshape(-1,3))
    points=vtk.vtkPoints(); points.SetData(numpy_to_vtk(pts,deep=False))
    cells=vtk.vtkCellArray()
    cells.SetData(numpy_to_vtkIdTypeArray(np.arange(0,len(pts)+1,3,dtype=np.int64),deep=True),
                  numpy_to_vtkIdTypeArray(np.arange(len(pts),dtype=np.int64),deep=True))
    poly=vtk.vtkPolyData();poly.SetPoints(points);poly.SetPolys(cells)
    mapper=vtk.vtkPolyDataMapper();mapper.SetInputData(poly)
    actor=vtk.vtkActor();actor.SetMapper(mapper)
    actor.GetProperty().SetColor(.29,.53,.49)
    actor.GetProperty().SetAmbient(.22);actor.GetProperty().SetDiffuse(.78)
    ren=vtk.vtkRenderer();ren.SetBackground(.95,.96,.96);ren.AddActor(actor)
    window=vtk.vtkRenderWindow();window.SetOffScreenRendering(1);window.SetSize(1200,1000);window.AddRenderer(ren)
    camera=ren.GetActiveCamera()
    centre=pts.mean(axis=0) if focal is None else np.asarray(focal)
    camera.SetFocalPoint(*centre);camera.SetPosition(*(centre+np.asarray(direction)*100))
    camera.SetViewUp(0,0,1);camera.ParallelProjectionOn();ren.ResetCamera()
    if focal is not None:camera.SetFocalPoint(*focal);camera.SetPosition(*(np.asarray(focal)+np.asarray(direction)*100))
    if scale is not None:camera.SetParallelScale(scale)
    ren.ResetCameraClippingRange();window.Render()
    capture=vtk.vtkWindowToImageFilter();capture.SetInput(window);capture.Update()
    writer=vtk.vtkPNGWriter();writer.SetFileName(str(output));writer.SetInputConnection(capture.GetOutputPort());writer.Write()
    window.Finalize()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('path',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--crop',type=float,nargs=4,metavar=('X','Y','Z','HALF'))
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    tris=read_triangles(args.path)
    if args.crop:
        centre=np.array(args.crop[:3]);half=args.crop[3]
        mid=tris.mean(axis=1)
        tris=tris[np.all(abs(mid-centre)<half,axis=1)]
    low=tris.min(axis=(0,1));high=tris.max(axis=(0,1))
    volume=0.;area=0.
    for a in range(0,len(tris),100000):
        t=tris[a:a+100000].astype(float)-(low+high)/2
        cross=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]);area+=np.linalg.norm(cross,axis=1).sum()/2
        volume+=np.einsum('ij,ij->i',t[:,0],np.cross(t[:,1],t[:,2])).sum()/6
    stats=dict(triangles=len(tris),bounds=[low.tolist(),high.tolist()],extent=(high-low).tolist(),area=area,signed_volume=volume)
    print(json.dumps(stats),flush=True)
    (args.out/'metrics.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
    for name,direction in [('outer',[-1,-.4,.2]),('inner',[1,.4,.3]),('front',[0,-1,0])]:
        render(tris,args.out/f'{name}.png',direction)
        print('rendered',name,flush=True)


if __name__=='__main__':main()

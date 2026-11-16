"""CAD-face identity check for the four front thickness overrides."""
from pathlib import Path
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk,numpy_to_vtkIdTypeArray


def main():
    data=np.load('.lattice-studio/dc0c304ea2030b1ece6cffc4/body_0.npz')
    vertices=data['vertices'];centre=(vertices.min(0)+vertices.max(0))/2;vertices-=centre
    faces=data['faces'];ids=data['face_ids']
    points=vtk.vtkPoints();points.SetData(numpy_to_vtk(vertices,deep=True))
    cells=vtk.vtkCellArray();cells.SetData(numpy_to_vtkIdTypeArray(np.arange(len(faces)+1,dtype=np.int64)*3,deep=True),
        numpy_to_vtkIdTypeArray(faces.ravel(),deep=True))
    poly=vtk.vtkPolyData();poly.SetPoints(points);poly.SetPolys(cells)
    colors=np.tile(np.array([178,198,191],np.uint8),(len(faces),1))
    colors[np.isin(ids,[0,6,7,13])]=[228,171,70]
    poly.GetCellData().SetScalars(numpy_to_vtk(colors,deep=True))
    mapper=vtk.vtkPolyDataMapper();mapper.SetInputData(poly)
    actor=vtk.vtkActor();actor.SetMapper(mapper)
    renderer=vtk.vtkRenderer();renderer.SetBackground(.95,.97,.96);renderer.AddActor(actor)
    for fid in (0,6,7,13):
        triangles=vertices[faces[ids==fid]];p=triangles.mean(axis=(0,1));p[1]-=12
        text=vtk.vtkBillboardTextActor3D();text.SetInput(f'{fid+1:03d}  3 mm');text.SetPosition(*p)
        text.GetTextProperty().SetFontSize(23);text.GetTextProperty().SetColor(.10,.20,.18)
        text.GetTextProperty().SetBold(True);renderer.AddActor(text)
    window=vtk.vtkRenderWindow();window.SetOffScreenRendering(1);window.SetSize(1000,1000);window.AddRenderer(renderer)
    camera=renderer.GetActiveCamera();camera.SetViewUp(0,0,1);camera.SetPosition(0,-400,0);camera.SetFocalPoint(0,0,0)
    camera.ParallelProjectionOn();renderer.ResetCamera();window.Render()
    output=Path('reference_meshes/new_model/front_3mm_20261110');output.mkdir(parents=True,exist_ok=True)
    capture=vtk.vtkWindowToImageFilter();capture.SetInput(window);capture.Update()
    writer=vtk.vtkPNGWriter();writer.SetFileName(str(output/'target_faces.png'));writer.SetInputConnection(capture.GetOutputPort());writer.Write()
    window.Finalize();print(output/'target_faces.png')


if __name__=='__main__':main()

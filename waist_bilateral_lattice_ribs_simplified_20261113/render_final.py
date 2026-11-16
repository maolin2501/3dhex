"""Render actual exported lattice geometry, including a labeled cutaway view."""
from pathlib import Path
import sys,json
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from PIL import Image,ImageDraw
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent
sys.path.insert(0,str(ROOT/'waist_tpu70a_20261112'))
import render_source as rs
r=rs.r

def actor(mesh,color,opacity=1.):
    normals=vtk.vtkPolyDataNormals();normals.SetInputData(mesh)
    normals.SplittingOff();normals.ConsistencyOff();normals.AutoOrientNormalsOff()
    mapper=vtk.vtkPolyDataMapper();mapper.SetInputConnection(normals.GetOutputPort())
    mapper.ScalarVisibilityOff()
    a=vtk.vtkActor();a.SetMapper(mapper);p=a.GetProperty()
    p.SetColor(*(v/255 for v in color));p.SetOpacity(opacity)
    p.SetAmbient(.30);p.SetDiffuse(.72);p.SetSpecular(.12);p.SetSpecularPower(24)
    return a

def panel(mesh,size,title,subtitle,direction,up):
    im=Image.new('RGB',size,r.WHITE);d=ImageDraw.Draw(im)
    d.text((28,22),title,font=r.font(32,True),fill=r.INK)
    d.text((30,70),subtitle,font=r.font(21),fill=r.MUTED)
    view=r.render([(mesh,r.FIXED)],(size[0]-40,size[1]-120),direction=direction,up=up)
    im.paste(view,(20,110));return im

def main():
    result=json.loads((OUT/'result.json').read_text(encoding='utf8'))
    reader=vtk.vtkPLYReader();reader.SetFileName(str(OUT/'bilateral_lattice_ribs_indexed.ply'))
    reader.Update();mesh=reader.GetOutput()
    assert mesh.GetNumberOfCells()==result['triangles']
    r.actor_for=actor
    left=panel(mesh,(1360,1320),'全件晶格 · 实际导出网格',
               'BCC 内部晶格 + 六角开孔表层；锯齿侧壁与支撑竖板保留实心',(.85,-1.1,.9),(0,0,1))
    right_top=panel(mesh,(980,640),'俯视 · 双边密齿与直通缝',
                    '原中央 15 mm 通缝、右侧 0.5 mm 细缝保留',(0,0,1),(0,1,0))
    plane=vtk.vtkPlane();plane.SetOrigin(0,.9,0);plane.SetNormal(0,1,0)
    clip=vtk.vtkClipPolyData();clip.SetInputData(mesh);clip.SetClipFunction(plane);clip.Update()
    right_bottom=panel(clip.GetOutput(),(980,660),'剖切显示 · 竖板及内部晶格',
                       '仅隐藏前半部用于观察；交付模型保持完整',(.55,-1,.5),(0,0,1))
    canvas=Image.new('RGB',(2400,1500),r.PAPER);d=ImageDraw.Draw(canvas)
    d.text((35,22),'双边波浪齿 · 晶格简化版',font=r.font(46,True),fill=r.INK)
    d.text((40,92),'单胞 4 mm  |  杆径 0.6 mm  |  保留层 / 竖板名义厚度 0.8 mm  |  35 道竖板',font=r.font(27),fill=r.MUTED)
    canvas.paste(left,(20,150));canvas.paste(right_top,(1400,150));canvas.paste(right_bottom,(1400,810))
    canvas.save(OUT/'preview.png')
    print('RENDERED',mesh.GetNumberOfCells(),flush=True)

if __name__=='__main__':main()

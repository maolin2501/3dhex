from pathlib import Path
import sys,json
import numpy as np
import trimesh
from PIL import Image,ImageDraw
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'waist_tpu70a_20261112'))
from latticefill.cad_io import read_body
from latticefill.mesh_quality import as_manifold,as_trimesh
import manifold3d as mf
import render_source as rs
r=rs.r

def main():
    doc=json.loads((OUT/'design.json').read_text(encoding='utf8'))
    body=read_body(OUT/'model',0);source=body.mesh;center=np.array([0,0,r.CZ])
    ms=source.copy();ms.apply_translation(-center);man=as_manifold(ms)
    ribparts=[]
    for item in doc['ribs']:
        side=-1 if item['side']=='left' else 1
        box=trimesh.creation.box(extents=[77.5,.8,117.])
        box.apply_translation([side*46.25,item['center_y_mm'],1257.5-r.CZ])
        result=man ^ as_manifold(box)
        if not result.is_empty():ribparts.append(as_trimesh(result))
    ribs=trimesh.util.concatenate(ribparts);ribs.export(OUT/'rib_layout.ply')
    protected=body.selected_mesh(doc['preserved_face_ids']);protected.apply_translation(-center)
    a=r.triangle_mesh(ms.vertices,ms.faces)
    b=r.triangle_mesh(protected.vertices,protected.faces)
    c=r.triangle_mesh(ribs.vertices,ribs.faces)
    panels=[]
    for title,direction,up in [('保留面与竖板位置 · 俯视',(0,0,1),(0,1,0)),
                                ('竖板沿齿中心连接到同侧外缘',(.8,-1,.7),(0,0,1))]:
        im=Image.new('RGB',(1160,1000),r.WHITE);d=ImageDraw.Draw(im)
        d.text((28,24),title,font=r.font(34,True),fill=r.INK)
        d.text((30,79),'橙色：保留齿面 0.8 mm  |  蓝色：支撑竖板 0.8 mm',font=r.font(22),fill=r.MUTED)
        view=r.render([(a,(177,194,187),.13),(b,(220,134,57),1.),(c,(70,128,179),.85)],
                      (1100,850),direction=direction,up=up)
        im.paste(view,(30,135));panels.append(im)
    im=Image.new('RGB',(2380,1160),r.PAPER);d=ImageDraw.Draw(im)
    d.text((36,22),'全件晶格化 · 实心保留与横向支撑布局',font=r.font(45,True),fill=r.INK)
    d.text((40,90),'布局检查图：浅灰区域将生成晶格；中央通缝及原右侧细缝不跨接',font=r.font(26),fill=r.MUTED)
    im.paste(panels[0],(20,145));im.paste(panels[1],(1200,145))
    im.save(OUT/'support_layout.png')
    print('LAYOUT READY',len(ribs.faces),flush=True)

if __name__=='__main__':main()

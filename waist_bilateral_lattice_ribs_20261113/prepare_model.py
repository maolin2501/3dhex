"""Select the confirmed tooth sidewalls and lay out transverse vertical ribs."""
from pathlib import Path
import sys,json
import numpy as np
import trimesh
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent
sys.path.insert(0,str(ROOT))
from latticefill.cad_io import tessellate_shape,save_bodies
from OCP.STEPControl import STEPControl_Reader
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS
from OCP.BRepAdaptor import BRepAdaptor_Surface

SOURCE=ROOT/'waist_bilateral_dense_wave_20261113/bilateral_dense_wave_20mm.step'

def main():
    rd=STEPControl_Reader();assert rd.ReadFile(str(SOURCE))==1;assert rd.TransferRoots()>0
    body=tessellate_shape(rd.OneShape(),'双边密齿源模型',tolerance=.04)[0]
    assert body.mesh.is_volume
    ex=TopExp_Explorer(rd.OneShape(),TopAbs_FACE);extrusions={};fid=0
    while ex.More():
        ad=BRepAdaptor_Surface(TopoDS.Face_s(ex.Current()))
        if str(ad.GetType()).endswith('SurfaceOfExtrusion'):
            extrusions[fid]=list(ad.Direction().Coord())
        ex.Next();fid+=1
    ids=[];summary=[]
    for f in body.face_info:
        mask=body.face_ids==f['id'];tris=body.mesh.triangles[mask]
        lo=tris.reshape(-1,3).min(0);hi=tris.reshape(-1,3).max(0)
        znormal=float(np.max(np.abs(body.mesh.face_normals[mask,2])))
        vertical=f['id'] in extrusions and abs(extrusions[f['id']][2])>.999999 and hi[2]-lo[2]>5
        central=lo[0]>=-27.502 and hi[0]<=27.502 and lo[1]>=-52.01 and hi[1]<=52.01
        waveform=(hi[0]-lo[0])>1 and (hi[1]-lo[1])<3.26
        preserve=bool(vertical and central and waveform)
        if preserve:ids.append(f['id'])
        summary.append(dict(id=f['id'],kind=f['type'],bounds=[lo.tolist(),hi.tolist()],
                            area=f['area'],preserve=preserve,max_abs_normal_z=znormal))
    (OUT/'face_selection.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    assert len(ids)>=64,len(ids)
    protected=body.selected_mesh(ids)
    (OUT/'model').mkdir(exist_ok=True);save_bodies([body],OUT/'model')
    ribs=[];meta=[]
    for side in [-1,1]:
        for y in np.arange(-52,52.01,6.5):
            centers=[float(y)]
            if side==1 and abs(y)<1e-6:centers=[-.65,.65]
            for yy in centers:
                # A 0.4 mm distance band on both sides produces a 0.8 mm plate.
                vertices=np.array([[side*7.5,yy,1199.],[side*85.,yy,1199.],
                                   [side*85.,yy,1316.],[side*7.5,yy,1316.]])
                ribs.append(trimesh.Trimesh(vertices,[[0,1,2],[0,2,3]],process=False))
                meta.append(dict(side='left' if side<0 else 'right',center_y_mm=yy,
                                 nominal_thickness_mm=.8,starts_at_abs_x_mm=7.5,
                                 source_clipped=True,radial_slit_split=bool(side==1 and abs(y)<1e-6)))
    planes=trimesh.util.concatenate(ribs)
    np.savez_compressed(OUT/'rib_midplanes.npz',vertices=planes.vertices,faces=planes.faces)
    np.savez_compressed(OUT/'protected_faces.npz',vertices=protected.vertices,faces=protected.faces)
    config=dict(lattice='bcc',cell=4.,wall=.6,axis='z',layer=4.,scope='body',unselected='perforate',
                surface_mode='perforated',surface_pattern='hex',surface_thickness=.8,
                surface_cell=0.,surface_width=0.,pore_rounding=0.,junction_blend=0.,edge_width=0.,
                surface_continuity='auto',seam_style='atlas',depth=6.,shell=.8,pitch=.2,smooth=2,
                memory_mode='chunked',quality='final',design_faces=[],preserve_faces=ids,
                preserve_thicknesses={})
    doc=dict(source=str(SOURCE),source_bounds=body.mesh.bounds.tolist(),source_volume_mm3=float(body.mesh.volume),
             source_faces=len(body.face_info),preserved_face_ids=ids,preserved_face_count=len(ids),
             preserved_thickness_mm=.8,ribs=meta,rib_count=len(meta),config=config,
             note='User approved tooth vertical walls and full transverse vertical ribs to same-side outer edge; preserve all original voids.')
    (OUT/'design.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2),encoding='utf8')
    (OUT/'face_selection.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print('SOURCE',len(body.mesh.faces),'preserved faces',len(ids),'ribs',len(meta),flush=True)

if __name__=='__main__':main()

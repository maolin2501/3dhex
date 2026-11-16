"""Paired, bounded verification on the supplied shoulder STEP geometry."""
from pathlib import Path
import secrets
import json
import numpy as np
from OCP.STEPControl import STEPControl_Reader
from OCP.TopAbs import TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
from OCP.gp import gp_Pnt
from latticefill.cad_io import tessellate_shape, save_bodies
from latticefill.studio_jobs import atomic_json, compute, body_metadata
from .inspect_reference import read_triangles, render


def main():
    root=Path('.lattice-studio').resolve()
    reader=STEPControl_Reader();reader.ReadFile(str(root/'20e6035866144d96f129f5b8/input.step'))
    reader.SetSystemLengthUnit(1.);reader.TransferRoots()
    explorer=TopExp_Explorer(reader.OneShape(),TopAbs_SOLID)
    centre=np.array([-257,0,1580]);half=12
    cutter=BRepPrimAPI_MakeBox(gp_Pnt(*(centre-half)),2*half,2*half,2*half).Shape()
    common=BRepAlgoAPI_Common(explorer.Current(),cutter).Shape()
    bodies=tessellate_shape(common,'右肩参考风格 · 局部对照',tolerance=.035)
    model=root/secrets.token_hex(12);model.mkdir();save_bodies(bodies,model)
    atomic_json(model/'request.json',dict(action='local_regression'))
    atomic_json(model/'status.json',dict(state='done',progress=100,message='局部对照模型',
        result=dict(filename='真实右肩 · 24 mm 局部对照',bodies=[body_metadata(b,i) for i,b in enumerate(bodies)])))
    cfg=dict(lattice='kelvin',cell=4,wall=.55,pitch=.18,axis='z',scope='body',
        shell=0,surface_thickness=.8,surface_mode='perforated',quality='final',smooth=2)
    configs={'previous':cfg,'reference_style':cfg|dict(lattice='bcc',cell=7.3,wall=1.2,pitch=.2,
        surface_cell=6.4,surface_width=.7,pore_rounding=.35,junction_blend=.25,edge_width=1)}
    records={}
    folder=Path('reference_meshes/shoulder_v1/comparison');folder.mkdir(exist_ok=True)
    for name,config in configs.items():
        job=root/secrets.token_hex(12);job.mkdir()
        req=dict(action='generate',model_dir=str(model),body=0,config=config)
        atomic_json(job/'request.json',req);print(name,job.name,flush=True)
        compute(job,req)
        result=json.loads((job/'design.json').read_text(encoding='utf-8'))
        records[name]=dict(job=job.name,stats=result['stats'],warnings=result['warnings'])
        render(read_triangles(job/'result.stl'),folder/f'{name}.png',[-1,-.4,.2])
        print(name,'verified',result['stats'],flush=True)
        atomic_json(folder/'results.json',records)


if __name__=='__main__':main()

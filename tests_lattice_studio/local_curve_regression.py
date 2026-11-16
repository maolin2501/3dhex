"""Bounded regression on the real STEP's damaged torus, not the whole body.

Run manually with the project Python. Creates a separate model/result job;
does not modify the user's import or completed outputs.
"""
from pathlib import Path
import secrets
import numpy as np
from OCP.STEPControl import STEPControl_Reader
from OCP.TopAbs import TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
from OCP.gp import gp_Pnt
from latticefill.cad_io import read_body, tessellate_shape, save_bodies
from latticefill.studio_jobs import atomic_json, compute, body_metadata


def main():
    root=Path('.lattice-studio').resolve()
    original=root/'20e6035866144d96f129f5b8'
    body=read_body(original,0)
    centres=body.selected_mesh([26]).triangles_center
    target=centres[np.argmin(np.linalg.norm(centres-[-260,0,1580],axis=1))]
    reader=STEPControl_Reader()
    reader.ReadFile(str(original/'input.step'))
    reader.SetSystemLengthUnit(1.)
    reader.TransferRoots()
    explorer=TopExp_Explorer(reader.OneShape(),TopAbs_SOLID)
    cutter=BRepPrimAPI_MakeBox(gp_Pnt(*(target-10)),20,20,20).Shape()
    common=BRepAlgoAPI_Common(explorer.Current(),cutter).Shape()
    bodies=tessellate_shape(common,'真实模型环面局部验证',tolerance=.04)
    model=root/secrets.token_hex(12);model.mkdir()
    save_bodies(bodies,model)
    atomic_json(model/'request.json',dict(action='local_regression'))
    atomic_json(model/'status.json',dict(state='done',progress=100,message='局部验证模型',
        result=dict(filename='真实模型 · 20 mm 局部验证',bodies=[body_metadata(b,i) for i,b in enumerate(bodies)])))
    cfg=dict(lattice='kelvin',cell=4,wall=.55,pitch=.18,axis='z',scope='body',
             shell=0,surface_thickness=.8,surface_mode='perforated',quality='final')
    job=root/secrets.token_hex(12);job.mkdir()
    req=dict(action='generate',model_dir=str(model),body=0,config=cfg)
    atomic_json(job/'request.json',req)
    print('LOCAL_CURVE_JOB',job.name,flush=True)
    compute(job,req)
    print('LOCAL_CURVE_VERIFIED',job.name,flush=True)


if __name__=='__main__':main()

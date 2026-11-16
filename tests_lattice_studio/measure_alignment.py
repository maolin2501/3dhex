"""Reproducible old/new centreline measurements; not a strength assessment."""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from latticefill.cad_io import read_body
from latticefill.core import FillParams
from latticefill.surface import PerforatedSurface
from latticefill.studio_jobs import atomic_json


def main():
    root=Path('.lattice-studio')
    job=sys.argv[1]
    data=json.loads((root/job/'design.json').read_text(encoding='utf-8'));cfg=data['config']
    body=read_body(root/data['model_id'],data['source_body'])
    active=set(f['id'] for f in body.face_info)-set(cfg['preserve_faces'])
    if cfg['scope']=='faces' and cfg['unselected']=='solid':active &= set(cfg['design_faces'])
    s=PerforatedSurface(body,active,-body.mesh.bounds.mean(axis=0),chart_mode=cfg['seam_style'])
    p=FillParams(**{k:v for k,v in cfg.items() if k in FillParams.__dataclass_fields__ and k!='stats'})
    s.prepare_edges(cfg['pitch'],p)
    angles=[];deviations=[];lateral=[];targets=[]
    for root_point,target,curve in s.network.terminals:
        direction=root_point-curve[0];direction/=np.linalg.norm(direction)
        chord=target-curve[0];chord/=np.linalg.norm(chord)
        angles.append(float(np.rad2deg(np.arccos(np.clip(direction@chord,-1,1)))))
        deviations.append(float(np.linalg.norm(np.cross(curve-curve[0],chord),axis=1).max()))
        lateral.append(float(np.linalg.norm(np.cross(target-root_point,direction))))
        targets.append(target)
    out=dict(job=job,mode=cfg['seam_style'],routes=len(angles),
        angle_deg=np.percentile(angles,[0,50,95,100]).tolist(),
        angle_above_30_pct=float(np.mean(np.asarray(angles)>30)*100),
        angle_above_60_pct=float(np.mean(np.asarray(angles)>60)*100),
        lateral_shift_mm=np.percentile(lateral,[0,50,95,100]).tolist(),
        centreline_deviation_mm=np.percentile(deviations,[0,50,95,100]).tolist(),
        close_anchor_pairs=len(cKDTree(targets).query_pairs(p.wall/2)),
        angle_note='For v9, the original shaft tangent is recovered from the retreated root. For v10 two-ended members, use generation straight_angle_deg for the original BCC direction.')
    folder=Path('reference_meshes/shoulder_v1/aligned');folder.mkdir(exist_ok=True)
    atomic_json(folder/f'alignment_{job}.json',out)
    print(json.dumps(out,ensure_ascii=False),flush=True)


if __name__=='__main__':main()

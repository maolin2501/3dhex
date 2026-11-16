"""Adaptive shared-surface pore network: new results, never overwrite v8."""
import json
import secrets
import sys
from pathlib import Path
import numpy as np
from latticefill.cad_io import read_body
from latticefill.core import FillParams
from latticefill.surface import PerforatedSurface
from latticefill.studio_jobs import atomic_json, compute
from .inspect_reference import read_triangles, render
from .render_continuity_comparison import crop

ROOT = Path('.lattice-studio').resolve()
FOLDER = Path('reference_meshes/shoulder_v1/adaptive')
FULL = '7727e8a5729059252a377d5c'
CROP = '824b688711a5ebaa5f38a034'


def main():
    mode=sys.argv[1] if len(sys.argv)>1 else 'crop'
    FOLDER.mkdir(exist_ok=True)
    previous=json.loads((ROOT/(CROP if mode=='crop' else FULL)/'design.json').read_text(encoding='utf-8'))
    config=previous['config']|dict(seam_style='adaptive',surface_continuity='auto',memory_mode='chunked')
    if mode=='graph':
        body=read_body(ROOT/previous['model_id'],0)
        active=set(f['id'] for f in body.face_info)-set(config['preserve_faces'])
        surface=PerforatedSurface(body,active,-body.mesh.bounds.mean(axis=0),chart_mode='adaptive')
        p=FillParams(**{k:v for k,v in config.items() if k in FillParams.__dataclass_fields__})
        surface.prepare_edges(config['pitch'],p)
        atomic_json(FOLDER/'graph_check.json',p.stats)
        print(p.stats,flush=True)
        return
    if mode=='render':
        for name,path in dict(before=ROOT/FULL/'result.stl',after=ROOT/sys.argv[2]/'result.stl',
            reference=Path('reference_meshes/shoulder_v1/722机器人-右肩晶格v1.stl')).items():
            tris=read_triangles(path)
            for part,centre in [('seam',[-254,-15,1618.375]),('struts',[-250,-22,1640])]:
                centre=np.array(centre)
                render(crop(tris,centre,12),FOLDER/f'{part}_{name}.png',[-1,-.4,.2],focal=centre,scale=14)
        return
    job=ROOT/secrets.token_hex(12);job.mkdir()
    req=dict(action='generate',model_dir=str(ROOT/previous['model_id']),body=0,config=config)
    atomic_json(job/'request.json',req)
    atomic_json(FOLDER/f'{mode}_job.json',dict(job=job.name,baseline=CROP if mode=='crop' else FULL))
    print('ADAPTIVE_JOB',mode,job.name,flush=True)
    try:compute(job,req)
    except Exception as exc:
        atomic_json(job/'status.json',dict(state='failed',progress=0,message=str(exc)))
        raise
    result=json.loads((job/'design.json').read_text(encoding='utf-8'))
    atomic_json(FOLDER/f'{mode}_result.json',result)
    if mode=='crop':
        render(read_triangles(job/'result.stl'),FOLDER/'crop_after.png',[-1,-.4,.2])
        render(read_triangles(ROOT/CROP/'result.stl'),FOLDER/'crop_before.png',[-1,-.4,.2])
    print('ADAPTIVE_DONE',job.name,result['stats'],result['performance'],flush=True)


if __name__=='__main__':main()

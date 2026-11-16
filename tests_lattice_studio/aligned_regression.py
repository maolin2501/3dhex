"""Independent v10 jobs and bounded graph checks; preserve previous outputs."""
import json
import secrets
import sys
from pathlib import Path
import numpy as np
from latticefill.cad_io import read_body
from latticefill.surface import PerforatedSurface
from latticefill.core import FillParams
from latticefill.studio_jobs import compute,atomic_json
from .inspect_reference import read_triangles,render
from .render_continuity_comparison import crop,view_crop

ROOT=Path('.lattice-studio').resolve()
FOLDER=Path('reference_meshes/shoulder_v1/aligned')
BASE='e7bec2da1a4edd605dae605a'


def main(seam_style='aligned',base=BASE,folder=FOLDER):
    BASE=base;FOLDER=folder
    mode=sys.argv[1] if len(sys.argv)>1 else 'graph'
    FOLDER.mkdir(exist_ok=True)
    old=json.loads((ROOT/('cbbd529cbbb9efc7b960415e' if mode=='crop' else BASE)/'design.json').read_text(encoding='utf-8'))
    cfg=old['config']|dict(seam_style=seam_style,surface_continuity='auto',memory_mode='chunked')
    model=ROOT/old['model_id']
    if mode=='cap':
        from .stitch_regression import crop_model
        model,cfg=crop_model(ROOT,old|dict(config=cfg),centre=[-219.,-15.,1681.])
    if mode=='graph':
        body=read_body(ROOT/old['model_id'],0)
        active=set(f['id'] for f in body.face_info)-set(cfg['preserve_faces'])
        s=PerforatedSurface(body,active,-body.mesh.bounds.mean(axis=0),chart_mode=seam_style)
        p=FillParams(**{k:v for k,v in cfg.items() if k in FillParams.__dataclass_fields__ and k!='stats'})
        s.prepare_edges(cfg['pitch'],p)
        atomic_json(FOLDER/'graph_check.json',p.stats)
        np.savez(FOLDER/'graph_sites.npz',sites=s.adaptive.sites+body.mesh.bounds.mean(axis=0))
        print(json.dumps(p.stats,ensure_ascii=False),flush=True)
        return
    if mode=='render':
        for label,path in dict(before=ROOT/BASE/'result.stl',after=ROOT/sys.argv[2]/'result.stl',
                reference=Path('reference_meshes/shoulder_v1/722机器人-右肩晶格v1.stl')).items():
            tris=read_triangles(path)
            for part,centre in [('seam',[-254,-15,1618.375]),('struts',[-250,-22,1640]),('cap',[-219,-15,1681])]:
                centre=np.array(centre)
                subset=(view_crop(tris,centre,[-1,-.4,.2]) if seam_style=='vertex' else crop(tris,centre,12))
                render(subset,FOLDER/f'{part}_{label}.png',[-1,-.4,.2],focal=centre,scale=14)
        return
    job=ROOT/secrets.token_hex(12);job.mkdir()
    req=dict(action='generate',model_dir=str(model),body=0,config=cfg)
    atomic_json(job/'request.json',req)
    atomic_json(FOLDER/f'{mode}_job.json',dict(job=job.name,baseline=BASE))
    print('ALIGNED_JOB',mode,job.name,flush=True)
    try:compute(job,req)
    except Exception as exc:
        atomic_json(job/'status.json',dict(state='failed',progress=0,message=str(exc)))
        raise
    result=json.loads((job/'design.json').read_text(encoding='utf-8'))
    atomic_json(FOLDER/f'{mode}_result.json',result)
    if mode in ('crop','cap'):render(read_triangles(job/'result.stl'),FOLDER/f'{mode}_after.png',[-1,-.4,.2])
    print('ALIGNED_DONE',job.name,result['stats'],result['performance'],flush=True)


if __name__=='__main__':main()

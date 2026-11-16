"""Repeat the user's failed full configuration in a NEW job directory."""
import json
from pathlib import Path
import secrets
import sys
import numpy as np
import trimesh
from latticefill import core
from latticefill.studio_jobs import atomic_json,compute


def main():
    root=Path('.lattice-studio').resolve()
    request=json.loads((root/'f80f915100c8746bdd9c7530/request.json').read_text(encoding='utf-8'))
    request['config']['memory_mode']='chunked'
    job=root/secrets.token_hex(12);job.mkdir()
    atomic_json(job/'request.json',request)
    print('FULL_LOWMEM_JOB',job.name,flush=True)
    finish=core._finish_mesh
    def checkpoint(out,mesh,p,pitch,density,progress):
        # Manual stress-test checkpoint only, not normal user generation.
        np.savez(job/'raw_mesh.npz',vertices=out.vertices,faces=out.faces)
        atomic_json(job/'raw_stats.json',dict(stats=p.stats,pitch=pitch,density=density))
        return finish(out,mesh,p,pitch,density,progress)
    core._finish_mesh=checkpoint
    if len(sys.argv)>1:
        source=root/sys.argv[1]
        import latticefill.studio_jobs as jobs
        def resume(mesh,p,**kwargs):
            with np.load(source/'raw_mesh.npz') as data:
                out=trimesh.Trimesh(data['vertices'],data['faces'],process=False)
            info=json.loads((source/'raw_stats.json').read_text(encoding='utf-8'))
            p.stats.update(info['stats'])
            p.stats['resumed_from']=source.name
            return finish(out,mesh,p,info['pitch'],info['density'],kwargs['progress'])
        jobs.fill_solid=resume
    try:
        compute(job,request)
    except Exception as exc:
        atomic_json(job/'status.json',dict(state='failed',progress=0,message=str(exc)))
        raise
    result=json.loads((job/'design.json').read_text(encoding='utf-8'))
    print('FULL_LOWMEM_DONE',job.name,result['stats'],result['performance'],flush=True)


if __name__=='__main__':main()

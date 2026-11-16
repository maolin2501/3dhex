"""Durable pre-Boolean geometry; no repeated field generation after a crash."""
import json
import os
import shutil
from pathlib import Path
import numpy as np
import trimesh

CHECKPOINT_FACE_MIN=1_000_000


def signature(params):
    return {k:v for k,v in vars(params).items() if k not in ('stats','work_dir','preview')}


def save_pretrim(mesh,p,pitch,density):
    from .studio_jobs import atomic_json
    directory=Path(p.work_dir)/'pretrim'
    directory.mkdir(exist_ok=True)
    if (directory/'ready.json').exists():raise ValueError('裁切检查点已存在，不能覆盖')
    required=mesh.vertices.nbytes+mesh.faces.nbytes+128*1024**2
    if shutil.disk_usage(directory).free<required:raise ValueError('裁切检查点磁盘空间不足')
    files={}
    for name,array in [('vertices',mesh.vertices),('faces',mesh.faces)]:
        temp=directory/f'{name}.tmp';final=directory/f'{name}.npy'
        with temp.open('wb') as stream:
            np.save(stream,np.asarray(array),allow_pickle=False)
            stream.flush();os.fsync(stream.fileno())
        temp.replace(final)
        files[final.name]=final.stat().st_size
    atomic_json(directory/'ready.json',dict(params=signature(p),stats=p.stats,
        pitch=pitch,density=density,files=files,stage='after_smoothing_before_trim'))


def load_pretrim(job,p):
    directory=Path(job)/'pretrim';ready=directory/'ready.json'
    if not ready.is_file():raise ValueError('裁切检查点不完整，不能恢复')
    meta=json.loads(ready.read_text(encoding='utf-8'))
    stored=dict(meta['params']);stored.setdefault('preserve_thicknesses',{})
    if stored!=signature(p):raise ValueError('检查点参数与当前参数不一致')
    for name,size in meta['files'].items():
        if name not in ('vertices.npy','faces.npy') or not (directory/name).is_file() or (directory/name).stat().st_size!=size:
            raise ValueError('裁切检查点数据不完整')
    # The native geometry binding requires writable arrays; copy-on-write maps
    # satisfy it without permitting mutations of the durable checkpoint.
    vertices=np.load(directory/'vertices.npy',mmap_mode='c',allow_pickle=False)
    faces=np.load(directory/'faces.npy',mmap_mode='c',allow_pickle=False)
    if vertices.ndim!=2 or vertices.shape[1]!=3 or faces.ndim!=2 or faces.shape[1]!=3:
        raise ValueError('裁切检查点数组无效')
    p.stats.update(meta['stats'],resumed_from=Path(job).name)
    return trimesh.Trimesh(vertices,faces,process=False),meta['pitch'],meta['density']

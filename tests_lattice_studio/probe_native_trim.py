"""Isolated native Boolean reproducer; deliberately imports no VTK or trimesh."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import manifold3d as manifold


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('job');parser.add_argument('--one-cpu',action='store_true')
    args=parser.parse_args();job=Path(args.job)
    if args.one_cpu:
        import ctypes
        kernel=ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype=ctypes.c_void_p
        handle=kernel.GetCurrentProcess();mask=ctypes.c_size_t();system=ctypes.c_size_t()
        if not kernel.GetProcessAffinityMask(ctypes.c_void_p(handle),ctypes.byref(mask),ctypes.byref(system)):
            raise ctypes.WinError()
        first=mask.value & -mask.value
        if not kernel.SetProcessAffinityMask(ctypes.c_void_p(handle),ctypes.c_size_t(first)):
            raise ctypes.WinError()
        print('single CPU affinity',first,flush=True)
    request=json.loads((job/'request.json').read_text(encoding='utf-8'))
    source=np.load(Path(request['model_dir'])/f"body_{request['body']}.npz")
    vertices=source['vertices'];centre=(vertices.min(axis=0)+vertices.max(axis=0))/2
    vertices-=centre
    start=time.perf_counter()
    a=manifold.Manifold(manifold.Mesh64(vertices,np.asarray(source['faces'],dtype=np.uint64)))
    print('source',a.status(),a.num_tri(),a.volume(),flush=True)
    v=np.load(job/'pretrim/vertices.npy',mmap_mode='c');f=np.load(job/'pretrim/faces.npy',mmap_mode='c')
    for i in range(0,len(v),262144):
        if not np.isfinite(v[i:i+262144]).all():raise ValueError('Nonfinite checkpoint vertices')
    if f.min()<0 or f.max()>=len(v):raise ValueError('Bad checkpoint indices')
    print('checkpoint',v.shape,f.shape,'finite and indexed',flush=True)
    b=manifold.Manifold(manifold.Mesh64(v,np.asarray(f,dtype=np.uint64)))
    print('lattice',b.status(),b.num_tri(),b.volume(),'seconds',time.perf_counter()-start,flush=True)
    solid=a^b
    print('evaluating Boolean',flush=True)
    print('trim',solid.status(),solid.num_tri(),solid.volume(),'seconds',time.perf_counter()-start,flush=True)
    if solid.status()!=manifold.Error.NoError or solid.is_empty() or solid.volume()<=0:
        raise ValueError('Invalid clipped mesh')
    del a,b,v,f
    data=solid.to_mesh64()
    output=job/('trim_single_cpu' if args.one_cpu else 'trim_isolated')
    output.mkdir(exist_ok=False)
    np.save(output/'vertices.npy',np.asarray(data.vert_properties[:,:3]))
    np.save(output/'faces.npy',np.asarray(data.tri_verts,dtype=np.int64))
    print('saved',output,flush=True)


if __name__=='__main__':main()

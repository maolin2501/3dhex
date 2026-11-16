from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
OUT=Path(__file__).resolve().parent;sys.path.insert(0,str(OUT.parent))
from latticefill.mesh_memory import export_stl_batched
ply=OUT/'open_wave_lattice_indexed.ply'
with ply.open('rb') as f:
    while True:
        line=f.readline().decode('ascii').strip()
        if line.startswith('element vertex'):nv=int(line.split()[-1])
        if line.startswith('element face'):nf=int(line.split()[-1])
        if line=='end_header':break
    offset=f.tell()
vertices=np.memmap(ply,dtype='<f4',offset=offset,shape=(nv,3),mode='r')
faces=np.memmap(ply,dtype=[('count','u1'),('v','<i4',(3,))],offset=offset+12*nv,shape=(nf,),mode='r')
mesh=SimpleNamespace(vertices=vertices,faces=faces['v'])
export_stl_batched(mesh,OUT/'open_wave_lattice_print.stl')
print('EXPORTED STL',nf,(OUT/'open_wave_lattice_print.stl').stat().st_size,flush=True)

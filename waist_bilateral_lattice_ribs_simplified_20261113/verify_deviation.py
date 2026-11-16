"""Bidirectional sampled distance comparison, with bounded mesh allocation."""
from pathlib import Path
import json,time,gc
import numpy as np
import vtk
OUT=Path(__file__).resolve().parent
SOURCE=OUT.parent/'waist_bilateral_lattice_ribs_20261113'

def sample(path,seed):
    with path.open('rb') as f:
        while True:
            line=f.readline().decode('ascii').strip()
            if line.startswith('element vertex'):nv=int(line.split()[-1])
            if line.startswith('element face'):nf=int(line.split()[-1])
            if line=='end_header':break
        offset=f.tell()
    vertices=np.memmap(path,dtype='<f4',offset=offset,shape=(nv,3),mode='r')
    faces=np.memmap(path,dtype=[('count','u1'),('v','<i4',(3,))],offset=offset+12*nv,shape=(nf,),mode='r')
    rng=np.random.default_rng(seed);indices=rng.integers(0,nf,60000)
    t=vertices[faces['v'][indices]].astype(float)
    bary=rng.dirichlet([1,1,1],len(indices))
    return np.einsum('ni,nij->nj',bary,t)

def measure(path,points):
    print('Building closest-surface locator',path.name,flush=True)
    reader=vtk.vtkPLYReader();reader.SetFileName(str(path));reader.Update()
    locator=vtk.vtkStaticCellLocator();locator.SetDataSet(reader.GetOutput())
    locator.SetNumberOfCellsPerNode(24);locator.BuildLocator()
    closest=[0.,0.,0.];cid=vtk.reference(0);sid=vtk.reference(0);d2=vtk.reference(0.)
    distances=np.empty(len(points))
    for i,p in enumerate(points):
        locator.FindClosestPoint(p,closest,cid,sid,d2)
        distances[i]=np.sqrt(float(d2))
    result=dict(samples=len(points),max_mm=float(distances.max()),p99_mm=float(np.quantile(distances,.99)),mean_mm=float(distances.mean()))
    del locator,reader;gc.collect();print(json.dumps(result),flush=True)
    return result

started=time.time()
original=SOURCE/'bilateral_lattice_ribs_indexed.ply';reduced=OUT/'bilateral_lattice_ribs_indexed.ply'
original_samples=sample(original,2026111301);reduced_samples=sample(reduced,2026111302)
reverse=measure(original,reduced_samples);forward=measure(reduced,original_samples)
report=dict(reduced_to_original=reverse,original_to_reduced=forward,
            elapsed_seconds=time.time()-started,
            note='Independent bidirectional random triangle-surface sampling; not an exhaustive Hausdorff bound.')
(OUT/'deviation_verification.json').write_text(json.dumps(report,indent=2),encoding='utf8')
tolerance=json.loads((OUT/'simplification.json').read_text(encoding='utf8'))['tolerance_mm']
assert max(reverse['max_mm'],forward['max_mm'])<=tolerance+.00002,report
print('DEVIATION VERIFIED',flush=True)

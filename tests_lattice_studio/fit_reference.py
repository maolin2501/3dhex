"""Fit analytic strut families to reference interior surface samples."""
import json
from pathlib import Path
import numpy as np
import trimesh
from scipy.optimize import differential_evolution
from .inspect_reference import read_triangles
from latticefill.cad_io import read_body
from latticefill.lattices import _bcc_dist, _kelvin_dist, _cubic_dist


def main():
    path=next(Path('reference_meshes/shoulder_v1').glob('*.stl'))
    tris=read_triangles(path)
    centres=tris.mean(axis=1)
    mask=np.all(abs(centres-[-252,0,1580])<[9,20,20],axis=1)
    tris=np.asarray(tris[mask],dtype=float)
    centres=tris.mean(axis=1)
    body=read_body(Path('.lattice-studio/20e6035866144d96f129f5b8'),0)
    rng=np.random.default_rng(943)
    areas=np.linalg.norm(np.cross(tris[:,1]-tris[:,0],tris[:,2]-tris[:,0]),axis=1)
    sample=centres[rng.choice(len(tris),8000,p=areas/areas.sum())]
    _,distance,_=trimesh.proximity.closest_point(body.mesh,sample)
    sample=sample[distance>1.5][:2000]
    sample-=np.array([-250,0,1580])
    print('interior samples',len(sample),flush=True)
    # Surface normals on cylindrical rods are perpendicular to their axes.
    face=rng.choice(len(tris),15000,p=areas/areas.sum())
    picked=centres[face]
    _,dist,_=trimesh.proximity.closest_point(body.mesh,picked)
    face=face[dist>1.5]
    normals=np.cross(tris[face,1]-tris[face,0],tris[face,2]-tris[face,0]);normals/=np.linalg.norm(normals,axis=1)[:,None]
    k=np.arange(4000);z=(k+.5)/4000
    theta=k*np.pi*(3-np.sqrt(5));rad=np.sqrt(1-z*z)
    directions=np.column_stack([rad*np.cos(theta),rad*np.sin(theta),z])
    score=np.concatenate([(abs(normals@part.T)<.025).mean(axis=0) for part in np.array_split(directions,20)])
    axes=[]
    for i in np.argsort(score)[::-1]:
        if all(abs(directions[i]@a)<.98 for a in axes):
            axes.append(directions[i]);print('AXIS',directions[i].tolist(),float(score[i]),flush=True)
        if len(axes)==8:break
    fits={}
    for kind,fn in [('bcc',_bcc_dist),('kelvin',_kelvin_dist),('cubic',_cubic_dist)]:
        def objective(v):
            cell=v[0];phase=np.asarray(v[1:4])*cell
            d=fn(*(sample-phase).T,cell)
            error=abs(d-v[4])
            return np.mean(np.minimum(error,.8)**2)
        candidates=[]
        for c in [5,7,9,11,13,15]:
            opt=differential_evolution(objective,[(c-1,c+1),(0,1),(0,1),(0,1),(.25,1.2)],popsize=16,maxiter=140,seed=11,tol=.001)
            candidates.append(opt)
            print(kind,c,float(np.sqrt(opt.fun)),opt.x.tolist(),flush=True)
        opt=min(candidates,key=lambda o:o.fun)
        fits[kind]=dict(error_mm=float(np.sqrt(opt.fun)),cell=float(opt.x[0]),phase=(opt.x[1:4]*opt.x[0]).tolist(),diameter=float(opt.x[4]*2))
        print(kind, fits[kind],flush=True)
    Path('reference_meshes/shoulder_v1/interior_fit.json').write_text(json.dumps(fits,indent=2),encoding='utf-8')


if __name__=='__main__':main()

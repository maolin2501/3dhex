"""Identical views of uncapped STL crops: real seam and free-end locations."""
from pathlib import Path
import sys
import numpy as np
from .inspect_reference import read_triangles, render


def crop(triangles, centre, half):
    chunks=[]
    for k in range(0,len(triangles),65536):
        block=triangles[k:k+65536]
        keep=np.all(abs(block.mean(axis=1)-centre)<half,axis=1)
        if keep.any():chunks.append(np.array(block[keep]))
    if not chunks:raise ValueError('Empty crop')
    return np.vstack(chunks)


def view_crop(triangles,centre,direction,scale=14,aspect=1.2):
    """Orthographic view crop with unlimited depth, not a cut through rods.

    All triangles overlapping the image rectangle survive along the view ray;
    a 3D AABB crop can otherwise create misleading free ends inside the image.
    """
    normal=np.asarray(direction,dtype=float);normal/=np.linalg.norm(normal)
    horizontal=np.cross([0.,0.,1.],normal);horizontal/=np.linalg.norm(horizontal)
    vertical=np.cross(normal,horizontal)
    basis=np.array([horizontal,vertical]);extent=np.array([scale*aspect,scale])
    chunks=[]
    for k in range(0,len(triangles),65536):
        block=triangles[k:k+65536]
        projected=(block-np.asarray(centre))@basis.T
        keep=np.all((projected.max(axis=1)>=-extent)&(projected.min(axis=1)<=extent),axis=1)
        if keep.any():chunks.append(np.array(block[keep]))
    if not chunks:raise ValueError('Empty view crop')
    return np.vstack(chunks)


def main():
    root=Path('.lattice-studio')
    output=Path('reference_meshes/shoulder_v1/continuity');output.mkdir(exist_ok=True)
    paths=dict(before=root/'a7affebbef2cba1b41d6848e/result.stl',
               after=root/sys.argv[1]/'result.stl',
               reference=Path('reference_meshes/shoulder_v1/722机器人-右肩晶格v1.stl'))
    for name,path in paths.items():
        tris=read_triangles(path)
        for part,centre in [('seam',[-254,-15,1618.375]),('struts',[-250,-22,1640])]:
            centre=np.array(centre)
            subset=crop(tris,centre,12)
            render(subset,output/f'{part}_{name}.png',[-1,-.4,.2],focal=centre,scale=14)
            print(name,part,len(subset),flush=True)


if __name__=='__main__':main()

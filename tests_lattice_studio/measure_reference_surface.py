"""Estimate outer-face pore spacing from a raster of the reference STL."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import cKDTree
from latticefill.cad_io import read_body, upgrade_face_charts
from latticefill.surface import chart_uv
from .inspect_reference import read_triangles


def main():
    folder=Path('reference_meshes/shoulder_v1')
    triangles=read_triangles(next(folder.glob('*.stl')))
    mids=triangles.mean(axis=1)
    triangles=np.asarray(triangles[np.all(abs(mids-[-257,0,1580])<[8,22,22],axis=1)],dtype=float)
    body=read_body(Path('.lattice-studio/20e6035866144d96f129f5b8'),0)
    upgrade_face_charts(body,Path('.lattice-studio/20e6035866144d96f129f5b8'),0)
    chart=body.face_info[26]['chart']
    q=triangles.mean(axis=1)-chart['origin']
    h=q@chart['axis'];rho=np.hypot(q@chart['u'],q@chart['v'])
    dist=abs(np.hypot(chart['radial_sign']*rho-chart['major_radius'],h)-chart['minor_radius'])
    triangles=triangles[dist<.065]
    uv=np.column_stack(chart_uv(triangles.reshape(-1,3),chart)).reshape(-1,3,2)
    period=2*np.pi*chart['radius'];uv[:,:,0]%=period
    low=uv.min(axis=(0,1));high=uv.max(axis=(0,1));pitch=.04
    im=Image.new('L',tuple((np.ceil((high-low)/pitch)+1).astype(int)),0);draw=ImageDraw.Draw(im)
    for t in uv:draw.polygon([tuple(v) for v in ((t-low)/pitch)],fill=255)
    im.save(folder/'surface_mask.png')
    mask=np.asarray(im)>0
    holes,n=ndimage.label(~mask)
    sizes=np.bincount(holes.ravel())*pitch**2
    border=set(np.unique(np.concatenate([holes[0],holes[-1],holes[:,0],holes[:,-1]])))
    ids=[i for i in range(1,n+1) if i not in border and sizes[i]>1]
    centres=np.array(ndimage.center_of_mass(~mask,holes,ids))*pitch
    spacing=cKDTree(centres).query(centres,k=2)[0][:,1]
    pore_apothems=np.array([ndimage.distance_transform_edt(holes==i).max()*pitch for i in ids])
    result=dict(pores=len(ids),spacing_median=float(np.median(spacing)),spacing_range=np.quantile(spacing,[.1,.9]).tolist(),
                pore_across_flats_median=float(np.median(2*pore_apothems)),
                web_width_estimate=float(np.median(spacing)-np.median(2*pore_apothems)))
    print(result,flush=True)
    (folder/'surface_measurements.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__':main()

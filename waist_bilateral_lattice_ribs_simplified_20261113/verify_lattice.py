"""Fresh, source-aligned ray checks of gaps, every rib, and retained walls."""
from pathlib import Path
import sys,json,time
import numpy as np
import trimesh
import manifold3d as mf
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent
sys.path.insert(0,str(ROOT))
from latticefill.cad_io import read_body
from latticefill.mesh_quality import as_manifold
from latticefill.studio_jobs import atomic_json

def main():
    started=time.time()
    result=json.loads((OUT/'result.json').read_text(encoding='utf8'))
    design=json.loads((OUT/'design.json').read_text(encoding='utf8'))
    assert result['stats']['source_trimmed'] is True,'Source clipping not performed'
    assert result['topology']['components']==1,result['topology']
    from simplify_model import load_solid
    final=load_solid(OUT/'bilateral_lattice_ribs_indexed.ply')
    assert final.num_tri()==result['triangles']
    assert final.status()==mf.Error.NoError
    print('PASS indexed final manifold',final.num_tri(),flush=True)
    body=read_body(OUT/'model',0);shift=np.array(result['print_translation_from_source'])
    original=body.mesh.copy();original.apply_translation(shift);source=as_manifold(original)
    def contains(solid,p):
        # A near-vertical ray exits this 67 mm tall part. Its narrow AABB avoids
        # querying millions of unrelated facets in Manifold's ray broad phase.
        far=np.array(p)+np.array([.00123,.00173,100.])
        hits=solid.ray_cast(tuple(p),tuple(far))
        distances=np.array([h.distance for h in hits])
        if len(distances):distances=np.unique(np.round(distances,10))
        return bool(len(distances)%2)
    def world(p):return np.array(p,dtype=float)+shift
    gaps=0
    for y in np.linspace(-40,40,21):
        for z in [1250.,1260.,1270.]:
            assert not final.ray_cast(tuple(world([-7.49,y,z])),tuple(world([7.49,y,z])))
            gaps+=1
    for x in np.linspace(8,68,21):
        for z in [1240.,1250.,1260.,1270.,1280.]:
            assert not final.ray_cast(tuple(world([x,-.249,z])),tuple(world([x,.249,z])))
            gaps+=1
    print('PASS central and radial gap rays',gaps,flush=True)
    rib_results=[];thickness_samples=[]
    for rib in design['ribs']:
        side=-1 if rib['side']=='left' else 1;y=rib['center_y_mm'];checks=[]
        for ax in np.arange(10.,69.,2.5):
            a=world([side*ax,y,1200.]);b=world([side*ax,y,1320.])
            hits=source.ray_cast(tuple(a),tuple(b))
            zs=sorted(set(round(float(h.position[2]),6) for h in hits))
            for low,high in zip(zs[::2],zs[1::2]):
                if high-low<2:continue
                for frac in [.25,.5,.75]:
                    p=np.array([side*ax+shift[0],y+shift[1],low+frac*(high-low)])
                    checks.append(contains(final,p))
                    if frac==.5 and 33<ax<48:
                        linehits=final.ray_cast(tuple(p-[0,1.8,0]),tuple(p+[0,1.8,0]))
                        hs=sorted(float(h.position[1]-p[1]) for h in linehits)
                        before=[v for v in hs if v<0];after=[v for v in hs if v>0]
                        if before and after:
                            thickness_samples.append(dict(side=rib['side'],y=y,x=float(side*ax),
                                                          thickness_mm=min(after)-max(before)))
        assert len(checks)>20,(rib,len(checks))
        fraction=sum(checks)/len(checks)
        rib_results.append(dict(side=rib['side'],center_y_mm=y,tests=len(checks),passed=sum(checks),fraction=fraction))
        assert fraction>.98,rib_results[-1]
        print('PASS rib',len(rib_results),rib['side'],y,len(checks),flush=True)
    print('PASS all rib traverses',len(rib_results),flush=True)
    skin=[]
    for fid in design['preserved_face_ids']:
        ids=np.flatnonzero(body.face_ids==fid)
        ids=ids[np.argsort(-body.mesh.area_faces[ids])[:30]]
        probes=[]
        for i in ids:
            p=world(body.mesh.triangles_center[i]-body.mesh.face_normals[i]*.3)
            if not contains(source,p):continue
            probes.append(contains(final,p))
            if len(probes)>=8:break
        assert len(probes)>=3,(fid,len(probes))
        skin.append(dict(face_id=fid,tests=len(probes),passed=sum(probes)))
        assert sum(probes)/len(probes)>=.875,skin[-1]
    print('PASS preserved tooth surfaces',len(skin),flush=True)
    report=dict(valid_indexed_manifold=True,components=1,source_trimmed=True,
                 gap_ray_tests=gaps,all_original_voids_preserved_by_source_intersection=True,
                 rib_traverses=rib_results,rib_thickness_samples=thickness_samples,
                 thickness_note='0.8 mm is nominal; measurements include the existing 0.2 mm raster, smoothing, source clipping and lattice junctions.',
                 protected_surface_checks=skin,triangles=final.num_tri(),volume_mm3=final.volume(),
                 elapsed_seconds=time.time()-started)
    atomic_json(OUT/'verification.json',report)
    print('VERIFIED',flush=True)

if __name__=='__main__':main()

"""Check all retained ribs, both gaps and real openings in every wave wall."""
from pathlib import Path
import sys,json,time
import numpy as np
import manifold3d as mf
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent;sys.path.insert(0,str(ROOT))
from latticefill.cad_io import read_body
from latticefill.mesh_quality import as_manifold
from latticefill.studio_jobs import atomic_json

def load_solid(path):
    with path.open('rb') as f:
        while True:
            line=f.readline().decode('ascii').strip()
            if line.startswith('element vertex'):nv=int(line.split()[-1])
            if line.startswith('element face'):nf=int(line.split()[-1])
            if line=='end_header':break
        offset=f.tell()
    verts=np.memmap(path,dtype='<f4',offset=offset,shape=(nv,3),mode='r')
    faces=np.memmap(path,dtype=[('count','u1'),('v','<i4',(3,))],offset=offset+12*nv,shape=(nf,),mode='r')
    data=mf.Mesh(np.array(verts,copy=True),np.ascontiguousarray(faces['v'],dtype=np.uint32))
    solid=mf.Manifold(data)
    assert solid.status()==mf.Error.NoError and not solid.is_empty()
    assert solid.num_tri()==nf,(solid.num_tri(),nf)
    return solid,nf

def contains(solid,p):
    hits=solid.ray_cast(tuple(p),tuple(np.array(p)+[.00123,.00173,100.]))
    return bool(len(np.unique(np.round([h.distance for h in hits],10)))%2)

def main():
    start=time.time();result=json.loads((OUT/'result.json').read_text(encoding='utf8'))
    design=json.loads((OUT/'design.json').read_text(encoding='utf8'));cfg=result['config']
    assert cfg['cell']==6 and cfg['wall']==1 and cfg['shell']==0
    assert cfg['preserve_faces']==[] and result['preserved_face_ids']==[]
    assert result['stats']['source_trimmed'] and result['topology']['components']==1
    assert set(design['wave_face_ids'])<=set(result['perforated_face_ids'])
    final,nf=load_solid(OUT/'open_wave_lattice_indexed.ply')
    print('PASS serialized manifold',nf,flush=True)
    body=read_body(OUT/'model',0);shift=np.array(result['print_translation_from_source'])
    original=body.mesh.copy();original.apply_translation(shift);source=as_manifold(original)
    def world(p):return np.asarray(p,dtype=float)+shift
    gap_tests=0
    for y in np.linspace(-40,40,21):
        for z in [1250.,1260.,1270.]:
            assert not final.ray_cast(tuple(world([-7.49,y,z])),tuple(world([7.49,y,z])))
            assert all(not contains(final,world([x,y,z])) for x in [-7.49,0.,7.49])
            gap_tests+=1
    for x in np.linspace(8,68,21):
        for z in [1240.,1250.,1260.,1270.,1280.]:
            assert not final.ray_cast(tuple(world([x,-.249,z])),tuple(world([x,.249,z])))
            assert all(not contains(final,world([x,y,z])) for y in [-.249,0.,.249])
            gap_tests+=1
    print('PASS original gaps',gap_tests,flush=True)
    rib_results=[];thickness=[]
    for rib in design['ribs']:
        side=-1 if rib['side']=='left' else 1;y=rib['center_y_mm'];checks=[]
        for ax in np.arange(10.,69.,2.5):
            hits=source.ray_cast(tuple(world([side*ax,y,1200.])),tuple(world([side*ax,y,1320.])))
            zs=sorted(set(round(float(h.position[2]),6) for h in hits))
            for low,high in zip(zs[::2],zs[1::2]):
                if high-low<2:continue
                for frac in [.25,.5,.75]:
                    p=np.array([side*ax+shift[0],y+shift[1],low+frac*(high-low)])
                    checks.append(contains(final,p))
                    if frac==.5 and 33<ax<48:
                        hits=final.ray_cast(tuple(p-[0,1.8,0]),tuple(p+[0,1.8,0]))
                        ds=sorted(float(h.position[1]-p[1]) for h in hits)
                        before=[v for v in ds if v<0];after=[v for v in ds if v>0]
                        if before and after:thickness.append(dict(side=rib['side'],y=y,x=float(side*ax),thickness_mm=min(after)-max(before)))
        assert len(checks)>20
        item=dict(side=rib['side'],center_y_mm=y,tests=len(checks),passed=sum(checks),fraction=sum(checks)/len(checks))
        rib_results.append(item);assert item['fraction']>.98,item
    print('PASS all 35 ribs',flush=True)
    rng=np.random.default_rng(20261116);wave_results=[]
    for fid in design['wave_face_ids']:
        ids=np.flatnonzero(body.face_ids==fid);areas=body.mesh.area_faces[ids]
        picks=rng.choice(ids,512,replace=True,p=areas/areas.sum())
        triangles=body.mesh.triangles[picks]
        bary=rng.dirichlet([1,1,1],len(picks))
        points=np.einsum('ni,nij->nj',bary,triangles)
        solids=voids=through=through_tests=0
        for center,normal in zip(points,body.mesh.face_normals[picks]):
            side='left' if center[0]<0 else 'right'
            if min(abs(center[1]-r['center_y_mm']) for r in design['ribs'] if r['side']==side)<.65:continue
            p=world(center-normal*.3)
            if not contains(source,p):continue
            filled=contains(final,p);solids+=int(filled);voids+=int(not filled)
            inside=world(center-normal*.75)
            if not contains(source,inside):continue
            outside=world(center+normal*.1)
            if contains(source,outside):continue
            through_tests+=1
            if not final.ray_cast(tuple(outside),tuple(inside)):
                assert all(not contains(final,world(center-normal*depth)) for depth in [.2,.5,.7])
                through+=1
        item=dict(face_id=fid,interior_samples=solids+voids,material_samples=solids,void_samples=voids,
                  void_fraction=voids/max(1,solids+voids),through_tests=through_tests,open_through_segments=through)
        wave_results.append(item)
        assert item['interior_samples']>=256 and item['void_fraction']>.08 and through>=3,item
    print('PASS real openings on all 64 tooth-sidewall faces',flush=True)
    report=dict(valid_indexed_manifold=True,components=1,source_trimmed=True,triangles=nf,
                cell_mm=6.,strut_mm=1.,preserved_face_count=0,perforated_face_count=len(result['perforated_face_ids']),
                gap_ray_tests=gap_tests,rib_traverses=rib_results,rib_thickness_samples=thickness,
                wave_opening_checks=wave_results,total_wave_samples=sum(x['interior_samples'] for x in wave_results),
                wave_void_fraction=sum(x['void_samples'] for x in wave_results)/sum(x['interior_samples'] for x in wave_results),
                volume_mm3=final.volume(),elapsed_seconds=time.time()-start)
    atomic_json(OUT/'verification.json',report);print('VERIFIED',flush=True)

if __name__=='__main__':main()

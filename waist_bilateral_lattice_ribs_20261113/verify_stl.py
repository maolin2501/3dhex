"""Bounded-memory verification of every serialized STL triangle against PLY."""
from pathlib import Path
import struct,json,time
import numpy as np
OUT=Path(__file__).resolve().parent
started=time.time();ply=OUT/'bilateral_lattice_ribs_indexed.ply'
with ply.open('rb') as stream:
    nv=nf=None
    while True:
        line=stream.readline().decode('ascii').strip()
        if line.startswith('element vertex'):nv=int(line.split()[-1])
        if line.startswith('element face'):nf=int(line.split()[-1])
        if line=='end_header':break
    offset=stream.tell()
vertices=np.memmap(ply,dtype='<f4',offset=offset,shape=(nv,3),mode='r')
faces=np.memmap(ply,dtype=[('count','u1'),('v','<i4',(3,))],offset=offset+12*nv,shape=(nf,),mode='r')
stl=OUT/'bilateral_lattice_ribs_print.stl'
with stl.open('rb') as stream:
    stream.seek(80);assert struct.unpack('<I',stream.read(4))[0]==nf
assert stl.stat().st_size==84+nf*50
records=np.memmap(stl,dtype=[('normal','<f4',(3,)),('v','<f4',(3,3)),('attribute','<u2')],offset=84,shape=(nf,),mode='r')
zero_area=0;repeated=0;smallest=float('inf')
for i in range(0,nf,65536):
    f=faces[i:i+65536];rec=records[i:i+65536]
    assert (f['count']==3).all()
    expected=vertices[f['v']]
    assert np.array_equal(expected,rec['v']),i
    assert np.isfinite(rec['v']).all() and np.isfinite(rec['normal']).all()
    tri=expected.astype(np.float64)
    area2=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
    zero_area+=int((area2==0).sum());smallest=min(smallest,float(area2.min()))
    repeated+=int((np.all(expected[:,0]==expected[:,1],axis=1)|np.all(expected[:,1]==expected[:,2],axis=1)|np.all(expected[:,2]==expected[:,0],axis=1)).sum())
report=dict(triangles=nf,bytes=stl.stat().st_size,all_triangles_match_validated_indexed_mesh=True,
            finite_coordinates_and_normals=True,zero_area_triangles=zero_area,
            repeated_vertex_triangles=repeated,min_double_area_mm2=smallest,elapsed_seconds=time.time()-started)
(OUT/'stl_verification.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report),flush=True)

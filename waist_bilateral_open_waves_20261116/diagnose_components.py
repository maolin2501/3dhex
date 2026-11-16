"""Resume the exact pretrim checkpoint and inventory signed surface shells."""
from pathlib import Path
import sys,json,gc
import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent;sys.path.insert(0,str(ROOT))
from latticefill.core import FillParams,_finish_mesh
from latticefill.mesh_checkpoint import load_pretrim
from latticefill.cad_io import read_body
from latticefill.mesh_quality import clean_fragments
from latticefill.mesh_memory import FACE_BATCH
from latticefill.studio_jobs import atomic_json
from generate_revision import progress

def inventory(mesh):
    f=mesh.faces.astype(np.int32)
    graph=coo_matrix((np.ones(f.size,bool),(f.ravel(),np.roll(f,-1,axis=1).ravel())),shape=(len(mesh.vertices),)*2).tocsr()
    del f
    count,labels=connected_components(graph,directed=False);del graph;gc.collect()
    volumes=np.zeros(count);counts=np.zeros(count,np.int64)
    low=np.full((count,3),np.inf);high=-low.copy()
    for k in range(0,len(mesh.faces),FACE_BATCH):
        f=mesh.faces[k:k+FACE_BATCH];part=labels[f[:,0]];t=mesh.vertices[f]
        np.add.at(volumes,part,np.einsum('ij,ij->i',t[:,0],np.cross(t[:,1],t[:,2]))/6)
        np.add.at(counts,part,1)
        np.minimum.at(low,part,t.min(axis=1));np.maximum.at(high,part,t.max(axis=1))
    report=[dict(id=i,signed_volume_mm3=float(volumes[i]),faces=int(counts[i]),bounds=[low[i].tolist(),high[i].tolist()]) for i in range(count) if counts[i]]
    return labels,sorted(report,key=lambda x:abs(x['signed_volume_mm3']),reverse=True)

def main():
    saved=OUT/'trimmed';saved.mkdir(exist_ok=True)
    if (saved/'ready.json').exists():
        out=trimesh.Trimesh(np.load(saved/'vertices.npy',mmap_mode='c'),np.load(saved/'faces.npy',mmap_mode='c'),process=False)
    else:
        meta=json.loads((OUT/'pretrim/ready.json').read_text(encoding='utf8'))
        p=FillParams(**meta['params'],work_dir=str(OUT))
        body=read_body(OUT/'model',0);body.mesh.apply_translation(-body.mesh.bounds.mean(axis=0))
        raw,pitch,density=load_pretrim(OUT,p)
        out=_finish_mesh(raw,body.mesh,p,pitch,density,progress,pretrim_ready=True);del raw;gc.collect()
        out,cleanup=clean_fragments(out,1.)
        np.save(saved/'vertices.npy',np.asarray(out.vertices));np.save(saved/'faces.npy',np.asarray(out.faces,dtype=np.int32))
        atomic_json(saved/'ready.json',dict(stats=p.stats,initial_cleanup=cleanup))
    progress(90,'Inspecting signed volume of disconnected surface shells')
    labels,report=inventory(out);np.save(saved/'labels.npy',labels)
    atomic_json(OUT/'component_diagnosis.json',dict(components=report,positive=sum(x['signed_volume_mm3']>0 for x in report),negative=sum(x['signed_volume_mm3']<0 for x in report)))
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()

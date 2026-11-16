"""Measure real material islands and shortest local connection candidates."""
from pathlib import Path
import sys,json
import numpy as np
from scipy.spatial import cKDTree
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent;sys.path.insert(0,str(ROOT))
from latticefill.studio_jobs import atomic_json

def main():
    saved=OUT/'trimmed';v=np.load(saved/'vertices.npy',mmap_mode='r');labels=np.load(saved/'labels.npy')
    report=json.loads((OUT/'component_diagnosis.json').read_text(encoding='utf8'))
    assert report['negative']==0
    main_id=report['components'][0]['id'];main_vertices=np.asarray(v[labels==main_id])
    tree=cKDTree(main_vertices);plans=[]
    for part in report['components'][1:]:
        points=np.asarray(v[labels==part['id']]);distance,index=tree.query(points,k=1,workers=-1)
        order=np.argsort(distance);candidates=[]
        for i in order:
            a=points[i];b=main_vertices[index[i]]
            if candidates and min(np.linalg.norm(a-np.array(c['a'])) for c in candidates)<.25:continue
            candidates.append(dict(a=a.tolist(),b=b.tolist(),gap_mm=float(distance[i])))
            if len(candidates)==40:break
        plans.append(dict(component_id=part['id'],volume_mm3=part['signed_volume_mm3'],candidates=candidates))
        print(part['id'],round(part['signed_volume_mm3'],5),'nearest gap',round(float(distance.min()),5),flush=True)
    atomic_json(OUT/'connection_plan.json',dict(main_component_id=main_id,nominal_rod_diameter_mm=1.,islands=plans))

if __name__=='__main__':main()

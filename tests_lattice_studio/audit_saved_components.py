"""Read-only indexed connectivity inspection of a completed large GLB."""
import json
import sys
from pathlib import Path
import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def main():
    job=Path(sys.argv[1]);scene=trimesh.load_scene(job/'display.glb',process=False)
    transform,name=scene.graph[next(iter(scene.graph.nodes_geometry))]
    mesh=scene.geometry[name];f=np.asarray(mesh.faces,dtype=np.int32)
    graph=coo_matrix((np.ones(f.size,bool),(f.ravel(),np.roll(f,-1,axis=1).ravel())),
                     shape=(len(mesh.vertices),)*2).tocsr()
    del f
    count,labels=connected_components(graph,directed=False);del graph
    volumes=np.zeros(count);counts=np.zeros(count,np.int64)
    low=np.full((count,3),np.inf);high=np.full((count,3),-np.inf)
    for i in range(0,len(mesh.vertices),65536):
        vertices=mesh.vertices[i:i+65536]@transform[:3,:3].T+transform[:3,3]
        np.minimum.at(low,labels[i:i+65536],vertices)
        np.maximum.at(high,labels[i:i+65536],vertices)
    centre=mesh.bounds.mean(axis=0)
    for i in range(0,len(mesh.faces),65536):
        faces=mesh.faces[i:i+65536];part=labels[faces[:,0]]
        tri=mesh.vertices[faces]-centre
        signed=np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2]))/6
        np.add.at(volumes,part,signed);np.add.at(counts,part,1)
    result=[dict(component=int(i),faces=int(counts[i]),volume_mm3=float(volumes[i]),
                 bounds= [low[i].tolist(),high[i].tolist()],size_mm=(high[i]-low[i]).tolist())
            for i in np.argsort(-np.abs(volumes)) if counts[i]>0]
    destination=(Path(sys.argv[2]) if len(sys.argv)>2 else
                 Path('reference_meshes/new_model/atlas_20261110/final_components.json'))
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()

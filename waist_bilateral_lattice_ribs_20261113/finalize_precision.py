"""Round-trip serialized coordinates through the manifold's topology repair."""
from pathlib import Path
import sys,json,time,gc
import numpy as np
import trimesh
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent;sys.path.insert(0,str(ROOT))
from latticefill.mesh_quality import as_manifold,as_trimesh
from latticefill.mesh_memory import export_stl_batched,clean_components_batched
from latticefill.studio_jobs import atomic_json

def count_zero(mesh):
    total=0
    for i in range(0,len(mesh.faces),65536):
        tri=mesh.vertices[mesh.faces[i:i+65536]].astype(np.float32).astype(np.float64)
        total+=int((np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)==0).sum())
    return total

def main():
    started=time.time();result=json.loads((OUT/'result.json').read_text(encoding='utf8'))
    mesh=trimesh.load_mesh(OUT/'bilateral_lattice_ribs_indexed.ply',process=False)
    before=len(mesh.faces);solid=as_manifold(mesh);original_volume=solid.volume()
    report=dict(input_faces=before,input_zero_area_faces=count_zero(mesh),tolerance_mm=solid.get_tolerance())
    del mesh;gc.collect()
    out=as_trimesh(solid)
    report['constructor_output_faces']=len(out.faces)
    report['constructor_zero_area_faces']=count_zero(out)
    print('CONSTRUCTOR',json.dumps(report),flush=True)
    if report['constructor_zero_area_faces']:
        del out;gc.collect();solid=solid.simplify(2e-5);out=as_trimesh(solid)
        report['simplify_tolerance_mm']=max(2e-5,solid.get_tolerance())
    assert count_zero(out)==0,'Float32 degeneracy remains'
    after_volume=solid.volume()
    assert abs(after_volume-original_volume)<.01,(original_volume,after_volume)
    del solid;gc.collect()
    out,topology=clean_components_batched(out,.6)
    assert topology['components']==1,topology
    export_stl_batched(out,OUT/'bilateral_lattice_ribs_print.stl')
    pending=OUT/'bilateral_lattice_ribs_indexed.partial.ply'
    out.export(pending);pending.replace(OUT/'bilateral_lattice_ribs_indexed.ply')
    report.update(output_faces=len(out.faces),output_zero_area_faces=0,topology=topology,
                  original_volume_mm3=original_volume,output_volume_mm3=after_volume,
                  elapsed_seconds=time.time()-started)
    result['precision_cleanup']=report;result['triangles']=len(out.faces)
    result['bounds_mm']=out.bounds.tolist();result['topology']=topology
    result['stats']['faces']=len(out.faces);result['stats']['volume']=topology['volume']
    result['material_fraction']=topology['volume']/result['source_volume_mm3']
    atomic_json(OUT/'result.json',result);atomic_json(OUT/'precision_cleanup.json',report)
    atomic_json(OUT/'status.json',dict(state='geometry_ready',progress=100,message='Float32-safe geometry exported; final QA pending'))
    print('CLEANED',json.dumps(report),flush=True)

if __name__=='__main__':main()

"""Full-body atlas/BCC generation with open wave walls and retained ribs."""
from pathlib import Path
import sys,json,time,gc,traceback,subprocess
import numpy as np
import trimesh
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent;sys.path.insert(0,str(ROOT))
from latticefill.cad_io import read_body
from latticefill.core import FillParams,fill_solid
from latticefill.surface import PerforatedSurface
from latticefill.studio_jobs import validate_design,estimate,atomic_json,process_memory
from latticefill.mesh_quality import clean_fragments,as_manifold,as_trimesh
from latticefill.mesh_memory import clean_components_batched,release_cache

def progress(percent,message):
    atomic_json(OUT/'status.json',dict(state='running',progress=percent,message=message,memory=process_memory()))
    print(time.strftime('%H:%M:%S'),percent,message,flush=True)

def main():
    started=time.time();doc=json.loads((OUT/'design.json').read_text(encoding='utf8'))
    cfg=validate_design(doc['config']);body=read_body(OUT/'model',0)
    assert cfg['preserve_faces']==[] and cfg['shell']==0
    est=estimate(body.mesh.bounds,cfg);assert est['allowed'],est
    raw=np.load(OUT/'rib_midplanes.npz');ribs=trimesh.Trimesh(raw['vertices'],raw['faces'],process=False)
    center=body.mesh.bounds.mean(axis=0)
    active={f['id'] for f in body.face_info}
    assert set(doc['wave_face_ids'])<=active and len(active)==145
    surface=PerforatedSurface(body,active,translation=-center,chart_mode='atlas')
    body.mesh.apply_translation(-center);ribs.apply_translation(-center)
    p=FillParams(lattice='bcc',cell=6.,wall=1.,shell=0.,layer=6.,axis='z',pitch=.2,
                 region_depth=6.,max_voxels=400000000,boundary_blur=.6,smooth=2,
                 unselected='perforate',surface_mode='perforated',surface_pattern='hex',
                 surface_thickness=.8,surface_cell=0.,surface_width=0.,pore_rounding=0.,
                 junction_blend=0.,edge_width=0.,surface_continuity='auto',seam_style='atlas',
                 memory_mode='chunked',work_dir=str(OUT))
    progress(2,'Generating 6 mm BCC / 1 mm struts; all 145 CAD faces perforated; only rib planes preserved')
    out=fill_solid(body.mesh,p,preserve_faces=None,preserve_regions=[(ribs,.4)],
                   surface_field=surface,progress=progress)
    progress(89,'Checking connected geometry')
    out,initial_topology=clean_fragments(out,1.)
    if initial_topology['components']!=1:
        # Preserve an exact native checkpoint and finish locally in bounded,
        # separate processes; never discard real material islands.
        saved=OUT/'trimmed';saved.mkdir(exist_ok=True)
        np.save(saved/'vertices.npy',np.asarray(out.vertices))
        np.save(saved/'faces.npy',np.asarray(out.faces,dtype=np.int32))
        atomic_json(saved/'ready.json',dict(stats=p.stats,initial_cleanup=initial_topology))
        progress(90,'Material islands require local, source-clipped connections')
        return False
    release_cache(out);shift=np.array([0.,0.,-out.bounds[0,2]])
    out.apply_translation(shift)
    before_faces=len(out.faces)
    out.vertices=out.vertices.astype(np.float32).astype(np.float64)
    solid=as_manifold(out);del out;gc.collect()
    out=as_trimesh(solid);del solid;gc.collect()
    out,topology=clean_components_batched(out,0.)
    assert topology['components']==1 and topology['removed_fragments']==0,topology
    progress(95,'Saving float32-safe indexed mesh; print formats follow in a separate process')
    out.export(OUT/'open_wave_lattice_indexed.ply')
    p.stats.update(topology);p.stats['faces']=len(out.faces)
    result=dict(config=cfg,estimate=est,stats=p.stats,topology=topology,
                bounds_mm=out.bounds.tolist(),triangles=len(out.faces),
                print_translation_from_source=(-center+shift).tolist(),
                preserved_face_ids=[],wave_face_ids=doc['wave_face_ids'],
                perforated_face_ids=sorted(active),rib_count=35,rib_nominal_thickness_mm=.8,
                source_volume_mm3=doc['source_volume_mm3'],
                material_fraction=topology['volume']/doc['source_volume_mm3'],
                float32_cleanup=dict(input_faces=before_faces,output_faces=len(out.faces)),
                initial_topology=initial_topology,elapsed_seconds=time.time()-started,peak_memory=process_memory())
    atomic_json(OUT/'result.json',result)
    atomic_json(OUT/'status.json',dict(state='geometry_ready',progress=100,message='Indexed geometry ready; export and independent QA pending'))
    print('GEOMETRY READY',json.dumps({k:result[k] for k in ['triangles','topology','material_fraction','elapsed_seconds']}),flush=True)
    return True

if __name__=='__main__':
    try:
        complete=main()
        if not complete:
            gc.collect()
            for name in ['diagnose_components.py','plan_connections.py','connect_islands.py']:
                subprocess.run([sys.executable,str(OUT/name)],check=True)
    except Exception as exc:
        traceback.print_exc();atomic_json(OUT/'status.json',dict(state='failed',message=str(exc)));raise

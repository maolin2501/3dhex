"""Use the workspace's default Lattice Studio engine plus confirmed rib planes."""
from pathlib import Path
import sys,json,time,traceback
import numpy as np
import trimesh
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent
sys.path.insert(0,str(ROOT))
from latticefill.cad_io import read_body
from latticefill.core import FillParams,fill_solid
from latticefill.surface import PerforatedSurface
from latticefill.studio_jobs import validate_design,estimate,atomic_json,process_memory
from latticefill.mesh_quality import clean_fragments
from latticefill.mesh_memory import export_stl_batched,release_cache

def progress(percent,message):
    data=dict(state='running',progress=percent,message=message,memory=process_memory())
    atomic_json(OUT/'status.json',data)
    print(time.strftime('%H:%M:%S'),percent,message,flush=True)

def main():
    started=time.time();doc=json.loads((OUT/'design.json').read_text(encoding='utf8'))
    cfg=validate_design(doc['config']);body=read_body(OUT/'model',0)
    est=estimate(body.mesh.bounds,cfg);assert est['allowed'],est
    protected=body.selected_mesh(cfg['preserve_faces'])
    raw=np.load(OUT/'rib_midplanes.npz')
    ribs=trimesh.Trimesh(raw['vertices'],raw['faces'],process=False)
    center=body.mesh.bounds.mean(axis=0)
    active={f['id'] for f in body.face_info}-set(cfg['preserve_faces'])
    surface=PerforatedSurface(body,active,translation=-center,chart_mode='atlas')
    for m in [body.mesh,protected,ribs]:m.apply_translation(-center)
    p=FillParams(lattice='bcc',cell=4.,wall=.6,shell=.8,layer=4.,axis='z',pitch=.2,
                 region_depth=6.,max_voxels=400000000,boundary_blur=.6,smooth=2,
                 unselected='perforate',surface_mode='perforated',surface_pattern='hex',
                 surface_thickness=.8,surface_cell=0.,surface_width=0.,pore_rounding=0.,
                 junction_blend=0.,edge_width=0.,surface_continuity='auto',seam_style='atlas',
                 memory_mode='chunked',work_dir=str(OUT))
    progress(2,'Generating full-body default BCC lattice, preserved tooth walls and transverse ribs')
    out=fill_solid(body.mesh,p,preserve_faces=protected,
                   preserve_regions=[(protected,.8),(ribs,.4)],surface_field=surface,progress=progress)
    progress(89,'Checking connected components and removing only sub-feature numerical fragments')
    out,topology=clean_fragments(out,.6)
    p.stats.update(topology)
    if topology['components']!=1:
        # Keep reviewable geometry; do not silently delete real disconnected components.
        print('ATTENTION: independent components',topology['components'],flush=True)
    release_cache(out)
    progress(94,'Writing validated full-resolution print STL')
    shift=np.array([0.,0.,-out.bounds[0,2]])
    export_stl_batched(out,OUT/'bilateral_lattice_ribs_print.stl',translation=shift)
    # Indexed PLY preserves topology for fresh QA without STL vertex welding.
    out.apply_translation(shift)
    out.export(OUT/'bilateral_lattice_ribs_indexed.ply')
    result=dict(config=cfg,estimate=est,stats=p.stats,topology=topology,
                bounds_mm=out.bounds.tolist(),triangles=len(out.faces),
                print_translation_from_source=(-center+shift).tolist(),
                preserved_face_ids=cfg['preserve_faces'],rib_count=doc['rib_count'],
                rib_nominal_thickness_mm=.8,elapsed_seconds=time.time()-started,
                peak_memory=process_memory(),source_volume_mm3=doc['source_volume_mm3'],
                material_fraction=float(topology['volume']/doc['source_volume_mm3']))
    atomic_json(OUT/'result.json',result)
    atomic_json(OUT/'status.json',dict(state='done',progress=100,message='Complete',result=result))
    print('DONE',json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        traceback.print_exc()
        atomic_json(OUT/'status.json',dict(state='failed',message=str(exc)))
        raise

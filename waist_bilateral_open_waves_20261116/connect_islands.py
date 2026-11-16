"""Connect measured material islands locally, clipped to the original solid.

No faces are filled or deleted. Every accepted connector overlaps the main
body and its original island in positive volume and survives source clipping.
"""
from pathlib import Path
import sys,json,gc,time
import numpy as np
import trimesh,manifold3d as mf
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent;sys.path.insert(0,str(ROOT))
from latticefill.cad_io import read_body
from latticefill.mesh_quality import as_manifold,as_trimesh
from latticefill.mesh_memory import clean_components_batched
from latticefill.studio_jobs import atomic_json,process_memory,validate_design,estimate
from latticefill.continuity import MeshProbe
from generate_revision import progress

def cylinder_between(a,b,offset):
    direction=(b-a)/np.linalg.norm(b-a);start=a-.5*direction+offset;end=b+.5*direction+offset
    u=np.cross(direction,[0,0,1] if abs(direction[2])<.9 else [0,1,0]);u/=np.linalg.norm(u)
    v=np.cross(direction,u);rotation=np.column_stack((u,v,direction))
    length=float(np.linalg.norm(end-start));matrix=np.column_stack((rotation,start))
    rod=mf.Manifold.cylinder(length,.5,circular_segments=32).transform(matrix)
    inverse=np.column_stack((rotation.T,-rotation.T@start))
    return rod,start,end,inverse

def overlap_profile(overlap,inverse,length):
    local=overlap.transform(inverse)
    zs=np.linspace(.025,length-.025,max(3,int(np.ceil(length/.05))))
    areas=np.array([local.slice(float(z)).area() for z in zs])
    active=areas>=.1;run=best=0
    for value in active:
        run=run+1 if value else 0;best=max(best,run)
    span=max(0,best-1)*float(zs[1]-zs[0])
    return dict(overlap_span_at_0_1_mm2_mm=span,max_section_mm2=float(areas.max()))

def main():
    started=time.time();saved=OUT/'trimmed';ready=json.loads((saved/'ready.json').read_text(encoding='utf8'))
    plan=json.loads((OUT/'connection_plan.json').read_text(encoding='utf8'))
    doc=json.loads((OUT/'design.json').read_text(encoding='utf8'))
    body=read_body(OUT/'model',0);center=body.mesh.bounds.mean(axis=0);body.mesh.apply_translation(-center)
    source=as_manifold(body.mesh);probe=MeshProbe(body.mesh)
    verts=np.load(saved/'vertices.npy',mmap_mode='c');faces=np.load(saved/'faces.npy',mmap_mode='c');labels=np.load(saved/'labels.npy')
    face_labels=labels[faces[:,0]];main_faces=faces[face_labels==plan['main_component_id']]
    used=np.unique(main_faces);remap=np.full(len(verts),-1,np.int32);remap[used]=np.arange(len(used))
    main_solid=as_manifold(trimesh.Trimesh(verts[used],remap[main_faces],process=False));del main_faces,remap,used;gc.collect()
    connections=[];rods=[];islands=[]
    for entry in plan['islands']:
        f=faces[face_labels==entry['component_id']];used,indices=np.unique(f,return_inverse=True)
        island=as_manifold(trimesh.Trimesh(verts[used],indices.reshape(-1,3),process=False));islands.append(island)
        success=False
        for candidate_number,candidate in enumerate(entry['candidates']):
            a=np.array(candidate['a']);b=np.array(candidate['b'])
            _,normal,_=probe.closest((a+b)/2)
            for inward in [.25,.5,0.]:
                raw,start,end,inverse=cylinder_between(a,b,-inward*normal)
                clipped=raw^source
                if clipped.is_empty() or len(clipped.decompose())!=1:continue
                depth=probe.depth(np.linspace(start,end,25))
                if depth.min()<-.01:continue
                island_intersection=clipped^island;overlap_island=island_intersection.volume()
                if overlap_island<.02:continue
                main_intersection=clipped^main_solid;overlap_main=main_intersection.volume()
                if overlap_main<.02:continue
                local=clipped.transform(inverse);length=np.linalg.norm(end-start)
                areas=[local.slice(float(z)).area() for z in np.linspace(.1,length-.1,max(15,int(np.ceil(length/.1))))]
                if min(areas)<.4:continue
                island_profile=overlap_profile(island_intersection,inverse,length)
                if island_profile['overlap_span_at_0_1_mm2_mm']<.2:continue
                main_profile=overlap_profile(main_intersection,inverse,length)
                if main_profile['overlap_span_at_0_1_mm2_mm']<.2:continue
                rods.append(clipped)
                connections.append(dict(component_id=entry['component_id'],island_volume_mm3=entry['volume_mm3'],
                    closest_gap_mm=candidate['gap_mm'],candidate_number=candidate_number,
                    start=start.tolist(),end=end.tolist(),length_mm=float(length),diameter_mm=1.,
                    inward_offset_mm=inward,source_clipped=True,connector_components=1,
                    source_centerline_min_depth_mm=float(depth.min()),
                    overlap_island_mm3=overlap_island,overlap_main_mm3=overlap_main,
                    island_overlap_profile=island_profile,main_overlap_profile=main_profile,
                    min_sampled_cross_section_mm2=min(areas),cross_sections_mm2=areas,volume_mm3=clipped.volume()))
                print('CONNECTED',entry['component_id'],'gap',round(candidate['gap_mm'],3),'area',round(min(areas),3),flush=True)
                success=True;break
            if success:break
        if not success:raise ValueError(f'No acceptable internal connection for island {entry["component_id"]}')
    atomic_json(OUT/'connection_verification.json',dict(connections=connections,state='connectors_verified'))
    progress(92,'Union of locally verified connectors and original material islands')
    original_volume=ready['initial_cleanup']['volume']
    merged=mf.Manifold.batch_boolean([main_solid]+islands+rods,mf.OpType.Add)
    assert merged.status()==mf.Error.NoError
    out=as_trimesh(merged);del merged,main_solid,islands,rods;gc.collect()
    out,connected=clean_components_batched(out,0.)
    assert connected['components']==1 and connected['removed_fragments']==0,connected
    shift=np.array([0.,0.,-out.bounds[0,2]]);out.apply_translation(shift);before_faces=len(out.faces)
    out.vertices=out.vertices.astype(np.float32).astype(np.float64)
    solid=as_manifold(out);del out;gc.collect();out=as_trimesh(solid);del solid;gc.collect()
    out,topology=clean_components_batched(out,0.)
    assert topology['components']==1 and topology['removed_fragments']==0,topology
    progress(97,'Saving connected float32-safe indexed mesh')
    out.export(OUT/'open_wave_lattice_indexed.ply')
    added=topology['volume']-original_volume
    assert 0<added<100,added
    repair=dict(connections=connections,state='verified',input_material_components=40,output_material_components=1,
                retained_original_islands=39,added_material_mm3=added,added_fraction=added/original_volume)
    atomic_json(OUT/'connection_verification.json',repair)
    cfg=validate_design(doc['config']);stats=ready['stats'];stats.update(topology);stats['faces']=len(out.faces)
    stats['routing_statistics_scope']='before_local_connectivity_repair'
    stats['final_attachment_mode']='atlas_bcc_with_39_local_source_clipped_struts'
    result=dict(config=cfg,estimate=estimate(body.mesh.bounds,cfg),stats=stats,topology=topology,
                bounds_mm=out.bounds.tolist(),triangles=len(out.faces),print_translation_from_source=(-center+shift).tolist(),
                preserved_face_ids=[],wave_face_ids=doc['wave_face_ids'],perforated_face_ids=sorted(f['id'] for f in body.face_info),
                rib_count=35,rib_nominal_thickness_mm=.8,source_volume_mm3=doc['source_volume_mm3'],
                material_fraction=topology['volume']/doc['source_volume_mm3'],
                float32_cleanup=dict(input_faces=before_faces,output_faces=len(out.faces)),initial_topology=ready['initial_cleanup'],
                local_connections_count=len(connections),local_connections_added_volume_mm3=added,
                postprocess_elapsed_seconds=time.time()-started,peak_memory=process_memory())
    atomic_json(OUT/'result.json',result)
    atomic_json(OUT/'status.json',dict(state='geometry_ready',progress=100,message='Connected indexed mesh saved; independent QA pending'))
    print('GEOMETRY READY',json.dumps({k:result[k] for k in ['triangles','topology','material_fraction']}),flush=True)

if __name__=='__main__':main()

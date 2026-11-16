"""Separately saved tolerance-limited simplification of the final lattice."""
from pathlib import Path
import sys,json,time,gc,shutil,hashlib
import numpy as np
import trimesh
import manifold3d as mf
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent
SOURCE=ROOT/'waist_bilateral_lattice_ribs_20261113'
sys.path.insert(0,str(ROOT))
from latticefill.mesh_quality import as_manifold,as_trimesh
from latticefill.mesh_memory import export_stl_batched,clean_components_batched
from latticefill.studio_jobs import atomic_json,process_memory

def state(stage,**kw):
    data=dict(stage=stage,**kw,memory=process_memory())
    atomic_json(OUT/'status.json',data);print(json.dumps(data),flush=True)

def describe(s):
    return dict(triangles=s.num_tri(),vertices=s.num_vert(),
                euler=s.num_vert()-s.num_edge()+s.num_tri(),volume_mm3=s.volume())

def zero_faces(mesh):
    total=0
    for i in range(0,len(mesh.faces),65536):
        t=mesh.vertices[mesh.faces[i:i+65536]].astype(np.float32).astype(np.float64)
        total+=int((np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)==0).sum())
    return total

def load_solid(path):
    # The PLY already contains float32 coordinates. Avoid Trimesh's full-file
    # parsing and 64-bit index copies before native manifold construction.
    with path.open('rb') as f:
        while True:
            line=f.readline().decode('ascii').strip()
            if line.startswith('element vertex'):nv=int(line.split()[-1])
            if line.startswith('element face'):nf=int(line.split()[-1])
            if line=='end_header':break
        offset=f.tell()
    vertices=np.memmap(path,dtype='<f4',offset=offset,shape=(nv,3),mode='r')
    faces=np.memmap(path,dtype=[('count','u1'),('v','<i4',(3,))],offset=offset+12*nv,shape=(nf,),mode='r')
    data=mf.Mesh(np.array(vertices,dtype=np.float32,copy=True,order='C'),
                 np.ascontiguousarray(faces['v'],dtype=np.uint32))
    state('constructing_manifold',triangles=nf,vertices=nv)
    solid=mf.Manifold(data)
    assert solid.status()==mf.Error.NoError and not solid.is_empty()
    return solid

def topology_reduce(path,tolerance):
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    reader=vtk.vtkPLYReader();reader.SetFileName(str(path));reader.Update()
    poly=reader.GetOutput();nf=poly.GetNumberOfCells();nv=poly.GetNumberOfPoints()
    source_check=json.loads((SOURCE/'verification.json').read_text(encoding='utf8'))
    assert nf==source_check['triangles'] and source_check['valid_indexed_manifold']
    before=dict(triangles=nf,vertices=nv,euler=nv-nf//2,volume_mm3=source_check['volume_mm3'])
    state('simplifying_preserve_topology',before=before,tolerance_mm=tolerance)
    dec=vtk.vtkDecimatePro();dec.SetInputData(poly);dec.SetTargetReduction(.9)
    dec.PreserveTopologyOn();dec.SplittingOff();dec.PreSplitMeshOff()
    dec.BoundaryVertexDeletionOff();dec.AccumulateErrorOn()
    dec.SetErrorIsAbsolute(1);dec.SetAbsoluteError(tolerance)
    dec.SetOutputPointsPrecision(vtk.vtkAlgorithm.DOUBLE_PRECISION)
    last=[-1]
    def progress(obj,event):
        step=int(obj.GetProgress()*20)
        if step>last[0]:state('simplifying_preserve_topology',percent=step*5);last[0]=step
    dec.AddObserver(vtk.vtkCommand.ProgressEvent,progress);dec.Update()
    result=dec.GetOutput()
    mesh=trimesh.Trimesh(np.array(vtk_to_numpy(result.GetPoints().GetData())),
                         np.array(vtk_to_numpy(result.GetPolys().GetConnectivityArray())).reshape(-1,3),process=False)
    del result,dec,poly,reader;gc.collect()
    return mesh,before

def main():
    started=time.time();state('loading_conservative')
    solid=load_solid(SOURCE/'bilateral_lattice_ribs_indexed.ply');before=describe(solid)
    trials=[]
    for tolerance in [.0001]:
        state('conservative_simplification',tolerance_mm=tolerance)
        reduced=solid.simplify(tolerance);after=describe(reduced)
        trials.append(dict(tolerance_mm=tolerance,after=after))
        state('trial_result',trial=trials[-1],expected_euler=before['euler'])
        if after['euler']==before['euler']:break
        del reduced;gc.collect()
    else:raise RuntimeError('No topology-preserving tolerance found')
    del solid;gc.collect()
    assert reduced.status()==mf.Error.NoError and not reduced.is_empty()
    assert before['euler']==after['euler'],(before,after)
    assert after['triangles']<before['triangles']
    relative_volume_error=abs(after['volume_mm3']/before['volume_mm3']-1)
    assert relative_volume_error<.015,relative_volume_error
    state('checking_float32',after=after,relative_volume_error=relative_volume_error)
    mesh=as_trimesh(reduced);del reduced;gc.collect()
    # Validate the same float32 positions used by STL, then publish this topology.
    mesh.vertices=mesh.vertices.astype(np.float32).astype(np.float64)
    serialized=as_manifold(mesh);del mesh;gc.collect()
    serialized_info=describe(serialized)
    assert serialized_info['euler']==before['euler']
    mesh=as_trimesh(serialized);del serialized;gc.collect()
    assert zero_faces(mesh)==0
    mesh,topology=clean_components_batched(mesh,0.)
    assert topology['components']==1 and topology['removed_fragments']==0,topology
    state('exporting',after=serialized_info,topology=topology)
    # Save compact indexed geometry first and exit before large STL export.
    # Native simplification's peak allocation can temporarily exhaust disk
    # available for paging even after its working arrays have been released.
    mesh.export(OUT/'bilateral_lattice_ribs_indexed.ply')
    report=dict(source_folder=str(SOURCE),method='manifold_adaptive_topology_checked',tolerance_mm=tolerance,trials=trials,
                before=before,after=serialized_info,topology=topology,
                relative_volume_error=relative_volume_error,
                source_stl_bytes=(SOURCE/'bilateral_lattice_ribs_print.stl').stat().st_size,
                simplified_stl_bytes=84+50*len(mesh.faces),
                face_reduction_fraction=1-len(mesh.faces)/before['triangles'],
                float32_zero_area_faces=0,elapsed_seconds=time.time()-started)
    result=json.loads((SOURCE/'result.json').read_text(encoding='utf8'))
    result['simplification']=report;result['triangles']=len(mesh.faces)
    result['topology']=topology;result['bounds_mm']=mesh.bounds.tolist()
    result['stats']['faces']=len(mesh.faces);result['stats']['volume']=topology['volume']
    result['material_fraction']=topology['volume']/result['source_volume_mm3']
    atomic_json(OUT/'result.json',result);atomic_json(OUT/'simplification.json',report)
    for name in ['design.json','support_layout.png','verify_lattice.py','verify_stl.py','render_final.py']:
        if name.endswith('.py') and (OUT/name).exists():continue
        shutil.copy2(SOURCE/name,OUT/name)
    shutil.copytree(SOURCE/'model',OUT/'model',dirs_exist_ok=True)
    state('ready_for_verification',report=report)

if __name__=='__main__':main()

"""Read-back validation: routed centerlines must remain inside exported STL.

Uses VTK's native STL reader and implicit distance queries, not Trimesh's
full-grid contains cache. Does not change or repair the result.
"""
import json
import sys
import time
from pathlib import Path
import numpy as np
from latticefill.cad_io import read_body
from latticefill.core import FillParams
from latticefill.surface import PerforatedSurface
from latticefill.studio_jobs import atomic_json, process_memory


def main():
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
    root=Path('.lattice-studio').resolve(); job=root/sys.argv[1]
    result=json.loads((job/'design.json').read_text(encoding='utf-8')); config=result['config']
    body=read_body(root/result['model_id'],result['source_body'])
    centre=body.mesh.bounds.mean(axis=0)
    active=set(f['id'] for f in body.face_info)-set(config['preserve_faces'])
    if config['scope']=='faces' and config['unselected']=='solid':active &= set(config['design_faces'])
    surface=PerforatedSurface(body,active,translation=-centre,chart_mode=config.get('seam_style'))
    args={k:v for k,v in config.items() if k in FillParams.__dataclass_fields__ and k!='stats'}
    p=FillParams(**args);surface.prepare_edges(config['pitch'],p)
    if config.get('seam_style') in ('aligned','vertex'):
        keys=['routed_terminals','omitted_boundary_members','graph_segments','aligned_phase']
        if config.get('seam_style')=='vertex':keys+=['vertex_terminal_ids','removed_old_outer_members']
        for key in keys:
            if p.stats.get(key)!=result['stats'].get(key):
                raise ValueError(f'重建连线图与生成记录不同 ({key})，不能用当前算法审计该历史结果')
    paths=list(surface.network.terminals)
    if surface.stitches is not None:
        paths.extend((a,b,curve) for (a,b),curve in zip(surface.stitches.anchors,surface.stitches.paths))
    extra_stitches=len(paths)-len(surface.network.terminals)
    from .adaptive_seam_probes import seam_crossings
    web_paths, web_pairs=seam_crossings(surface,p)
    paths.extend(web_paths)
    samples=np.vstack([curve for _,_,curve in paths])+centre
    started=time.perf_counter()
    reader=vtk.vtkSTLReader();reader.SetFileName(str(job/'result.stl'));reader.MergingOn();reader.Update()
    function=vtk.vtkImplicitPolyDataDistance();function.SetInput(reader.GetOutput())
    distances=[]
    for k in range(0,len(samples),65536):
        points=vtk.vtkPoints();points.SetData(numpy_to_vtk(np.ascontiguousarray(samples[k:k+65536]),deep=False))
        data=vtk.vtkPolyData();data.SetPoints(points)
        query=vtk.vtkSampleImplicitFunctionFilter();query.SetInputData(data);query.SetImplicitFunction(function)
        query.ComputeGradientsOff();query.Update()
        distances.append(-vtk_to_numpy(query.GetOutput().GetPointData().GetScalars()).copy())
    normal_depth=np.concatenate(distances)
    # Closest-triangle pseudonormals can give sign flips at tiny, quantized
    # STL triangles. Classify with orientation-independent ray parity instead;
    # keep the nearest-surface distance only as a magnitude.
    all_points=vtk.vtkPoints();all_points.SetData(numpy_to_vtk(np.ascontiguousarray(samples),deep=False))
    all_data=vtk.vtkPolyData();all_data.SetPoints(all_points)
    enclosed=vtk.vtkSelectEnclosedPoints();enclosed.SetInputData(all_data)
    enclosed.SetSurfaceData(reader.GetOutput());enclosed.SetTolerance(1e-7);enclosed.CheckSurfaceOff();enclosed.Update()
    inside=vtk_to_numpy(enclosed.GetOutput().GetPointData().GetArray('SelectedPoints')).astype(bool)
    depth=np.where(inside,abs(normal_depth),-abs(normal_depth))
    # A signed distance is 1-Lipschitz. If min(endpoint depth)-L/2 is
    # positive, the whole segment between adjacent samples is in material,
    # not just the sampled positions. The generated graph uses this polyline.
    lower_bounds=[];offset=0;failures=[]
    for index,(root_point,target,curve) in enumerate(paths):
        local=depth[offset:offset+len(curve)];offset+=len(curve)
        length=np.linalg.norm(np.diff(curve,axis=0),axis=1)
        bounds=np.minimum(local[:-1],local[1:])-length/2
        lower_bounds.append(bounds)
        if (bounds<=0).any():
            failures.append(dict(terminal=index,root=(root_point+centre).tolist(),target=(target+centre).tolist(),
                points=(curve+centre).tolist(),depths=local.tolist()))
    lower_bounds=np.concatenate(lower_bounds)
    audit=dict(job=job.name,samples=len(depth),routed_terminals=len(surface.network.terminals),
        surface_core_connections=extra_stitches,
        adaptive_seam_crossings=len(web_paths),
        adaptive_seam_face_pairs=[dict(faces=list(pair),crossings=web_pairs.count(pair)) for pair in sorted(set(web_pairs))],
        inside_method='ray_parity',normal_sign_disagreements=int(np.count_nonzero(inside != (normal_depth>0))),
        outside_samples=int(np.count_nonzero(depth < -.002)),minimum_material_depth_mm=float(depth.min()),
        nonpositive_segment_bounds=int(np.count_nonzero(lower_bounds <= 0)),
        minimum_segment_clearance_bound_mm=float(lower_bounds.min()),
        failures=failures,
        depth_percentiles_mm=np.percentile(depth,[0,1,50,99,100]).tolist(),
        time_seconds=time.perf_counter()-started,memory=process_memory(),graph_stats=surface.network.stats)
    if config.get('seam_style')=='vertex':
        from .vertex_audit import audit as audit_vertex
        audit['vertex_audit']=audit_vertex(surface,p,reader.GetOutput(),centre)
    atomic_json(job/'continuity_audit.json',audit)
    print(json.dumps(audit,ensure_ascii=False),flush=True)
    if audit['outside_samples'] or audit['nonpositive_segment_bounds'] or not audit.get('vertex_audit',{}).get('passed',True):
        raise SystemExit('导出后的转接中心线连续性未通过保守界检查，需要检查')


if __name__=='__main__':main()

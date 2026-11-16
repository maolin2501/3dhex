"""Vertex identity, side-contact and removed-shaft read-back checks."""
import numpy as np


def audit(surface,p,poly,centre):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk,vtk_to_numpy
    a=surface.adaptive;n=surface.network
    error=0.
    for target in a.vertices:
        distances=a.tree.query(target,k=3)[0]
        error=max(error,float(np.ptp(distances)))
    invalid=0;side_contacts=0;core_exposed=0
    for root,target,curve in n.terminals:
        index=a.vertex_id(target)
        port=a.inner_vertices[index]
        split=np.linalg.norm(curve-port,axis=1).argmin()
        invalid+=int(np.linalg.norm(curve[split]-port)>1e-7)
        side_contacts+=int(not a.contact_safe(curve[split:],[target]))
        core_exposed+=int(not a.core_safe(curve[:split+1]))
    # Select old-shaft samples conservatively separated from every legitimate
    # source of material: the new graph, sheet, preserved layers and real rims.
    points=np.vstack(n.replaced_paths)[::2]
    safety=p.junction_blend+2*p.pitch
    keep=(n.field(points)<-safety)&(a.source.depth(points)>.1)
    depth=a.face_probe.depth(points)
    sheet=np.minimum(a.field(points),p.surface_thickness-depth)
    keep &= sheet < -safety
    if surface.edge_tree is not None:
        keep &= surface.edge_tree.query(points)[0]>p.edge_width+safety
    points=points[keep]
    if p.shell:
        from latticefill.continuity import MeshProbe
        import trimesh
        meshes=[m for fid,m in surface.meshes.items() if fid not in surface.active]
        if meshes:
            protected=MeshProbe(trimesh.util.concatenate(meshes))
            points=points[abs(protected.depth(points))>p.shell+safety]
    # Whole-job sampling is bounded; no voxel-scale all-pairs allocation.
    if len(points)>100000:points=points[np.linspace(0,len(points)-1,100000,dtype=int)]
    unexpected=0
    if len(points):
        vpoints=vtk.vtkPoints();vpoints.SetData(numpy_to_vtk(np.ascontiguousarray(points+centre),deep=False))
        data=vtk.vtkPolyData();data.SetPoints(vpoints)
        select=vtk.vtkSelectEnclosedPoints();select.SetInputData(data);select.SetSurfaceData(poly)
        select.SetTolerance(1e-7);select.CheckSurfaceOff();select.Update()
        unexpected=int(vtk_to_numpy(select.GetOutput().GetPointData().GetArray('SelectedPoints')).sum())
    result=dict(max_three_site_distance_spread_mm=error,invalid_vertex_paths=invalid,
        side_contact_paths=side_contacts,exposed_core_paths=core_exposed,
        old_shaft_void_samples=len(points),unexpected_old_shaft_material_samples=unexpected)
    result['passed']=bool(error<a.stats['vertex_tolerance_mm'] and not(invalid or side_contacts or core_exposed or unexpected))
    return result

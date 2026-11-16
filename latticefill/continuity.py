"""Boundary-conforming BCC graph. All preprocessing is bounded by segment batches.

Unlike a depth-clipped periodic field, every active-face terminal is routed to
material in the perforated sheet. The analytic BCC field remains in the core.
"""
from __future__ import annotations

import itertools
import numpy as np
from scipy.spatial import cKDTree

# Preprocessing must not undo the disk-backed field's bounded-memory design.
MAX_GRAPH_CELLS = 2_000_000
MAX_GRAPH_SAMPLES = 5_000_000


class MeshProbe:
    def __init__(self, mesh, face_ids=None):
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
        self.vtk = vtk
        points = vtk.vtkPoints()
        points.SetData(numpy_to_vtk(np.asarray(mesh.vertices), deep=True))
        cells = vtk.vtkCellArray()
        cells.SetData(numpy_to_vtkIdTypeArray(np.arange(len(mesh.faces)+1, dtype=np.int64)*3, deep=True),
                      numpy_to_vtkIdTypeArray(mesh.faces.ravel().astype(np.int64), deep=True))
        self.poly = vtk.vtkPolyData(); self.poly.SetPoints(points); self.poly.SetPolys(cells)
        self.function = vtk.vtkImplicitPolyDataDistance(); self.function.SetInput(self.poly)
        self.locator = vtk.vtkStaticCellLocator(); self.locator.SetDataSet(self.poly); self.locator.BuildLocator()
        self.normals = np.asarray(mesh.face_normals).copy()
        self.ids = np.arange(len(mesh.faces)) if face_ids is None else np.asarray(face_ids)

    def depth(self, points):
        """True triangle signed distance, positive inside (not voxel seed distance)."""
        from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
        if len(points) < 32:
            return np.array([-self.function.EvaluateFunction(point) for point in points])
        result = np.empty(len(points))
        for i in range(0, len(points), 65536):
            pts = self.vtk.vtkPoints(); pts.SetData(numpy_to_vtk(np.ascontiguousarray(points[i:i+65536]), deep=False))
            data = self.vtk.vtkPolyData(); data.SetPoints(pts)
            sample = self.vtk.vtkSampleImplicitFunctionFilter()
            sample.SetInputData(data); sample.SetImplicitFunction(self.function); sample.ComputeGradientsOff(); sample.Update()
            result[i:i+65536] = -vtk_to_numpy(sample.GetOutput().GetPointData().GetScalars())
        return result

    def closest(self, point):
        out = [0., 0., 0.]
        cell, sub, dist = self.vtk.reference(0), self.vtk.reference(0), self.vtk.reference(0.)
        self.locator.FindClosestPoint(point, out, cell, sub, dist)
        index = int(cell)
        return np.asarray(out), self.normals[index], int(self.ids[index])

    def normals_at(self, points):
        """Normals of actual closest triangles, including exactly-on-face points.

        Distance gradients near zero can amplify roundoff into arbitrary
        directions. Query cell IDs instead; no surface-size working copies.
        """
        result=np.empty((len(points),3),float)
        out=[0.,0.,0.]
        cell,sub,dist=self.vtk.reference(0),self.vtk.reference(0),self.vtk.reference(0.)
        for i,point in enumerate(points):
            self.locator.FindClosestPoint(point,out,cell,sub,dist)
            result[i]=self.normals[int(cell)]
        return result


def chart_scales(chart, pattern, cell):
    su = sv = 1.
    if chart['kind'] in ('cylinder', 'torus'):
        circumference = 2*np.pi*chart['radius']
        su = max(3, round(circumference/cell))*cell/circumference
        if chart.get('v_periodic'):
            period = np.sqrt(3)*cell if pattern == 'hex' else cell
            circumference_v = 2*np.pi*chart['minor_radius']
            sv = max(1, round(circumference_v/period))*period/circumference_v
    return su, sv


def chart_point(uv, chart, scales):
    """Inverse of the same scaled chart used by pattern_field, including tori."""
    u, v = np.asarray(uv)/scales
    origin, e1, e2 = (np.asarray(chart[k]) for k in ('origin', 'u', 'v'))
    if chart['kind'] in ('plane', 'projection'):
        return origin+u*e1+v*e2
    theta = u/chart['radius']
    radial = np.cos(theta)*e1+np.sin(theta)*e2
    axis = np.asarray(chart['axis'])
    if chart['kind'] == 'cylinder':
        return origin+chart['radius']*radial+v*axis
    angle = v/chart['minor_radius']+chart['v_mid']
    return origin+(chart['major_radius']+chart['minor_radius']*np.cos(angle))*radial+chart['minor_radius']*np.sin(angle)*axis


def web_target(uv, pattern, cell):
    """Nearest rib centreline in pattern coordinates, not nearest pore centre."""
    from .lattices import _hex_local
    uv = np.asarray(uv)
    if pattern == 'hex':
        x, y, _ = _hex_local(uv[0:1], uv[1:2], cell)
        local = np.array([x[0], y[0]])
        a = np.arange(6)*np.pi/3+np.pi/6
        vertices = np.column_stack((np.cos(a), np.sin(a)))*cell/np.sqrt(3)
        candidates = []
        for start, end in zip(vertices, np.roll(vertices, -1, axis=0)):
            edge = end-start
            candidates.append(start+np.clip(np.dot(local-start, edge)/np.dot(edge, edge), 0, 1)*edge)
        target = min(candidates, key=lambda p: np.linalg.norm(p-local))
        return uv+target-local
    if pattern == 'square':
        local = uv-cell*np.round(uv/cell)
        target = uv.copy(); axis = np.argmin(abs(local)); target[axis] -= local[axis]
        return target
    # Circle holes: the nearest bisector of the pore array is always material.
    local = uv-cell*np.round(uv/cell)
    axis = np.argmax(abs(local)); target = uv.copy()
    target[axis] += (1 if local[axis] >= 0 else -1)*cell/2-local[axis]
    return target


def bezier(a, b, c, d, step):
    # Bound the derivative of the Bezier curve, not just its total length;
    # uniform t is not uniform arc length.
    length = 3*max(np.linalg.norm(b-a),np.linalg.norm(c-b),np.linalg.norm(d-c))
    t = np.linspace(0, 1, max(2, int(np.ceil(length/step))+1))[:, None]
    return (1-t)**3*a+3*(1-t)**2*t*b+3*(1-t)*t*t*c+t**3*d


class BoundaryBCC:
    def __init__(self, surface, p, pitch):
        from .surface import chart_uv, pattern_field
        self.radius = p.wall/2
        self.band_depth = p.surface_thickness+2*p.wall+2*pitch
        self.step = min(pitch/2, p.wall/6, .15)
        self.paths = []
        self.terminals = []
        self.sample_count = 0
        self.stats = dict(routed_terminals=0, seam_anchors=0, graph_segments=0,
                          unresolved_terminals=0, continuity_mode='boundary_bcc')
        self.aligned=p.seam_style in ('aligned','vertex','atlas')
        self.vertex=p.seam_style in ('vertex','atlas')
        self.replaced_paths=[]
        self.replacement_sample_count=0
        self.replacement_radius=self.radius+2*pitch+self.step
        self.grazing_removed=0
        self.straight_angles=[]
        self.boundary_members=0
        self.omitted_members=0
        aligned_edges=[]
        if self.aligned:
            self.band_depth=p.surface_thickness+np.sqrt(3)*p.cell/2+p.wall
        source = (surface.anchor_sampler.source if surface.anchor_sampler is not None
                  else MeshProbe(surface.source_mesh, surface.source_face_ids))
        probes = {}
        cell = p.surface_cell or p.cell; width = p.surface_width or p.wall

        def anchor(point):
            if surface.anchor_sampler is not None:
                return surface.anchor_sampler.anchor(point)
            closest, normal, fid = source.closest(point)
            if fid not in surface.active:
                return None
            probe = probes.setdefault(fid, None)
            if probe is None:
                probe = probes[fid] = MeshProbe(surface.meshes[fid])
            chart = surface.charts[fid]; scales = chart_scales(chart, p.surface_pattern, cell)
            uv = np.array(chart_uv(closest[None], chart)).ravel()*scales
            target_uv = web_target(uv, p.surface_pattern, cell)
            guess = closest if chart['kind'] == 'projection' else chart_point(target_uv, chart, scales)
            target, normal, _ = probe.closest(guess)
            # Projection charts need their height restored on the actual face.
            if chart['kind'] == 'projection':
                for _ in range(10):
                    actual = np.array(chart_uv(target[None], chart)).ravel()*scales
                    delta = (target_uv-actual)/scales
                    if np.linalg.norm(delta) < 1e-5: break
                    # Solve in the local tangent plane: projecting an orthogonal
                    # chart displacement repeatedly converges too slowly on
                    # steep freeform patches and can choose the wrong branch.
                    basis = np.array([chart['u'], chart['v'], normal])
                    step = np.linalg.lstsq(basis, [delta[0],delta[1],0.], rcond=None)[0]
                    step *= min(1., cell/max(np.linalg.norm(step),1e-9))
                    target, normal, _ = probe.closest(target+step)
            material = pattern_field(target[None], chart, p.surface_pattern, cell, width, p.pore_rounding)[0]
            if material < min(width/8, pitch/2):
                # A trimmed chart can put its nominal web outside the CAD face.
                # The explicit shared seam rail supplies a valid fallback anchor.
                if surface.seam_tree is None:
                    raise ValueError('曲面边界找不到连续孔边筋；请扩大边界加强宽度或分割设计面')
                distance, index = surface.seam_tree.query(closest)
                if distance > cell:
                    raise ValueError(f'局部曲面投影偏差过大，无法可靠连接孔边筋 (face={fid}, uv={uv.tolist()}, target_uv={target_uv.tolist()}, material={material}, seam_distance={distance}, point={closest.tolist()}, chart={chart})')
                target = surface.seam_tree.data[index].copy()
                _, normal, _ = probe.closest(target)
                self.stats['seam_anchors'] += 1
            # Interior-side anchor lies within the sheet, never outside the CAD envelope.
            target = target-normal*min(p.surface_thickness/2, width/2)
            clearance = min(p.surface_thickness/4, width/4)
            for _ in range(8):
                if source.depth(target[None])[0] >= clearance: break
                closest, normal, _ = source.closest(target)
                target = closest-normal*(clearance+1e-5)
            if source.depth(target[None])[0] < 0:
                raise ValueError('孔边连接点越出原实体；请检查曲面法向或增加局部厚度')
            return target

        lo, hi = np.floor(surface.source_mesh.bounds/p.cell).astype(int)
        if np.prod(hi-lo+1) > MAX_GRAPH_CELLS:
            raise ValueError('连续杆网的单元数量过多；请增大单元或缩小实体，或关闭连续杆端转接进行对照')
        # Iterate cell rows, not a potentially enormous full bounding-box graph.
        n = max(4, int(np.ceil(np.sqrt(3)*p.cell/2/min(p.wall/2, p.surface_thickness/2))))
        fractions = np.linspace(0, 1, n+1)
        bits = np.array(list(itertools.product((0, 1), repeat=3)))
        for ix in range(lo[0], hi[0]+1):
            for iy in range(lo[1], hi[1]+1):
                cells = np.column_stack((np.full(hi[2]-lo[2]+1, ix), np.full(hi[2]-lo[2]+1, iy), np.arange(lo[2], hi[2]+1)))
                starts = np.repeat((cells+.5)*p.cell, 8, axis=0)
                ends = (cells[:, None, :]+bits).reshape(-1, 3)*p.cell
                samples = starts[:, None, :]+fractions[None, :, None]*(ends-starts)[:, None, :]
                depths = source.depth(samples.reshape(-1, 3)).reshape(-1, n+1)
                inside = depths >= p.surface_thickness
                for a, b, depths_row, inside_row in zip(starts, ends, depths, inside):
                    if not inside_row.any():
                        continue
                    crossings = np.flatnonzero(inside_row[1:] != inside_row[:-1])
                    roots = []
                    for j in crossings:
                        low, high = fractions[j], fractions[j+1]
                        for _ in range(8):
                            mid = (low+high)/2
                            state = source.depth((a+mid*(b-a))[None])[0] >= p.surface_thickness
                            if state == inside_row[j]: low = mid
                            else: high = mid
                        roots.append((low+high)/2)
                    limits = [0.]+roots+[1.]
                    for j, (s, t) in enumerate(zip(limits, limits[1:])):
                        if not (inside_row[0] ^ bool(j % 2)) or t-s < 1e-8:
                            continue
                        start, end = a+s*(b-a), a+t*(b-a)
                        if self.aligned:
                            aligned_edges.append((start,end,s>0,t<1))
                            continue
                        left = anchor(start) if s > 0 else None
                        right = anchor(end) if t < 1 else None
                        self._route(start, end, left, right, source, p)
        if self.aligned:
            terminals=[]
            for start,end,left,right in aligned_edges:
                direction=(end-start)/np.linalg.norm(end-start)
                if left:terminals.append((start,-direction))
                if right:terminals.append((end,direction))
            surface.anchor_sampler.fit_to_terminals(terminals)
            from .segment_guard import SegmentGuard
            guard=surface.anchor_sampler.segment_guard=SegmentGuard(p.cell,p.wall)
            if self.vertex:
                self._route_vertices(aligned_edges,source,p,surface,guard)
            else:
                # Register unchanged core members before routing the boundary.
                for start,end,left,right in aligned_edges:
                    if not (left and source.closest(start)[2] in surface.active) and not (right and source.closest(end)[2] in surface.active):
                        guard.add(start,end)
                for start,end,left,right in aligned_edges:
                    self._route_aligned(start,end,left,right,source,p,surface)
        if not self.paths:
            raise ValueError('实体过薄，无法生成连续内部 BCC；请减小单元或面层厚度')
        if self.aligned:
            self.stats.update(straight_boundary_members=self.boundary_members-self.omitted_members,
                omitted_boundary_members=self.omitted_members,
                boundary_members_considered=self.boundary_members,
                straight_angle_deg=np.percentile(self.straight_angles,[0,50,95,100]).tolist() if self.straight_angles else [],
                straight_centerline_deviation_mm=0.,
                crossing_candidate_rejections=surface.anchor_sampler.segment_guard.rejections,
                continuity_mode='aligned_straight_bcc')
            if self.vertex:
                sampler=surface.anchor_sampler
                vertex_ids=[sampler.vertex_id(target) for _,target,_ in self.terminals]
                self.stats.update(continuity_mode='vertex_only_bcc',grazing_members_removed=self.grazing_removed,
                    vertex_connected_nodes=len(set(vertex_ids)),vertex_terminal_ids=vertex_ids,
                    nonvertex_terminals=0,side_contact_candidate_rejections=sampler.side_contact_rejections,
                    removed_old_outer_members=self.omitted_members,
                    vertex_posts_unconnected=len(sampler.vertices)-len(vertex_ids),
                    surface_connection_segments=2,straight_boundary_members=len(vertex_ids),
                    straight_centerline_deviation_mm=None,straight_angle_deg=[],
                    individual_segment_deviation_mm=0.,
                    connection_path_kind='vertex_post_and_straight_core_bar')
                if not vertex_ids:raise ValueError('未保留任何顶点连接；请减小单元或调整孔距')
                if len(vertex_ids)<max(1,.25*len(sampler.vertices)):
                    raise ValueError(f'可连接多边形顶点过少 ({len(vertex_ids)}/{len(sampler.vertices)})，未输出过疏的外层连接')
            if not self.vertex and self.omitted_members>max(4,.25*self.boundary_members):
                raise ValueError(f'直杆匹配失败比例过高 ({self.omitted_members}/{self.boundary_members})；请调整孔距或轴向，未输出大量删杆结果')
        points = np.vstack(self.paths)
        self.tree = cKDTree(points)
        self.replacement_tree=None
        if self.replaced_paths:
            replacements=np.vstack(self.replaced_paths)
            if len(replacements)+len(points)>MAX_GRAPH_SAMPLES:raise ValueError('顶点替换图超出低内存中心线预算')
            self.replacement_tree=cKDTree(replacements)
            self.stats['replaced_original_members']=len(self.replaced_paths)
            self.stats['replacement_samples']=len(replacements)
        self.routed_tree = (cKDTree(np.vstack([curve for _,_,curve in self.terminals]))
                            if surface.anchor_sampler is not None and self.terminals else None)
        self.stats['graph_samples'] = len(points)
        self.stats['graph_segments'] = len(self.paths)
        self.stats['max_centerline_step_mm'] = self.step
        # Audit actual sheet/rail support after normal offsets and projection,
        # not just the nominal UV target. Closed triangles alone cannot prove
        # that a rod terminal is attached.
        margins = []
        for _, target, _ in self.terminals:
            if surface.adaptive is not None:
                margins.append(surface.adaptive.support(target))
                continue
            _, _, fid = source.closest(target)
            depth = source.depth(target[None])[0]
            support = p.shell-depth if fid not in surface.active else min(
                pattern_field(target[None], surface.charts[fid], p.surface_pattern, cell, width, p.pore_rounding)[0],
                p.surface_thickness-depth)
            if fid in surface.active:
                if surface.seam_tree is not None:
                    support = max(support, max(width/2,p.surface_thickness*.75)-surface.seam_tree.query(target)[0])
                if surface.edge_tree is not None:
                    support = max(support, p.edge_width-surface.edge_tree.query(target)[0])
            margins.append(support)
        minimum = min(margins, default=0.)
        self.stats['min_anchor_material_margin_mm'] = float(minimum)
        if margins and minimum <= 0:
            raise ValueError('连杆转接未与孔边筋可靠相交，已停止输出；请调整曲面孔距或边缘实边宽度')

    def _route_vertices(self,edges,source,p,surface,guard):
        sampler=surface.anchor_sampler;nodes={}
        def line(a,b):
            return np.linspace(a,b,max(2,int(np.ceil(np.linalg.norm(b-a)/self.step))+1))
        for start,end,left,right in edges:
            active=(left and source.closest(start)[2] in surface.active) or (right and source.closest(end)[2] in surface.active)
            path=line(start,end)
            # Boundary shafts are rebuilt from polygon vertices, never retained
            # merely because their old endpoint happened to land on web material.
            unsafe=active or not sampler.contact_safe(path,[])
            if unsafe:
                self.boundary_members+=1;self.omitted_members+=1
                self.grazing_removed+=int(not active)
                self._replace_original(start,end,left,right,p)
                continue
            self._track_vertex_samples(len(path))
            self.paths.append(path);guard.add(start,end)
            for point in (start,end):
                periodic=point/p.cell
                on_node=(np.max(abs(periodic-np.round(periodic)))<1e-5 or
                         np.max(abs(periodic-.5-np.round(periodic-.5)))<1e-5)
                if on_node and sampler.face_probe.depth(point[None])[0]>=sampler.standoff+p.wall*.2:
                    nodes[tuple(np.round(point,7))]=point.copy()
        if not nodes:raise ValueError('没有足够深的内部 BCC 节点可连接多边形顶点')
        core_nodes=np.array([nodes[key] for key in sorted(nodes)])
        tree=cKDTree(core_nodes);usage=np.zeros(len(core_nodes),int)
        connected=set();valid_posts={}
        for index,(vertex,port) in enumerate(zip(sampler.vertices,sampler.inner_vertices)):
            if not np.isfinite(port).all():continue
            post=line(port,vertex)
            if not sampler.contact_safe(post,[vertex]):continue
            if source.depth(post).min()<sampler.material_margin:continue
            valid_posts[index]=post
            distances,indices=tree.query(port,k=min(24,len(core_nodes)))
            options=[]
            for distance,i in zip(np.atleast_1d(distances),np.atleast_1d(indices)):
                if distance>p.cell*1.5 or distance<p.wall*.25 or usage[i]>=4:continue
                root=core_nodes[i];bar=line(root,port)
                if not sampler.core_safe(bar):continue
                if not guard.clear(root,port) or not guard.clear(port,vertex):continue
                direction=(port-root)/distance
                deviation=float(np.degrees(np.arccos(np.clip(np.sum(abs(direction))/np.sqrt(3),-1,1))))
                cost=distance/p.cell+deviation/45+usage[i]*.2
                options.append((cost,int(i),bar,deviation))
            if not options:continue
            _,i,bar,deviation=min(options,key=lambda item:item[0])
            usage[i]+=1;guard.add(core_nodes[i],port);guard.add(port,vertex)
            self._track_vertex_samples(len(bar)+len(post))
            self.paths.extend([bar,post])
            curve=np.vstack([bar,post[1:]])
            self.terminals.append((core_nodes[i],vertex,curve))
            self.straight_angles.append(deviation)
            connected.add(index)
        self.stats['vertex_direct_core_nodes']=len(connected)
        # Continue the offset polygon graph through genuine neighbouring
        # vertices (two common sites). Never attach a new bar to an edge point.
        adjacency={i:set() for i in valid_posts};edge_owners={}
        for i in valid_posts:
            for pair in itertools.combinations(sampler.vertex_triples[i],2):
                key=tuple(sorted(pair))
                for j in edge_owners.get(key,[]):adjacency[i].add(j);adjacency[j].add(i)
                edge_owners.setdefault(key,[]).append(i)
        import heapq
        frontier=[]
        def enqueue(j):
            for i in sorted(adjacency[j]-connected):
                distance=np.linalg.norm(sampler.inner_vertices[i]-sampler.inner_vertices[j])
                if distance<=sampler.cell*1.5:heapq.heappush(frontier,(distance,i,j))
        for j in sorted(connected):enqueue(j)
        while frontier:
            _,i,j=heapq.heappop(frontier)
            if i in connected:continue
            bar=line(sampler.inner_vertices[j],sampler.inner_vertices[i])
            if not sampler.core_safe(bar):continue
            if not guard.clear(bar[0],bar[-1]) or not guard.clear(bar[-1],sampler.vertices[i]):continue
            port=sampler.inner_vertices[i];vertex=sampler.vertices[i];post=valid_posts[i]
            guard.add(bar[0],port);guard.add(port,vertex)
            self._track_vertex_samples(len(bar)+len(post))
            self.paths.extend([bar,post]);self.terminals.append((bar[0],vertex,np.vstack([bar,post[1:]])))
            connected.add(i)
            enqueue(i)
        self.stats['vertex_neighbour_core_nodes']=len(connected)-self.stats['vertex_direct_core_nodes']
        self.stats.update(routed_terminals=len(self.terminals),core_alignment_angle_deg=
            np.percentile(self.straight_angles,[0,50,95,100]).tolist() if self.straight_angles else [])
        self.sample_count=sum(len(path) for path in self.paths)
        if self.sample_count>MAX_GRAPH_SAMPLES:raise ValueError('顶点连杆图超出低内存预算')

    def _route_aligned(self,start,end,left_clip,right_clip,source,p,surface):
        direction=end-start;length=np.linalg.norm(direction);direction/=length
        left=left_clip and source.closest(start)[2] in surface.active
        right=right_clip and source.closest(end)[2] in surface.active
        if not left and not right:
            paths=[np.linspace(start,end,max(2,int(np.ceil(length/self.step))+1))]
        else:
            self.boundary_members+=1
            sampler=surface.anchor_sampler
            target_a,target_b=start,end
            angles=[]
            if left:
                match=sampler.match(start,end,-direction,self.step)
                if match is None:
                    self.omitted_members+=1;return
                target_a,_,angle=match;angles.append(angle)
            if right:
                match=sampler.match(end,target_a,direction,self.step)
                if match is None:
                    self.omitted_members+=1;return
                target_b,_,angle=match;angles.append(angle)
            span=np.linalg.norm(target_b-target_a)
            curve=np.linspace(target_a,target_b,max(2,int(np.ceil(span/self.step))+1))
            cosine=np.dot((target_b-target_a)/span,direction)
            if cosine<np.cos(np.deg2rad(50)) or source.depth(curve).min()<min(.12,(p.surface_width or p.wall)/6):
                self.omitted_members+=1;return
            if not sampler.segment_guard.clear(target_a,target_b):
                self.omitted_members+=1;return
            sampler.segment_guard.add(target_a,target_b)
            paths=[curve]
            middle=len(curve)//2
            if left:
                sampler.contacts.append(target_a)
                self.terminals.append((start,target_a,curve[:middle+1][::-1] if right else curve[::-1]))
            if right:
                sampler.contacts.append(target_b)
                self.terminals.append((end,target_b,curve[middle:] if left else curve))
            self.straight_angles.extend(angles)
            self.stats['routed_terminals']+=int(left)+int(right)
        self.sample_count+=sum(len(path) for path in paths)
        if self.sample_count>MAX_GRAPH_SAMPLES:raise ValueError('直杆中心线超出低内存预算')
        self.paths.extend(paths)

    def _replace_original(self,start,end,left_clip,right_clip,p):
        direction=(end-start)/np.linalg.norm(end-start)
        # Suppress the entire old boundary shaft, even across nearest CAD label
        # changes. Restore retained neighbours from the explicit graph afterward.
        extension=p.surface_thickness+p.wall
        a=start-direction*extension if left_clip else start
        b=end+direction*extension if right_clip else end
        path=np.linspace(a,b,max(2,int(np.ceil(np.linalg.norm(b-a)/self.step))+1))
        self.replacement_sample_count+=len(path)
        self._track_vertex_samples(0)
        self.replaced_paths.append(path)

    def _track_vertex_samples(self,count):
        self.sample_count+=count
        if self.sample_count+self.replacement_sample_count>MAX_GRAPH_SAMPLES:
            raise ValueError('顶点替换图超出低内存中心线预算')

    def replace_periodic(self,values,points):
        result=values.copy()
        if self.replacement_tree is None:return result
        for start in range(0,len(points),65536):
            q=points[start:start+65536]
            distance=self.replacement_tree.query(q,distance_upper_bound=self.replacement_radius)[0]
            mask=np.isfinite(distance)
            if mask.any():
                old=result[start:start+len(q)]
                old[mask]=np.maximum(np.minimum(old[mask],distance[mask]-self.replacement_radius),self.field(q[mask]))
        return result

    def _route(self, start, end, left, right, source, p):
        direction = end-start; length = np.linalg.norm(direction); direction /= length
        retreat = min(p.wall*.8, length*.25)
        a = start+direction*retreat if left is not None else start
        b = end-direction*retreat if right is not None else end
        paths = [np.linspace(a, b, max(2, int(np.ceil(np.linalg.norm(b-a)/self.step))+1))]
        for root, inner, target, tangent in ((start, a, left, -direction), (end, b, right, direction)):
            if target is None: continue
            span = np.linalg.norm(target-inner)
            # A cubic departs along the original shaft and flattens into the
            # sheet. Retry with less curvature if a concave boundary interferes.
            curve = None
            for tension in (.35, .15, 0.):
                candidate = bezier(inner, inner+tangent*min(retreat, span*tension),
                                   target+(inner-target)*tension, target, self.step)
                if np.min(source.depth(candidate)) >= -1e-5:
                    curve = candidate; break
            if curve is None:
                # Concave patches: a straight chord can leave a thin wall even
                # when both endpoints are inside. Project the local path back
                # into the wall, then relax it with fixed, attached endpoints.
                curve = candidate.copy()
                clearance = min(p.surface_thickness/4, p.wall/4)
                for iteration in range(16):
                    if iteration:
                        curve[1:-1] = .5*curve[1:-1]+.25*(curve[:-2]+curve[2:])
                    depths = source.depth(curve)
                    for j in np.flatnonzero(depths < clearance):
                        if j in (0, len(curve)-1): continue
                        point, normal, _ = source.closest(curve[j])
                        curve[j] = point-normal*clearance
                # Resample after projection to retain the ball-overlap bound.
                pieces = []
                for first, last in zip(curve, curve[1:]):
                    count = max(1, int(np.ceil(np.linalg.norm(last-first)/self.step)))
                    pieces.append(first+np.arange(count)[:,None]/count*(last-first))
                curve = np.vstack(pieces+[curve[-1:]])
                if np.min(source.depth(curve)) < -1e-5:
                    raise ValueError(f'局部连杆转接越出原实体；未输出悬空连接 (start={inner.tolist()}, target={target.tolist()}, min_depth={source.depth(curve).min():.6f})')
            paths.append(curve)
            self.terminals.append((root, target, curve))
            self.stats['routed_terminals'] += 1
        self.sample_count += sum(len(path) for path in paths)
        if self.sample_count > MAX_GRAPH_SAMPLES:
            raise ValueError('连续杆网超过低内存中心线预算；请增大单元或缩小实体，未继续申请大型数组')
        self.paths.extend(paths)

    def field(self, points):
        # Union of densely overlapping balls; sagitta error <= step^2/(8r).
        out = np.empty(len(points), dtype=np.float32)
        for i in range(0, len(points), 65536):
            out[i:i+65536] = self.radius-self.tree.query(points[i:i+65536])[0]
        return out

    def routed_field(self, points):
        out = np.full(len(points), -np.inf, dtype=np.float32)
        if self.routed_tree is not None:
            for i in range(0,len(points),65536):
                out[i:i+65536] = self.radius-self.routed_tree.query(points[i:i+65536],
                    distance_upper_bound=4*self.radius)[0]
        return out

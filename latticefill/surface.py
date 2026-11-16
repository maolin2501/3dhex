"""Perforated face sheets, distinct from a volume strut network.

Patterns live in millimetre surface charts. Cylinders use arc length, planes
use an orthogonal frame; other surfaces use a documented local projection.
The final mesh intersection keeps the original tessellated appearance faces.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from .lattices import _hex_phi, _hex_local


def smooth_union(a, b, radius):
    """Bounded polynomial union; no change outside the local blend band."""
    if radius <= 0:
        return np.maximum(a, b)
    h=np.maximum(2*radius-np.abs(a-b),0)/(2*radius)
    return np.maximum(a,b)+h*h*radius/2


def rounded_hex_field(u, v, cell, width, radius):
    if radius <= 0:
        return _hex_phi(u,v,cell,width)
    # Erode the void polygon, then dilate by radius. Straight portions keep
    # their original spacing and web width; only the six pore corners round.
    px,py,_=_hex_local(u,v,cell)
    apothem=(cell-width)/2
    radius=min(radius,.8*apothem)
    inner=apothem-radius
    angles=np.arange(6)*np.pi/3+np.pi/6
    vertices=np.column_stack([np.cos(angles),np.sin(angles)])*(inner/(np.sqrt(3)/2))
    distance=np.full_like(px,np.inf,dtype=float)
    for a,b in zip(vertices,np.roll(vertices,-1,axis=0)):
        edge=b-a
        t=np.clip(((px-a[0])*edge[0]+(py-a[1])*edge[1])/np.dot(edge,edge),0,1)
        distance=np.minimum(distance,np.hypot(px-a[0]-t*edge[0],py-a[1]-t*edge[1]))
    support=np.maximum(abs(px),np.maximum(abs(.5*px+np.sqrt(3)/2*py),abs(.5*px-np.sqrt(3)/2*py)))
    return np.where(support<=inner,-distance,distance)-radius


def face_chart(mesh, kind, stored=None):
    if stored:
        return dict(stored)
    if kind == 'Torus':
        raise ValueError('环面缺少解析坐标，请重新导入 STEP；禁止将环面退化为平面投影排孔')
    # Compatibility with previously imported models that lack CAD frames.
    centre = mesh.vertices.mean(axis=0)
    _, _, axes = np.linalg.svd(mesh.vertices - centre, full_matrices=False)
    if kind == 'Cylinder':
        _, _, normals = np.linalg.svd(mesh.face_normals, full_matrices=False)
        axis = normals[-1]
        axis *= 1 if axis[np.argmax(abs(axis))] >= 0 else -1
        u = np.eye(3)[np.argmin(abs(axis))]
        u = u - np.dot(u, axis) * axis
        u /= np.linalg.norm(u)
        v = np.cross(axis, u)
        xy = (mesh.vertices - centre) @ np.column_stack((u, v))
        a, b, c = np.linalg.lstsq(np.column_stack((2*xy, np.ones(len(xy)))),
                                 (xy*xy).sum(axis=1), rcond=None)[0]
        centre += a*u + b*v
        radius = float(np.sqrt(c+a*a+b*b))
        return dict(kind='cylinder', origin=centre.tolist(), u=u.tolist(),
                    v=v.tolist(), axis=axis.tolist(), radius=radius)
    u, v = axes[:2]
    if kind != 'Plane':
        scale = abs(mesh.face_normals @ np.cross(u, v))
        if mesh.area_faces[scale < .15].sum() > .05 * mesh.area:
            raise ValueError('该自由曲面无法安全投影排孔，请在 CAD 中分割为较小曲面后重新导入，避免长条和碎片')
    return dict(kind='plane' if kind == 'Plane' else 'projection',
                origin=centre.tolist(), u=u.tolist(), v=v.tolist())


def chart_uv(points, chart):
    q = points - chart['origin']
    x, y = q @ np.asarray(chart['u']), q @ np.asarray(chart['v'])
    if chart['kind'] == 'torus':
        sign = chart['radial_sign']
        theta = np.arctan2(sign*y, sign*x)
        height = q @ np.asarray(chart['axis'])
        angle = np.arctan2(height, sign*np.hypot(x,y)-chart['major_radius'])
        delta = (angle-chart['v_mid']+np.pi) % (2*np.pi)-np.pi
        return theta*chart['radius'], delta*chart['minor_radius']
    if chart['kind'] == 'cylinder':
        # A full cylinder has a chart seam. Use an integer number of repeats
        # around it (see pattern_field) so the pore network is continuous.
        return np.arctan2(y, x) * chart['radius'], q @ np.asarray(chart['axis'])
    return x, y


def pattern_field(points, chart, pattern, cell, width, rounding=0):
    u, v = chart_uv(points, chart)
    if chart['kind'] in ('cylinder', 'torus'):
        circumference = 2*np.pi*chart['radius']
        if circumference < 1e-6:
            raise ValueError('曲面位于参数奇点，无法安全排孔，请分割该曲面')
        repeats = max(3, round(circumference/cell))
        u = u * (repeats*cell/circumference)
        if chart.get('v_periodic'):
            period = np.sqrt(3)*cell if pattern == 'hex' else cell
            v *= max(1,round(2*np.pi*chart['minor_radius']/period))*period/(2*np.pi*chart['minor_radius'])
    if pattern == 'hex':
        return rounded_hex_field(u, v, cell, width, rounding)
    x = abs(u - cell*np.round(u/cell))
    y = abs(v - cell*np.round(v/cell))
    if pattern == 'square':
        if rounding > 0:
            radius=min(rounding,.4*(cell-width))
            dx=(cell/2-x)-((cell-width)/2-radius)
            dy=(cell/2-y)-((cell-width)/2-radius)
            return np.hypot(np.maximum(dx,0),np.maximum(dy,0))+np.minimum(np.maximum(dx,dy),0)-radius
        return width/2 - np.minimum(x, y)
    if pattern == 'circle':
        return np.sqrt(x*x+y*y) - (cell-width)/2
    raise ValueError('无效曲面孔型')


class PerforatedSurface:
    def __init__(self, body, active_ids, translation=None, chart_mode=None):
        self.meshes = {}
        self.charts = {}
        self.active = set(active_ids)
        self.approximate = []
        self.edge_tree = None
        self.seam_tree = None
        self.network = None
        self.anchor_sampler = None
        self.stitches = None
        self.adaptive = None
        self.source_mesh = body.mesh.copy()
        self.source_face_ids = body.face_ids.copy()
        if translation is not None:
            self.source_mesh.apply_translation(translation)
        adjacent=body.mesh.face_adjacency
        ids=body.face_ids[adjacent]
        active=np.isin(ids,list(self.active))
        # Protect the design/protected boundary and physical CAD creases, not
        # every tessellation edge or every tangent CAD patch seam.
        boundary=(active[:,0]!=active[:,1]) | (active.any(axis=1) &
            (ids[:,0]!=ids[:,1]) & (body.mesh.face_adjacency_angles>np.deg2rad(35)))
        edge_ids=body.mesh.face_adjacency_edges[boundary]
        self.edge_segments=body.mesh.vertices[edge_ids].copy()
        rim = active[:,0] != active[:,1]
        self.rim_segments = body.mesh.vertices[body.mesh.face_adjacency_edges[rim]].copy()
        design_seam = active.all(axis=1) & (ids[:,0] != ids[:,1])
        self.design_seam_edges = body.mesh.face_adjacency_edges[design_seam].copy()
        self.design_seam_pairs = ids[design_seam].copy()
        # CAD chart phases are independent even across tangent patches. A narrow
        # shared rail joins their webs; it is not a blanket solid skin.
        seam = active.any(axis=1) & (ids[:,0] != ids[:,1])
        self.seam_segments = body.mesh.vertices[body.mesh.face_adjacency_edges[seam]].copy()
        if translation is not None:
            self.edge_segments+=translation
            self.seam_segments+=translation
            self.rim_segments+=translation
        for face in body.face_info:
            fid = face['id']
            mesh = body.selected_mesh([fid])
            chart = (face_chart(mesh, face['type'], face.get('chart')) if fid in self.active and chart_mode not in ('adaptive','aligned','vertex','atlas')
                     else dict(kind='unused',origin=[0,0,0]))
            if translation is not None:
                mesh.apply_translation(translation)
                chart['origin'] = (np.asarray(chart['origin']) + translation).tolist()
            self.meshes[fid] = mesh
            self.charts[fid] = chart
            if fid in self.active and chart['kind'] == 'projection':
                self.approximate.append(fid)

    def prepare(self, shape, origin, pitch, params=None):
        self.origin, self.pitch = origin, pitch
        self.grid_order=np.arange(3)
        self.labels = np.zeros(shape, dtype=np.int32)
        for fid, mesh in self.meshes.items():
            points = mesh.voxelized(pitch=pitch).points
            ijk = np.rint((points-origin)/pitch).astype(np.int64)
            valid = np.all((ijk >= 0) & (ijk < np.asarray(shape)), axis=1)
            self.labels[tuple(ijk[valid].T)] = fid+1
        self.prepare_edges(pitch,params)

    def prepare_edges(self,pitch,params):
        if params is not None and params.seam_style in ('vertex','atlas') and not self.active:
            raise ValueError('顶点模式没有可开孔的设计面；请取消部分面的保留标记')
        self.edge_tree=None
        self.seam_tree=None
        self.network=None
        self.anchor_sampler=None
        self.stitches=None
        self.adaptive=None
        stitch = (params is not None and params.surface_continuity == 'auto'
                  and params.lattice == 'bcc' and params.seam_style in ('stitch','adaptive'))
        adaptive = params is not None and params.seam_style in ('adaptive','aligned','vertex','atlas')
        stitch = stitch or adaptive
        edges = self.rim_segments if stitch else self.edge_segments
        if params is not None and params.edge_width > 0 and len(edges):
            samples=[]
            for a,b in edges:
                n=max(1,int(np.ceil(np.linalg.norm(b-a)/(pitch/2))))
                samples.append(a+np.linspace(0,1,n+1)[:,None]*(b-a))
            self.edge_tree=cKDTree(np.vstack(samples))
        if adaptive and self.active:
            from .adaptive_surface import AdaptiveSurface
            from .aligned_surface import AlignedSurface
            from .vertex_surface import VertexSurface
            from .atlas_surface import AtlasSurface
            self.adaptive={'atlas':AtlasSurface,'vertex':VertexSurface,'aligned':AlignedSurface,'adaptive':AdaptiveSurface}[params.seam_style](self,params,pitch)
            self.anchor_sampler=self.adaptive
            params.stats.update(self.adaptive.stats)
        if params is not None and params.surface_continuity == 'auto':
            if len(self.seam_segments) and not stitch:
                samples=[]
                for a,b in self.seam_segments:
                    n=max(1,int(np.ceil(np.linalg.norm(b-a)/(pitch/2))))
                    samples.append(a+np.linspace(0,1,n+1)[:,None]*(b-a))
                self.seam_tree=cKDTree(np.vstack(samples))
            params.stats['seam_rail_length_mm']=float(np.linalg.norm(self.seam_segments[:,1]-self.seam_segments[:,0],axis=1).sum()) if self.seam_tree is not None and len(self.seam_segments) else 0.
            if params.lattice == 'bcc' and self.active:
                from .continuity import BoundaryBCC
                if stitch and not adaptive:
                    from .surface_stitch import SurfaceAnchors
                    self.anchor_sampler=SurfaceAnchors(self,params,pitch)
                self.network=BoundaryBCC(self,params,pitch)
                params.stats.update(self.network.stats)
                if stitch and not adaptive:
                    from .surface_stitch import SurfaceStitches
                    self.stitches=SurfaceStitches(self,params,pitch)
                    params.stats.update(self.stitches.stats)
                if adaptive:
                    params.stats.update(self.adaptive.stats)
                    params.stats['surface_core_connections']=len(self.network.terminals)

    def depth_limit(self,p):
        return max(p.surface_thickness,p.edge_width,
                   self.network.band_depth if self.network is not None else 0)

    def combine(self, lattice, sdf, k0, p):
        k1 = k0 + sdf.shape[2]
        depth_limit=self.depth_limit(p)
        halo = int(np.ceil(depth_limit/self.pitch)) + 5
        a, b = max(0, k0-halo), min(self.labels.shape[2], k1+halo)
        labels = self.labels[:, :, a:b]
        if not np.any(labels):
            return lattice
        indices = ndimage.distance_transform_edt(labels == 0, return_distances=False,
                                                 return_indices=True)
        nearest = labels[tuple(indices)][:, :, k0-a:k1-a]
        del indices
        return self.combine_nearest(lattice,sdf,k0,p,nearest)

    def combine_nearest(self,lattice,sdf,k0,p,nearest):
        depth_limit=self.depth_limit(p)
        band = (sdf > -2*self.pitch) & (sdf < depth_limit + 2*self.pitch)
        result = lattice.copy()
        if self.network is not None and self.network.vertex:
            mask=sdf>-2*self.pitch
            ijk=np.argwhere(mask)
            if len(ijk):
                points=(self.origin+(ijk+[0,0,k0])*self.pitch)[:,np.argsort(self.grid_order)]
                result[mask]=self.network.replace_periodic(result[mask],points)
        for fid in self.active:
            mask = band & (nearest == fid+1)
            ijk = np.argwhere(mask)
            if not len(ijk):
                continue
            points = self.origin + (ijk + [0, 0, k0])*self.pitch
            points=points[:,np.argsort(self.grid_order)]
            cell=p.surface_cell or p.cell
            width=p.surface_width or p.wall
            sheet = (self.adaptive.field(points) if self.adaptive is not None else
                     pattern_field(points, self.charts[fid], p.surface_pattern, cell, width, p.pore_rounding))
            depth = sdf[mask]
            # Near the face the sheet REPLACES rods; otherwise rods sticking
            # through a hole could reproduce the original reported defect.
            core = (self.network.field(points) if self.network is not None else
                    np.minimum(lattice[mask], depth-p.surface_thickness))
            sheet = np.minimum(sheet, p.surface_thickness + self.pitch - depth)
            joined=smooth_union(core,sheet,p.junction_blend)
            if self.edge_tree is not None:
                distance=self.edge_tree.query(points)[0]
                joined=smooth_union(joined,p.edge_width-distance,p.junction_blend)
            if self.seam_tree is not None:
                distance=self.seam_tree.query(points)[0]
                rail=np.minimum(max(width/2,p.surface_thickness*.75)-distance,
                                p.surface_thickness+self.pitch-depth)
                joined=smooth_union(joined,rail,p.junction_blend)
            result[mask] = joined
        if self.anchor_sampler is not None and self.network is not None:
            # Explicit connections survive nearest-face label changes and may
            # extend deeper than the chart band. Do not increase the EDT halo.
            mask = sdf > -2*self.pitch
            ijk = np.argwhere(mask)
            if len(ijk):
                points = self.origin+(ijk+[0,0,k0])*self.pitch
                points = points[:,np.argsort(self.grid_order)]
                # The BCC route already participates in the face-band union;
                # union it idempotently here, or duplicate smoothing fattens it.
                joined = np.maximum(result[mask], self.network.routed_field(points))
                result[mask] = (smooth_union(joined,self.stitches.field(points),p.junction_blend)
                                if self.stitches is not None else joined)
        return result


def trim_to_source(result, source, low_memory=False):
    """Mesh Boolean, not a B-rep/STEP Boolean; never silently repair the source."""
    import trimesh
    if low_memory:
        from .mesh_quality import as_manifold, as_trimesh
        from .mesh_memory import release_cache
        release_cache(result)
        a,b=as_manifold(source),as_manifold(result)
        if a.volume()<=0 or b.volume()<=0:return result,False
        solid=a^b
        if solid.is_empty() or solid.volume()<=0:
            raise ValueError('原曲面裁切失败；请检查源实体闭合性或提高采样精度')
        # Force evaluation, then release both input solids before copying output.
        del a,b
        return as_trimesh(solid),True
    if not source.is_volume or not result.is_volume:
        return result, False
    clipped = trimesh.boolean.intersection([source, result], engine='manifold')
    if clipped is None or not len(clipped.faces) or not clipped.is_volume:
        raise ValueError('原曲面裁切失败；请检查源实体闭合性或提高采样精度')
    return clipped, True

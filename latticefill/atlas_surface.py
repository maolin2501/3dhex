"""Bounded local projection charts, reconciled into one vertex-only pore web.

Regular interiors, not a promise of globally regular polygons on arbitrary
curvature. Charts use actual triangle intersections (no ring projection loss).
"""
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from .vertex_surface import VertexSurface
from .surface_stitch import SurfaceAnchors
from .continuity import MeshProbe
from .adaptive_surface import MAX_ADAPTIVE_SITES, QUERY_BATCH


class OrientedSiteTree:
    """Nearest compatible sheet; opposite sides of a thin wall do not compete."""
    def __init__(self, sites, normals, normal_at=None):
        self.data = np.asarray(sites)
        self.normals = np.asarray(normals)
        self.raw = cKDTree(self.data)
        self.normal_at=normal_at

    def query(self, points, k=1, normals=None):
        points = np.asarray(points)
        scalar = points.ndim == 1
        q = points.reshape(-1, 3)
        reference=np.asarray(normals) if normals is not None else self.normal_at(q)
        reference=reference.reshape(-1,3)
        count = min(len(self.data), max(32, k*3))
        distance, ids = self.raw.query(q, k=count)
        distance = distance.reshape(len(q), count); ids = ids.reshape(len(q), count)
        compatible = np.einsum('ijk,ik->ij', self.normals[ids], reference) > .35
        ranked = np.where(compatible, distance, np.inf)
        order = np.argsort(ranked, axis=1, kind='stable')[:, :min(k,count)]
        chosen = np.take_along_axis(ids, order, axis=1)
        ds = np.take_along_axis(ranked, order, axis=1)
        missing = ~np.isfinite(ds)
        chosen[missing] = np.broadcast_to(chosen[:, :1], chosen.shape)[missing]
        ds[missing] = np.broadcast_to(ds[:, :1], ds.shape)[missing]
        if count < k:
            chosen=np.pad(chosen,((0,0),(0,k-count)),mode='edge')
            ds=np.pad(ds,((0,0),(0,k-count)),mode='edge')
        if k == 1: chosen=chosen[:,0]; ds=ds[:,0]
        return (ds[0],chosen[0]) if scalar else (ds,chosen)


class AtlasSurface(VertexSurface):
    def __init__(self, surface, p, pitch):
        SurfaceAnchors.__init__(self, surface, p, pitch)
        if p.surface_pattern not in ('hex','square'):
            raise ValueError('曲面分区铺网支持六边形或方形，请选择有效孔型')
        self.material_margin=min(.16,self.width*.23,p.surface_thickness*.23)
        self.phase_margin=min(.18,self.width*.26,p.surface_thickness*.26)
        selected=np.isin(surface.source_face_ids,list(surface.active))
        self.mesh=trimesh.Trimesh(surface.source_mesh.vertices.copy(),
            surface.source_mesh.faces[selected],process=False)
        self.mesh.remove_unreferenced_vertices()
        if not len(self.mesh.faces) or self.mesh.area <= 0:
            raise ValueError('曲面分区没有可开孔区域')
        self.face_probe=MeshProbe(self.mesh)
        self.vertex_tree=None; self.side_contact_rejections=0
        self.node_radius=max(p.wall/2,self.width/2)+p.junction_blend+pitch/2
        self.web_tree=None; self.contacts=[]
        self.neighbours=12
        self.layout='local_projection_atlas'
        self.row_count=0; self.row_repeats=[]; self.layout_phase=(0.,0.)
        self.patch_count=0; self.supplemental_sites=0
        self.sites,self.site_normals=self._chart_sites()
        self._rebuild_tree()
        self._coverage_samples()
        self._complete_boundaries()
        self.stats=dict(seam_style='atlas',surface_network='atlas_vertex_rows',
            attachment_mode='polygon_vertices_only',vertex_edge_fallbacks=0,
            aligned_layout=self.layout,atlas_patches=self.patch_count,
            adaptive_sites=len(self.sites),fixed_face_patterns=True,
            surface_pattern=p.surface_pattern,seam_rail_length_mm=0.,
            surface_core_connections=0,atlas_supplemental_sites=self.supplemental_sites)
        self.check_coverage()

    def _rebuild_tree(self):
        if len(self.sites)>MAX_ADAPTIVE_SITES:
            raise ValueError('曲面分区孔网取点超出低内存预算，请增大孔距或缩小区域')
        if len(self.sites)<3:
            raise ValueError('曲面分区不足三个孔中心，请缩小孔距')
        self.tree=OrientedSiteTree(self.sites,self.site_normals,normal_at=self.face_probe.normals_at)
        self.neighbours=min(12,len(self.sites))

    def _chart_sites(self):
        mesh=self.mesh; normals=mesh.face_normals; tris=mesh.triangles
        adjacent=[[] for _ in mesh.faces]
        for a,b in mesh.face_adjacency:
            if np.dot(normals[a],normals[b])>.5:
                adjacent[a].append(int(b));adjacent[b].append(int(a))
        visited=np.zeros(len(tris),bool)
        sites=[]; directions=[]; buckets={}
        spacing=self.cell; bucket=spacing*.5

        def insert(point,normal):
            key=tuple(np.floor(point/bucket).astype(int))
            for dx in (-1,0,1):
                for dy in (-1,0,1):
                    for dz in (-1,0,1):
                        for j in buckets.get((key[0]+dx,key[1]+dy,key[2]+dz),[]):
                            if np.dot(normal,directions[j])>.35 and np.linalg.norm(point-sites[j])<spacing*.48:
                                return
            buckets.setdefault(key,[]).append(len(sites));sites.append(point);directions.append(normal)
            if len(sites)>MAX_ADAPTIVE_SITES:
                raise ValueError('曲面分区取点超出低内存预算')

        for seed in np.argsort(-mesh.area_faces,kind='stable'):
            if visited[seed]:continue
            normal=normals[seed]; stack=[int(seed)]; visited[seed]=True; patch=[]
            while stack:
                i=stack.pop();patch.append(i)
                for j in sorted(adjacent[i]):
                    if not visited[j] and np.dot(normals[j],normal)>.82:
                        visited[j]=True;stack.append(j)
            self.patch_count+=1
            v=np.eye(3)['xyz'.index(self.p.axis)]
            v=v-np.dot(v,normal)*normal
            if np.linalg.norm(v)<.1:
                v=np.eye(3)[np.argmin(abs(normal))];v-=np.dot(v,normal)*normal
            v/=np.linalg.norm(v);u=np.cross(v,normal)
            # Shared world-origin phase avoids arbitrary CAD-face phase resets.
            row_step=spacing*(np.sqrt(3)/2 if self.p.surface_pattern=='hex' else 1)
            for i in patch:
                t=tris[i];uv=np.column_stack((t@u,t@v))
                edge1=uv[1]-uv[0];edge2=uv[2]-uv[0]
                det=edge1[0]*edge2[1]-edge1[1]*edge2[0]
                if abs(det)<1e-10:continue
                for row in range(int(np.ceil(uv[:,1].min()/row_step)),int(np.floor(uv[:,1].max()/row_step))+1):
                    offset=.5*(row%2) if self.p.surface_pattern=='hex' else 0
                    left=int(np.ceil(uv[:,0].min()/spacing-offset));right=int(np.floor(uv[:,0].max()/spacing-offset))
                    if right-left>10000:raise ValueError('曲面局部取点范围超出预算')
                    for col in range(left,right+1):
                        q=np.array([(col+offset)*spacing,row*row_step])-uv[0]
                        a=(q[0]*edge2[1]-q[1]*edge2[0])/det
                        b=(edge1[0]*q[1]-edge1[1]*q[0])/det
                        if min(a,b,1-a-b)<-1e-8:continue
                        insert(t[0]+a*(t[1]-t[0])+b*(t[2]-t[0]),normals[i])
        if len(sites)<3:raise ValueError('局部铺网没有足够孔中心，请减小孔距')
        return np.asarray(sites),np.asarray(directions)

    def _coverage_samples(self):
        rng=np.random.default_rng(20261110)
        count=min(100000,max(4096,int(self.mesh.area/self.cell**2*64)))
        ids=rng.choice(len(self.mesh.faces),count,p=self.mesh.area_faces/self.mesh.area)
        uv=rng.random((count,2));uv[uv.sum(axis=1)>1]=1-uv[uv.sum(axis=1)>1]
        t=self.mesh.triangles[ids]
        self.coverage_points=t[:,0]+uv[:,0,None]*(t[:,1]-t[:,0])+uv[:,1,None]*(t[:,2]-t[:,0])
        self.coverage_normals=self.mesh.face_normals[ids]

    def _complete_boundaries(self):
        # Only seam/trim gaps receive supplemental sites; regular chart interiors
        # are not relaxed. Every added point is on an actual source triangle.
        for _ in range(3):
            dist=self.tree.query(self.coverage_points,normals=self.coverage_normals)[0]
            candidates=np.flatnonzero(dist>self.cell*.85)
            if not len(candidates):break
            added=[];added_normals=[]
            for i in candidates[np.argsort(-dist[candidates],kind='stable')]:
                q=self.coverage_points[i];normal=self.coverage_normals[i]
                if added:
                    delta=np.asarray(added)-q
                    nearby=(np.asarray(added_normals)@normal>.35)&(np.einsum('ij,ij->i',delta,delta)<(self.cell*.85)**2)
                    if nearby.any():continue
                added.append(q);added_normals.append(normal)
                if len(added)+len(self.sites)>MAX_ADAPTIVE_SITES:raise ValueError('曲面覆盖修补超出取点预算')
            self.sites=np.vstack([self.sites,added]);self.site_normals=np.vstack([self.site_normals,added_normals])
            self.supplemental_sites+=len(added);self._rebuild_tree()

    def check_coverage(self):
        dist=self.tree.query(self.coverage_points,normals=self.coverage_normals)[0]
        fraction=float(np.mean(dist>self.cell))
        self.stats.update(atlas_coverage_p95_mm=float(np.percentile(dist,95)),
            atlas_coverage_max_mm=float(dist.max()),atlas_uncovered_fraction=fraction,
            atlas_coverage_samples=len(dist),adaptive_sites=len(self.sites))
        if fraction>.01 or np.percentile(dist,99)>self.cell:
            raise ValueError(f'曲面铺网覆盖不足（{fraction:.1%} 区域超过目标孔距），已停止，未生成大孔')
        return self.stats

    def field(self,points):
        from .surface import smooth_union
        result=np.empty(len(points),np.float32)
        for start in range(0,len(points),QUERY_BATCH):
            q=points[start:start+QUERY_BATCH];d,ids=self.tree.query(q,k=self.neighbours)
            # No compatible pore centre means no rib on that local sheet.
            d=np.where(np.isfinite(d),d,0.)
            first=self.sites[ids[:,0]];phi=np.full(len(q),-self.cell,np.float64)
            for j in range(1,self.neighbours):
                separation=np.linalg.norm(self.sites[ids[:,j]]-first,axis=1)
                gap=(d[:,j]**2-d[:,0]**2)/(2*np.maximum(separation,1e-12))
                candidate=np.where((separation>1e-8)&np.isfinite(gap),self.width/2-gap,-self.cell)
                phi=smooth_union(phi,candidate,self.p.pore_rounding)
            result[start:start+len(q)]=phi
        return result

    def fit_to_terminals(self,terminals):
        # Core matching cannot shift the chart grid after coverage validation.
        self._build_vertices()
        self.check_coverage()

    def _build_vertices(self):
        super()._build_vertices()
        valid=[]
        for point,triple in zip(self.vertices,self.vertex_triples):
            d,ids=self.tree.query(point,k=self.neighbours)
            active=set(ids[d<=d.min()+self.stats['vertex_tolerance_mm']])
            valid.append(set(triple).issubset(active))
        keep=np.asarray(valid)
        self.stats['atlas_rejected_nonjunctions']=int((~keep).sum())
        for name in ('vertices','outer_vertices','inner_vertices','vertex_triples'):
            setattr(self,name,getattr(self,name)[keep])
        if not len(self.vertices):raise ValueError('分区孔网没有有效多边形交点，请调整孔距')
        self.vertex_tree=cKDTree(self.vertices)
        self.stats.update(polygon_vertices=len(self.vertices),vertex_posts_available=int(np.isfinite(self.inner_vertices[:,0]).sum()))

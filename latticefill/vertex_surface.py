"""Polygon-vertex-only attachments on the shared restricted Voronoi sheet.

A vertex is the intersection of TWO site bisectors and the sheet mid-depth
level. Edge midpoints, arbitrary supported material, and rim fallbacks are not
vertices. Local tangent Delaunay proposes triples; 3D equal-distance and actual
source-depth checks, not the 2D triangulation, decide which nodes are valid.
"""
import numpy as np
from scipy.spatial import cKDTree,Delaunay,QhullError
from .aligned_surface import AlignedSurface

MAX_VERTEX_TRIPLES=30000


class VertexSurface(AlignedSurface):
    def __init__(self,surface,p,pitch):
        super().__init__(surface,p,pitch)
        self.vertex_tree=None
        self.side_contact_rejections=0
        self.node_radius=max(p.wall/2,self.width/2)+p.junction_blend+pitch/2

    def _statistics(self):
        super()._statistics()
        self.stats.update(seam_style='vertex',attachment_mode='polygon_vertices_only',
                          surface_network='vertex_aligned_rows',vertex_edge_fallbacks=0)

    def _triples(self):
        triples=set()
        for i,point in enumerate(self.sites):
            ids=np.atleast_1d(self.tree.query(point,k=min(12,len(self.sites)))[1])
            _,normal,_=self.face_probe.closest(point)
            u=np.eye(3)[np.argmin(abs(normal))];u-=normal*np.dot(u,normal);u/=np.linalg.norm(u)
            v=np.cross(normal,u);delta=self.sites[ids]-point
            coords=np.column_stack((delta@u,delta@v))
            try:triangulation=Delaunay(coords)
            except QhullError:continue
            for face in triangulation.simplices:
                if 0 not in face:continue
                tri=tuple(sorted(int(x) for x in ids[face]))
                pts=self.sites[list(tri)]
                if np.linalg.norm(pts[:,None]-pts[None,:],axis=2).max()>2*self.cell:continue
                triples.add(tri)
                if len(triples)>MAX_VERTEX_TRIPLES:raise ValueError('多边形顶点候选超出低内存预算')
        return sorted(triples)

    def _locus(self,tri):
        a,b,c=self.sites[list(tri)];u=b-a;v=c-a
        normal=np.cross(u,v);size=np.linalg.norm(normal)
        if size<self.cell**2*1e-6:return None
        normal/=size
        delta=np.linalg.solve(np.array([u,v,normal]),[np.dot(u,u)/2,np.dot(v,v)/2,0.])
        centre=a+delta
        if np.linalg.norm(delta)>self.cell*1.5:return None
        closest,n,_=self.face_probe.closest(centre)
        if np.dot(normal,n)<0:normal=-normal
        if np.dot(normal,n)<.3:return None
        return centre,normal,float(np.dot(closest-centre,normal))

    def phase_cost(self,expected):
        # Cheap projected circumcentres choose a shared phase; exact vertices
        # are rebuilt and validated once the phase is fixed.
        projected=[]
        for tri in self._triples():
            locus=self._locus(tri)
            if locus is None:continue
            centre,normal,t=locus
            projected.append(centre+(t-min(self.p.surface_thickness/2,self.width/2))*normal)
        if not projected:return float('inf')
        return float(np.mean((cKDTree(projected).query(expected)[0]/self.cell)**2))

    def fit_to_terminals(self,terminals):
        super().fit_to_terminals(terminals)
        self._build_vertices()

    def _build_vertices(self):
        vertices=[];outer=[];triples=[];inner=[]
        goal=min(self.p.surface_thickness/2,self.width/2)
        self.standoff=self.p.surface_thickness+self.p.wall/2+self.p.junction_blend+self.pitch
        tolerance=max(1e-6,self.cell*1e-5)
        for tri in self._triples():
            locus=self._locus(tri)
            if locus is None:continue
            centre,normal,t=locus
            span=max(self.cell/2,2*goal)
            ts=t+np.linspace(-span,span,17)
            depth=self.source.depth(centre+ts[:,None]*normal)
            crossings=np.flatnonzero((depth[:-1]>=goal)&(depth[1:]<goal))
            if not len(crossings):continue
            j=min(crossings,key=lambda j:abs((ts[j]+ts[j+1])/2-t))
            lo,hi=ts[j],ts[j+1]
            for _ in range(20):
                mid=(lo+hi)/2
                if self.source.depth((centre+mid*normal)[None])[0]>=goal:lo=mid
                else:hi=mid
            target=centre+(lo+hi)/2*normal
            closest,n,fid=self.source.closest(target)
            if fid not in self.surface.active:continue
            distances=np.linalg.norm(self.sites[list(tri)]-target,axis=1)
            nearest=self.tree.query(target,k=1)[0]
            if np.ptp(distances)>tolerance or distances.min()-nearest>tolerance:continue
            if nearest>self.cell or self.support(target)<=self.material_margin:continue
            # Keep actual triangle-bisector vertices, not an averaged cluster.
            if any(np.linalg.norm(target-q)<tolerance*4 for q in vertices):continue
            out_lo=(lo+hi)/2;out_hi=out_lo+max(4*goal,self.cell/4)
            if self.source.depth((centre+out_hi*normal)[None])[0]>0:continue
            for _ in range(20):
                mid=(out_lo+out_hi)/2
                if self.source.depth((centre+mid*normal)[None])[0]>0:out_lo=mid
                else:out_hi=mid
            surface_vertex=centre+(out_lo+out_hi)/2*normal
            if self.source.closest(surface_vertex)[2] not in self.surface.active:continue
            end_t=(lo+hi)/2
            inner_ts=np.linspace(end_t-max(self.cell,self.standoff*4),end_t,25)
            inner_depth=self.face_probe.depth(centre+inner_ts[:,None]*normal)
            hits=np.flatnonzero((inner_depth[:-1]>=self.standoff)&(inner_depth[1:]<self.standoff))
            inner_target=np.full(3,np.nan)
            if len(hits):
                j=hits[-1];a,b=inner_ts[j],inner_ts[j+1]
                for _ in range(20):
                    mid=(a+b)/2
                    if self.face_probe.depth((centre+mid*normal)[None])[0]>=self.standoff:a=mid
                    else:b=mid
                inner_target=centre+(a+b)/2*normal
                if self.source.depth(inner_target[None])[0]<self.material_margin:inner_target[:]=np.nan
            vertices.append(target);triples.append(tri);outer.append(surface_vertex);inner.append(inner_target)
        if not vertices:raise ValueError('未找到有效多边形顶点；请减小孔距或调整设计区域')
        self.vertices=np.asarray(vertices);self.outer_vertices=np.asarray(outer)
        self.inner_vertices=np.asarray(inner)
        self.vertex_triples=np.asarray(triples);self.vertex_tree=cKDTree(self.vertices)
        self.stats.update(polygon_vertices=len(vertices),vertex_phase_cost='mean_squared_vertex_distance_over_cell',
                          vertex_tolerance_mm=tolerance,node_contact_radius_mm=self.node_radius,
                          vertex_standoff_mm=self.standoff,vertex_posts_available=int(np.isfinite(self.inner_vertices[:,0]).sum()))

    def candidates(self,point):
        if self.vertex_tree is None:self._build_vertices()
        distances,ids=self.vertex_tree.query(point,k=min(12,len(self.vertices)))
        return [self.vertices[i].copy() for d,i in zip(np.atleast_1d(distances),np.atleast_1d(ids)) if d<self.cell*.8]

    def match_options(self,near):
        # Deliberately NO straight-ray material point, edge or rim fallback.
        return self.candidates(near)

    def vertex_id(self,target):
        distance,index=self.vertex_tree.query(target)
        if distance>self.stats['vertex_tolerance_mm']*4:raise ValueError('外层连杆落点不是登记的多边形顶点')
        return int(index)

    def contact_safe(self,curve,targets):
        """Conservative tube/sheet proximity screen outside allowed node zones.

        An endpoint at a vertex is insufficient if the shaft first hits a rib
        elsewhere. Check the whole shaft including its radius and blend band.
        Empty targets means an unchanged/grazing member may not touch the sheet.
        """
        padding=self.p.wall/2+self.p.junction_blend+self.pitch/2
        depth=self.face_probe.depth(curve)
        ids=np.flatnonzero(depth<self.p.surface_thickness+padding)
        if not len(ids):return True
        gap=np.maximum(-self.field(curve[ids]),0.)
        axial=np.maximum(depth[ids]-self.p.surface_thickness,0.)
        ids=ids[np.hypot(gap,axial)<padding]
        if not len(ids):return True
        nodes=[self.outer_vertices[self.vertex_id(t)] for t in targets]
        for point in curve[ids]:
            closest,_,_=self.face_probe.closest(point)
            if not nodes or min(np.linalg.norm(closest-node) for node in nodes)>self.node_radius:
                self.side_contact_rejections+=1
                return False
        return True

    def core_safe(self,curve):
        """Keep oblique bars below the face layer, including beneath pores."""
        depth=self.source.depth(curve)
        if depth.min()<min(.12,self.width/6):return False
        if self.face_probe.depth(curve).min()<self.standoff-self.pitch*.1:return False
        return self.contact_safe(curve,[])

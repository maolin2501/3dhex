"""Deterministic, direction-guided pore rows shared across CAD faces.

Axis-shaped parts use meridian-spaced rings, with integer repeat counts and
alternating half-row phase. Small, nearly planar patches use a triangular grid.
This is an axis-guided layout, not a general 6-RoSy parameterization solver.
"""
from __future__ import annotations
import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from .adaptive_surface import AdaptiveSurface, MAX_ADAPTIVE_SITES
from .surface_stitch import SurfaceAnchors
from .continuity import MeshProbe


class AlignedSurface(AdaptiveSurface):
    def __init__(self, surface, p, pitch):
        SurfaceAnchors.__init__(self, surface, p, pitch)
        self.material_margin=min(.16,self.width*.23,p.surface_thickness*.23)
        self.phase_margin=min(.18,self.width*.26,p.surface_thickness*.26)
        selected=np.isin(surface.source_face_ids,list(surface.active))
        mesh=trimesh.Trimesh(vertices=surface.source_mesh.vertices.copy(),
            faces=surface.source_mesh.faces[selected],process=False)
        mesh.remove_unreferenced_vertices()
        self.mesh=mesh
        self.layout_phase=(0.,.3)
        if mesh.area/(np.sqrt(3)/2*self.cell**2)>MAX_ADAPTIVE_SITES:
            raise ValueError('定向孔网取点超出低内存预算，请增大曲面孔距或缩小区域')
        self.face_probe=MeshProbe(mesh)
        q=mesh.triangles_center; n=mesh.face_normals; w=mesh.area_faces
        mean=np.sum(n*w[:,None],axis=0); mean/=max(np.linalg.norm(mean),1e-12)
        planar=np.sum(w[abs(n@mean)>.8])/w.sum()>.98
        self.row_count=0
        self.row_repeats=[]
        self.projection_errors=[]
        self.projection_attempts=0
        if planar:
            sites=self._plane_sites(mesh,mean,p.axis)
            layout='planar_triangular_rows'
        else:
            sites=self._ring_sites(mesh,p.axis)
            layout='meridian_ring_rows'
        if len(sites)<3:
            raise ValueError('定向孔网未得到足够曲面取点，请调整轴向或孔距')
        self._set_sites(sites)
        self.web_tree=None
        self.contacts=[]
        self.layout=layout
        self._statistics()

    def _set_sites(self,sites):
        sites=np.asarray(sites)
        if len(sites)<3:raise ValueError('定向孔网未得到足够曲面取点，请调整轴向或孔距')
        # Collapsing projected sites are removed, not relaxed into a random net.
        tree=cKDTree(sites); removed=set()
        for i in range(len(sites)):
            if i not in removed:
                removed.update(j for j in tree.query_ball_point(sites[i],self.cell*.3) if j>i)
        self.sites=sites[[i for i in range(len(sites)) if i not in removed]]
        if len(self.sites)<3:raise ValueError('定向孔网投影后取点重合过多，请调整轴向或孔距')
        if len(self.sites)>MAX_ADAPTIVE_SITES:
            raise ValueError('定向孔网种点数量超出预算')
        self.tree=cKDTree(self.sites); self.neighbours=min(12,len(self.sites))
    def _statistics(self):
        spacing=self.tree.query(self.sites,k=2)[0][:,1]
        self.stats=dict(seam_style='aligned',surface_network='aligned_surface_rows',
            aligned_layout=self.layout,aligned_rows=self.row_count,aligned_row_repeats=self.row_repeats,
            aligned_phase=list(self.layout_phase),
            adaptive_sites=len(self.sites),adaptive_surface_samples=0,
            adaptive_site_spacing_mm=np.percentile(spacing,[0,50,100]).tolist(),
            adaptive_cell_area_mm2=float(self.mesh.area/len(self.sites)),
            aligned_projection_error_mm=np.percentile(self.projection_errors,[50,95,100]).tolist(),
            seam_rail_length_mm=0.,surface_core_connections=0,fixed_face_patterns=False)

    def _accept(self,point,sites,limit):
        self.projection_attempts+=1
        if self.projection_attempts>MAX_ADAPTIVE_SITES*40:
            raise ValueError('定向孔网投影工作量超出预算，请缩小区域或更换轴向')
        target,_,_=self.face_probe.closest(point)
        error=np.linalg.norm(point-target)
        if error<=limit:
            sites.append(target);self.projection_errors.append(float(error))

    def _plane_sites(self,mesh,normal,axis):
        v=np.eye(3)['xyz'.index(axis)]
        v=v-(v@normal)*normal
        if np.linalg.norm(v)<.1:
            v=np.eye(3)[np.argmin(abs(normal))];v-=np.dot(v,normal)*normal
        v/=np.linalg.norm(v);u=np.cross(v,normal)
        origin=np.average(mesh.triangles_center,axis=0,weights=mesh.area_faces)
        coords=np.column_stack(((mesh.vertices-origin)@u,(mesh.vertices-origin)@v))
        h=np.sqrt(3)/2*self.cell
        sites=[]
        for row in range(int(np.floor(coords[:,1].min()/h))-1,int(np.ceil(coords[:,1].max()/h))+2):
            self.row_count+=1
            for col in range(int(np.floor(coords[:,0].min()/self.cell))-1,int(np.ceil(coords[:,0].max()/self.cell))+2):
                point=origin+u*(col+.5*(row%2)+self.layout_phase[0])*self.cell+v*(row+self.layout_phase[1]-.3)*h
                self._accept(point,sites,self.cell*.25)
        return sites

    def _ring_sites(self,mesh,axis):
        k='xyz'.index(axis); ij=[a for a in range(3) if a!=k]
        q=mesh.triangles_center;normals=mesh.face_normals;w=mesh.area_faces
        a=np.column_stack((-normals[:,ij[1]],normals[:,ij[0]]))
        rhs=np.sum(a*q[:,ij],axis=1)
        # Iteratively reweight non-radial fillets, without dropping their faces.
        weight=w.copy()
        for _ in range(4):
            centre=np.linalg.lstsq(a*np.sqrt(weight)[:,None],rhs*np.sqrt(weight),rcond=None)[0]
            residual=abs(a@centre-rhs)
            weight=w/np.maximum(1.,residual/max(self.cell*.3,.1))
        singular=np.linalg.svd(a*np.sqrt(weight)[:,None],compute_uv=False)
        if singular[-1]<.05*singular[0]:
            raise ValueError('当前轴向无法构建定向曲面孔列，请更换结构轴向或使用旧自适应模式')
        tris=mesh.triangles
        low,high=mesh.bounds[:,k]
        count=max(16,int(np.ceil((high-low)/max(self.cell/8,.2))))
        if count>4096:raise ValueError('轴向截面数量超出低内存预算')
        heights=np.linspace(low+1e-4,high-1e-4,count)
        radii=[]; valid_z=[]
        for z in heights:
            mask=(tris[:,:,k].min(axis=1)<=z)&(tris[:,:,k].max(axis=1)>=z)
            t=tris[mask];crossings=[]
            for e in range(3):
                first=t[:,e];last=t[:,(e+1)%3]
                dz=last[:,k]-first[:,k]
                good=(abs(dz)>1e-10)&((first[:,k]-z)*(last[:,k]-z)<=0)
                if good.any():
                    points=first[good]+((z-first[good,k])/dz[good])[:,None]*(last[good]-first[good])
                    crossings.extend(np.linalg.norm(points[:,ij]-centre,axis=1))
            if crossings:
                radii.append(float(np.median(crossings)));valid_z.append(z)
        heights=np.asarray(valid_z);radii=gaussian_filter1d(np.asarray(radii),1.)
        if len(heights)<4:raise ValueError('定向孔网缺少有效截面')
        arc=np.r_[0,np.cumsum(np.hypot(np.diff(heights),np.diff(radii)))]
        h=np.sqrt(3)/2*self.cell
        samples=np.arange(h*self.layout_phase[1],arc[-1],h)
        sites=[]
        for row,s in enumerate(samples):
            z=np.interp(s,arc,heights);r=np.interp(s,arc,radii)
            repeats=max(3,int(round(2*np.pi*r/self.cell)))
            if repeats>MAX_ADAPTIVE_SITES:raise ValueError('环向孔数超出预算')
            self.row_count+=1;self.row_repeats.append(repeats)
            for theta in 2*np.pi*(np.arange(repeats)+.5*(row%2)+self.layout_phase[0])/repeats:
                point=np.zeros(3);point[k]=z;point[ij]=centre+r*np.array([np.cos(theta),np.sin(theta)])
                self._accept(point,sites,self.cell*.65)
        return sites

    def fit_to_terminals(self,terminals):
        """Choose a shared row phase against the unbent BCC intersections.

        This couples both layers BEFORE individual matches. Half-step angular
        phases preserve reflection symmetry; no independent per-face shifting.
        """
        if not terminals:return
        expected=[]
        for point,direction in terminals:
            face,n,fid=self.source.closest(point)
            if fid not in self.surface.active:continue
            cosine=np.dot(n,direction)
            if cosine<.2:continue
            depth=np.linalg.norm(point-face)
            target=point+direction*max(0.,(depth-min(self.p.surface_thickness/2,self.width/2))/cosine)
            expected.append(target)
        if not expected:return
        expected=np.asarray(expected)
        best=None
        initial=self.phase_cost(expected)
        for u in (0.,.5):
            for v in (.3,.55,.8):
                self.layout_phase=(u,v)
                self.row_count=0;self.row_repeats=[];self.projection_errors=[]
                sites=(self._plane_sites(self.mesh,np.sum(self.mesh.face_normals*self.mesh.area_faces[:,None],axis=0)/
                         np.linalg.norm(np.sum(self.mesh.face_normals*self.mesh.area_faces[:,None],axis=0)),self.p.axis)
                       if self.layout=='planar_triangular_rows' else self._ring_sites(self.mesh,self.p.axis))
                self._set_sites(sites)
                score=self.phase_cost(expected)
                if best is None or score<best[0]:
                    best=(score,self.sites.copy(),self.layout_phase,self.row_count,self.row_repeats[:],self.projection_errors[:])
        score,sites,self.layout_phase,self.row_count,self.row_repeats,self.projection_errors=best
        self._set_sites(sites);self._statistics()
        self.stats.update(aligned_phase_fit_samples=len(expected),aligned_phase_cost_before=initial,aligned_phase_cost_after=score)

    def phase_cost(self,expected):
        return float(np.mean(np.maximum(self.phase_margin-self.field(expected),0.)**2))

    def match_options(self,near):
        options=self.candidates(near)
        if self.support(near)>self.material_margin and self.source.depth(near[None])[0]>self.material_margin:options.insert(0,near)
        return options

    def candidates(self,point):
        closest,normal,fid=self.source.closest(point)
        if fid not in self.surface.active:return []
        candidates=[]
        direct=self.inset(closest,normal)
        if self.support(direct)>self.material_margin:candidates.append(direct)
        _,indices=self.tree.query(closest,k=min(10,len(self.sites)))
        a=self.sites[indices[0]]
        for index in indices[1:]:
            b=self.sites[index];axis=b-a;axis/=np.linalg.norm(axis);middle=(a+b)/2
            on_face,n=closest.copy(),normal.copy()
            for _ in range(14):
                target=self.inset(on_face,n);error=np.dot(target-middle,axis)
                if abs(error)<1e-5:break
                tangent=axis-np.dot(axis,n)*n;size=np.dot(tangent,tangent)
                if size<.015:break
                delta=-error*tangent/size
                delta*=min(1.,self.cell/2/max(np.linalg.norm(delta),1e-9))
                on_face,n,_=self.face_probe.closest(on_face+delta)
            target=self.inset(on_face,n)
            if np.linalg.norm(target-point)<self.cell*.65 and self.support(target)>self.material_margin:
                if not any(np.linalg.norm(target-q)<.1 for q in candidates):candidates.append(target)
        return candidates

    def match(self,boundary,inner,direction,step):
        """Straight candidates scored by direction BEFORE nearest distance."""
        near=boundary
        # Continue the original shaft toward the sheet mid-depth first.
        distances=np.linspace(0,self.cell,33)
        ray=boundary+distances[:,None]*direction
        depth=self.source.depth(ray)
        goal=min(self.p.surface_thickness/2,self.width/2)
        hits=np.flatnonzero(depth<goal)
        if len(hits) and hits[0]>0:
            j=hits[0];a,b=distances[j-1],distances[j]
            for _ in range(12):
                t=(a+b)/2
                if self.source.depth((boundary+t*direction)[None])[0]>goal:a=t
                else:b=t
            near=boundary+(a+b)/2*direction
        options=self.match_options(near)
        scored=[]
        for target in options:
            delta=target-inner;length=np.linalg.norm(delta)
            if length<self.p.wall:continue
            cosine=np.dot(delta,direction)/length
            if cosine<np.cos(np.deg2rad(50)):continue
            shift=np.linalg.norm(target-boundary-direction*np.dot(target-boundary,direction))
            if shift>self.cell*.45:continue
            contact=np.asarray(self.contacts)
            close=np.linalg.norm(contact-target,axis=1) if len(contact) else np.array([])
            if np.count_nonzero(close<self.p.wall*.6)>=2:continue
            curve=np.linspace(inner,target,max(2,int(np.ceil(length/step))+1))
            if self.source.depth(curve).min()<min(.12,self.width/6):continue
            if hasattr(self,'contact_safe') and not self.contact_safe(curve,[target]):continue
            if hasattr(self,'segment_guard') and not self.segment_guard.clear(inner,target):continue
            cost=5*(1-cosine)+shift/self.cell+.2*np.count_nonzero(close<self.p.wall)
            scored.append((cost,target,curve,float(np.degrees(np.arccos(np.clip(cosine,-1,1))))))
        if not scored:return None
        return min(scored,key=lambda item:item[0])[1:]

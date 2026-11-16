"""One restricted Voronoi pore network over all active CAD faces.

CAD face IDs do not partition the material field. Deterministic farthest-point
seeding and surface Lloyd relaxation give variable polygonal cells; their
shared bisectors are the ribs. No independently capped seam rods are added.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
import trimesh
from .continuity import MeshProbe
from .surface_stitch import SurfaceAnchors

MAX_ADAPTIVE_SITES = 5000
MAX_SURFACE_SAMPLES = 320_000
QUERY_BATCH = 16384


class AdaptiveSurface(SurfaceAnchors):
    def __init__(self, surface, p, pitch):
        super().__init__(surface, p, pitch)
        selected = np.isin(surface.source_face_ids, list(surface.active))
        mesh = trimesh.Trimesh(vertices=surface.source_mesh.vertices.copy(),
                              faces=surface.source_mesh.faces[selected], process=False)
        self.face_probe = MeshProbe(mesh)
        area = mesh.area
        count = max(4, int(round(area/(np.sqrt(3)/2*self.cell**2))))
        if count > MAX_ADAPTIVE_SITES:
            raise ValueError('自适应曲面取点数量超出低内存预算；请增大曲面孔距或缩小设计区域')
        sample_count = max(1024, count*64)
        if sample_count > MAX_SURFACE_SAMPLES:
            raise ValueError('自适应曲面采样超出低内存预算；请增大曲面孔距')
        # Area sampling does not depend on how the same triangles are labelled
        # as CAD faces. A fixed seed makes reloads and export audits repeatable.
        rng = np.random.default_rng(71329)
        tri = rng.choice(len(mesh.faces), sample_count, p=mesh.area_faces/area)
        uv = rng.random((sample_count,2)); uv[uv.sum(axis=1)>1] = 1-uv[uv.sum(axis=1)>1]
        triangles = mesh.triangles[tri]
        samples = (triangles[:,0]+uv[:,0,None]*(triangles[:,1]-triangles[:,0])
                   +uv[:,1,None]*(triangles[:,2]-triangles[:,0]))
        del triangles
        sites = np.empty((count,3))
        nearest = np.full(sample_count, np.inf)
        index = int(np.argmax(np.linalg.norm(samples-samples.mean(axis=0),axis=1)))
        for i in range(count):
            sites[i] = samples[index]
            delta = samples-sites[i]
            nearest = np.minimum(nearest,np.einsum('ij,ij->i',delta,delta))
            index = int(np.argmax(nearest))
        # Relax the sites on the actual curved surface, not in separate UV
        # charts. Most cells approach hexagons without requiring six sides.
        for _ in range(6):
            labels = cKDTree(sites).query(samples)[1]
            weights = np.bincount(labels,minlength=count)
            means = np.column_stack([np.bincount(labels,weights=samples[:,axis],minlength=count)
                                      for axis in range(3)])
            for i in np.flatnonzero(weights):
                target = means[i]/weights[i]
                sites[i] = self.face_probe.closest(target)[0]
        # Remove collapsed sites on narrow trimmed boundaries. Duplicate sites
        # cannot define a bisector and must not create NaNs or micro-cells.
        tree = cKDTree(sites); removed = set()
        for i in range(count):
            if i in removed: continue
            removed.update(j for j in tree.query_ball_point(sites[i],self.cell*.2) if j>i)
        self.sites = sites[[i for i in range(count) if i not in removed]]
        if len(self.sites) < 2:
            raise ValueError('设计区域过小，无法构建自适应孔网')
        self.tree = cKDTree(self.sites)
        self.neighbours = min(12,len(self.sites))
        normals = mesh.face_normals[tri]
        inset = samples-normals*min(p.surface_thickness/2,self.width/2)
        # A bounded fallback set of actual, supported web material. It is not
        # an independent skeleton or a new set of caps at a CAD seam.
        phi = self.field(inset)
        depth = self.source.depth(inset)
        valid = (phi>self.width/4) & (depth>min(self.width/5,p.surface_thickness/5)) & (depth<p.surface_thickness*.8)
        self.web_tree = cKDTree(inset[valid]) if valid.any() else None
        distances = self.tree.query(self.sites,k=2)[0][:,1]
        self.stats = dict(seam_style='adaptive', surface_network='shared_surface_voronoi',
            adaptive_sites=len(self.sites), adaptive_surface_samples=sample_count,
            adaptive_site_spacing_mm=np.percentile(distances,[0,50,100]).tolist(),
            adaptive_cell_area_mm2=float(area/len(self.sites)), seam_rail_length_mm=0.,
            surface_core_connections=0, fixed_face_patterns=False)

    def field(self, points):
        """Material near shared cell boundaries; continuous across CAD IDs.

        Signed distance to each candidate bisector uses squared distance
        differences divided by site separation, not a nearest-point dot mask.
        Only small K-neighbour arrays are allocated per query batch.
        """
        from .surface import smooth_union
        result = np.empty(len(points),dtype=np.float32)
        for start in range(0,len(points),QUERY_BATCH):
            q = points[start:start+QUERY_BATCH]
            d, ids = self.tree.query(q,k=self.neighbours)
            first = self.sites[ids[:,0]]
            phi = None
            for j in range(1,self.neighbours):
                other = self.sites[ids[:,j]]
                separation = np.linalg.norm(other-first,axis=1)
                gap = (d[:,j]**2-d[:,0]**2)/(2*separation)
                candidate = self.width/2-gap
                phi = candidate if phi is None else smooth_union(phi,candidate,self.p.pore_rounding)
            result[start:start+len(q)] = phi
        return result

    def support(self, target):
        _,_,fid = self.source.closest(target)
        depth = self.source.depth(target[None])[0]
        if fid not in self.surface.active: return self.p.shell-depth
        support = min(self.field(target[None])[0],self.p.surface_thickness-depth)
        if self.surface.edge_tree is not None:
            support = max(support,self.p.edge_width-self.surface.edge_tree.query(target)[0])
        return float(support)

    def anchor(self, point):
        closest,normal,fid = self.source.closest(point)
        if fid not in self.surface.active: return None
        _,indices = self.tree.query(closest,k=min(8,len(self.sites)))
        a = self.sites[indices[0]]
        candidates = []
        for index in indices[1:]:
            b = self.sites[index]; axis = b-a; axis /= np.linalg.norm(axis)
            middle = (a+b)/2
            on_face,n = closest.copy(),normal.copy()
            for _ in range(12):
                target = self.inset(on_face,n)
                error = np.dot(target-middle,axis)
                if abs(error) < 1e-5: break
                tangent = axis-np.dot(axis,n)*n
                squared = np.dot(tangent,tangent)
                if squared < .015: break
                delta = -error*tangent/squared
                delta *= min(1.,self.cell/2/max(np.linalg.norm(delta),1e-9))
                on_face,n,_ = self.face_probe.closest(on_face+delta)
            target = self.inset(on_face,n)
            distance = np.linalg.norm(target-point)
            if distance < self.cell and self.support(target)>min(self.width/8,self.pitch/2):
                candidates.append((distance,target))
        if candidates: return min(candidates,key=lambda item:item[0])[1]
        if self.web_tree is not None:
            distances,ids = self.web_tree.query(point,k=min(16,len(self.web_tree.data)))
            for distance,index in zip(np.atleast_1d(distances),np.atleast_1d(ids)):
                target = self.web_tree.data[index]
                if distance < self.cell and self.support(target) > min(self.width/8,self.pitch/2):
                    return target.copy()
        raise ValueError(f'自适应孔网没有找到可靠杆端落点，已停止输出 (face={fid})')

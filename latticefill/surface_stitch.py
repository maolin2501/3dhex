"""Discrete pore-web to core connections, without a continuous CAD seam rail.

Seams locate truncated ribs, not new ribs. Each connected material interval on
either chart gets an inward curved branch to an existing BCC member. Sampling
and spatial indices have explicit budgets independent of the voxel grid.
"""
from __future__ import annotations

from collections import defaultdict
import itertools
import numpy as np
from scipy.spatial import cKDTree
from .continuity import MeshProbe, chart_point, chart_scales, web_target, bezier

MAX_STITCH_SAMPLES = 2_000_000


def resample(path, step):
    pieces = []
    for a, b in zip(path, path[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b-a)/step)))
        pieces.append(a+np.arange(n)[:, None]/n*(b-a))
    return np.vstack(pieces+[path[-1:]])


def seam_paths(edges, pairs, vertices):
    """Walk each face-pair edge graph once; do not duplicate tessellation joins."""
    groups = defaultdict(list)
    for edge, pair in zip(edges, pairs):
        groups[tuple(sorted(pair))].append(tuple(map(int, edge)))
    for pair, group in sorted(groups.items()):
        adjacent = defaultdict(set)
        remaining = set(tuple(sorted(e)) for e in group)
        for a, b in remaining:
            adjacent[a].add(b); adjacent[b].add(a)
        while remaining:
            ends = sorted(v for e in remaining for v in e if len(adjacent[v]) != 2)
            start = ends[0] if ends else min(min(e) for e in remaining)
            walk = [start]
            while True:
                candidates = sorted(v for v in adjacent[walk[-1]]
                    if tuple(sorted((v, walk[-1]))) in remaining)
                if not candidates: break
                nxt = candidates[0]
                remaining.remove(tuple(sorted((walk[-1], nxt))))
                walk.append(nxt)
                if nxt == start or len(adjacent[nxt]) != 2: break
            yield pair, vertices[walk]


def material_runs(values, closed=False):
    indices = np.flatnonzero(values > .01)
    if not len(indices): return []
    runs = list(np.split(indices, np.flatnonzero(np.diff(indices) > 1)+1))
    if closed and len(runs) > 1 and runs[0][0] == 0 and runs[-1][-1] == len(values)-1:
        runs = [np.r_[runs[-1], runs[0]]]+runs[1:-1]
    return runs


class SurfaceAnchors:
    """Find real material on a trimmed face; never fall back to a fake seam."""
    def __init__(self, surface, p, pitch):
        self.surface, self.p, self.pitch = surface, p, pitch
        self.source = MeshProbe(surface.source_mesh, surface.source_face_ids)
        self.probes = {}
        self.cell, self.width = p.surface_cell or p.cell, p.surface_width or p.wall

    def probe(self, fid):
        if fid not in self.probes:
            self.probes[fid] = MeshProbe(self.surface.meshes[fid])
        return self.probes[fid]

    def support(self, target):
        from .surface import pattern_field
        _, _, fid = self.source.closest(target)
        depth = self.source.depth(target[None])[0]
        if fid not in self.surface.active: return self.p.shell-depth
        material = min(pattern_field(target[None], self.surface.charts[fid],
            self.p.surface_pattern, self.cell, self.width, self.p.pore_rounding)[0],
            self.p.surface_thickness-depth)
        if self.surface.edge_tree is not None:
            material = max(material, self.p.edge_width-self.surface.edge_tree.query(target)[0])
        return material

    def inset(self, point, normal):
        clearance = min(self.p.surface_thickness/4, self.width/4)
        target = point-normal*min(self.p.surface_thickness/2, self.width/2)
        for _ in range(8):
            if self.source.depth(target[None])[0] >= clearance: break
            closest, normal, _ = self.source.closest(target)
            target = closest-normal*(clearance+1e-5)
        if self.source.depth(target[None])[0] < 0:
            raise ValueError('曲面取点连接越出原实体；请检查局部壁厚和曲面法向')
        return target

    def anchor(self, point):
        from .surface import chart_uv
        closest, normal, fid = self.source.closest(point)
        if fid not in self.surface.active: return None
        chart = self.surface.charts[fid]; probe = self.probe(fid)
        scales = chart_scales(chart, self.p.surface_pattern, self.cell)
        uv = np.array(chart_uv(closest[None], chart)).ravel()*scales
        # Nearby web alternatives matter at trimmed freeform chart boundaries.
        offsets = [(0, 0)]+[q for q in itertools.product((-1, 0, 1), repeat=2) if q != (0, 0)]
        choices = []
        for offset in offsets:
            target_uv = web_target(uv+np.asarray(offset)*self.cell/2, self.p.surface_pattern, self.cell)
            guess = closest if chart['kind'] == 'projection' else chart_point(target_uv, chart, scales)
            target, normal, _ = probe.closest(guess)
            if chart['kind'] == 'projection':
                for _ in range(12):
                    actual = np.array(chart_uv(target[None], chart)).ravel()*scales
                    delta = (target_uv-actual)/scales
                    if np.linalg.norm(delta) < 1e-5: break
                    basis = np.array([chart['u'], chart['v'], normal])
                    step = np.linalg.lstsq(basis, [*delta, 0.], rcond=None)[0]
                    step *= min(1., self.cell/max(np.linalg.norm(step), 1e-9))
                    target, normal, _ = probe.closest(target+step)
            target = self.inset(target, normal)
            if self.support(target) > min(self.width/8, self.pitch/2):
                choices.append(target)
                if offset == (0, 0): return target
        if choices:
            return min(choices, key=lambda q: np.linalg.norm(q-point))
        # Only actual protected-face rims may be used as a final fallback.
        if self.surface.edge_tree is not None:
            distance, index = self.surface.edge_tree.query(closest)
            if distance < self.cell:
                target = self.surface.edge_tree.data[index]
                _, normal, _ = self.source.closest(target)
                target = self.inset(target, normal)
                if self.support(target) > .01: return target
        raise ValueError(f'曲面 {fid} 的截断孔边未找到可靠连接点；已停止，未添加分界直筋')

    def seam_anchor(self, fid, point):
        """Retreat onto the SAME truncated web before offsetting into the sheet.

        Offsetting directly at a CAD seam can select the opposite chart (whose
        corresponding location is a hole). That makes an apparently attached
        cap. Require positive support in the actual nearest-face material.
        """
        from .surface import pattern_field
        probe = self.probe(fid)
        point, normal, _ = probe.closest(point)
        threshold = min(self.width/8, self.pitch/2)
        u = np.eye(3)[np.argmin(abs(normal))]
        u -= np.dot(u,normal)*normal; u /= np.linalg.norm(u)
        v = np.cross(normal,u)
        angles = np.arange(16)*2*np.pi/16
        # Near high-curvature trim corners the normal offset can change charts
        # over more than one rib width. Search no farther than half a pore.
        radii = sorted(set(min(r,self.cell/2) for r in
                           [0., .2*self.width, .4*self.width, .7*self.width,
                            self.width, 1.5*self.width, 2*self.width,
                            3*self.width, self.cell/2]))
        for radius in radii:
            choices = []
            for angle in (angles if radius else (0.,)):
                on_face, n, _ = probe.closest(point+radius*(np.cos(angle)*u+np.sin(angle)*v))
                material = pattern_field(on_face[None], self.surface.charts[fid],
                    self.p.surface_pattern,self.cell,self.width,self.p.pore_rounding)[0]
                if material < threshold: continue
                target = self.inset(on_face,n)
                if self.support(target) > threshold:
                    choices.append((target,n))
            if choices:
                return min(choices,key=lambda q: np.linalg.norm(q[0]-point))
        raise ValueError(f'曲面 {fid} 接缝孔边取点未落在实际材料内；已停止，未输出悬空连接 (point={point.tolist()})')


class SurfaceStitches:
    def __init__(self, surface, p, pitch):
        from .surface import pattern_field
        sampler = surface.anchor_sampler; source = sampler.source
        self.radius = (p.surface_width or p.wall)/2
        self.step = min(pitch/2, self.radius/3, .15)
        self.paths = []; self.anchors = []; count = 0
        # Targets lie on existing members at least one rod radius behind the
        # sheet, so both sides connect through the core, not along a seam bar.
        core = np.vstack([path[::max(1, int(.6/surface.network.step))]
                          for path in surface.network.paths])
        core = core[source.depth(core) >= p.surface_thickness+p.wall/2]
        if not len(core):
            raise ValueError('实体内部没有足够空间连接曲面孔边；请减小面层厚度或单元')
        tree = cKDTree(core)
        cell, width = p.surface_cell or p.cell, p.surface_width or p.wall
        candidates = []; seam_length = 0.
        for pair, path in seam_paths(surface.design_seam_edges, surface.design_seam_pairs,
                                     surface.source_mesh.vertices):
            seam_length += np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
            closed = np.linalg.norm(path[0]-path[-1]) < 1e-8
            points = resample(path, min(pitch/2, width/4, cell/40))
            if closed: points = points[:-1]
            fields = [pattern_field(points, surface.charts[fid], p.surface_pattern,
                                    cell, width, p.pore_rounding) for fid in pair]
            for side, fid in enumerate(pair):
                for run in material_runs(fields[side], closed):
                    # Where both charts have material they already overlap.
                    if np.max(np.minimum(fields[side][run], fields[1-side][run])) > width/6:
                        continue
                    index = run[np.argmax(fields[side][run])]
                    if surface.edge_tree is not None and surface.edge_tree.query(points[index])[0] < p.edge_width/2:
                        # Already joined to the genuine protected/open rim.
                        continue
                    anchor, normal = sampler.seam_anchor(fid, points[index])
                    candidates.append((anchor, normal))
        seen = set()
        for anchor, normal in candidates:
            key = tuple(np.rint(anchor/(self.radius/2)).astype(int))
            if key in seen: continue
            seen.add(key)
            distance, index = tree.query(anchor)
            if distance > 1.5*p.cell:
                raise ValueError('曲面孔边距内部晶格过远，无法进行局部平滑连接')
            target = core[index]
            # Depart into the wall. A curved branch, not a line running along
            # a CAD boundary, absorbs phase mismatch between the two charts.
            curve = bezier(anchor, anchor-normal*min(distance*.4, p.wall),
                           target+(anchor-target)*.2, target, self.step)
            for iteration in range(16):
                depths = source.depth(curve)
                bad = np.flatnonzero(depths < min(self.radius*.6, p.surface_thickness/4))
                if not len(bad): break
                if iteration: curve[1:-1] = .5*curve[1:-1]+.25*(curve[:-2]+curve[2:])
                for j in bad:
                    if j in (0, len(curve)-1): continue
                    point, inward, _ = source.closest(curve[j])
                    curve[j] = point-inward*min(self.radius*.6, p.surface_thickness/4)
            curve = resample(curve, self.step)
            if np.min(source.depth(curve)) < -1e-5:
                raise ValueError('曲面局部转接越出原实体，未输出悬空连杆')
            count += len(curve)
            if count > MAX_STITCH_SAMPLES:
                raise ValueError('曲面连接超过低内存采样预算；请增大孔距或缩小实体')
            self.paths.append(curve); self.anchors.append((anchor, target))
        self.tree = cKDTree(np.vstack(self.paths)) if self.paths else None
        self.stats = dict(seam_style='stitch', seam_rail_length_mm=0.,
            stitched_seam_length_mm=float(seam_length), surface_core_connections=len(self.paths),
            stitch_samples=count, stitch_diameter_mm=2*self.radius,
            min_stitch_anchor_material_margin_mm=float(min(
                (sampler.support(a) for a,_ in self.anchors), default=0.)))

    def field(self, points):
        out = np.full(len(points), -np.inf, dtype=np.float32)
        if self.tree is not None:
            for i in range(0, len(points), 65536):
                out[i:i+65536] = self.radius-self.tree.query(points[i:i+65536],
                    distance_upper_bound=4*self.radius)[0]
        return out

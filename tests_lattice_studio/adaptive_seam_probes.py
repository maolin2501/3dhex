"""Independent cross-CAD-label paths in the shared pore web, for STL audit.

These probes do not generate/repair geometry. They sample expected sheet
material on BOTH sides of each actual CAD seam, avoiding protected rims.
"""
import numpy as np


def seam_crossings(surface, p):
    adaptive = surface.adaptive
    if adaptive is None:
        return [], []
    paths, pairs = [], []
    width = p.surface_width or p.wall
    half = min(.3, width*.4)
    offsets = np.linspace(-half, half, 9)
    for a, b in surface.seam_segments:
        length = np.linalg.norm(b-a)
        if length < 1e-8:
            continue
        tangent = (b-a)/length
        for t in (np.arange(max(1, int(np.ceil(length/.25))))+.5)/max(1, int(np.ceil(length/.25))):
            centre = a+t*(b-a)
            if surface.edge_tree is not None and surface.edge_tree.query(centre)[0] < p.edge_width+.2:
                continue
            closest, normal, _ = adaptive.source.closest(centre)
            inset = adaptive.inset(closest, normal)
            if adaptive.field(inset[None])[0] < width*.4:
                continue
            across = np.cross(normal, tangent)
            norm = np.linalg.norm(across)
            if norm < .1:
                continue
            across /= norm
            points, labels = [], []
            for offset in offsets:
                point, n, fid = adaptive.source.closest(centre+offset*across)
                points.append(adaptive.inset(point, n))
                labels.append(fid)
            if labels[0] == labels[-1] or not set(labels) <= surface.active:
                continue
            curve = np.asarray(points)
            depth = adaptive.source.depth(curve)
            phi = np.minimum(adaptive.field(curve), p.surface_thickness-depth)
            if phi.min() < width*.25 or depth.min() < width*.25:
                continue
            if np.linalg.norm(np.diff(curve,axis=0),axis=1).max() > .11:
                continue
            paths.append((curve[0], curve[-1], curve))
            pairs.append(tuple(sorted((labels[0], labels[-1]))))
    return paths, pairs

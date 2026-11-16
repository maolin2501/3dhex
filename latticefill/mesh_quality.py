"""Topology-preserving display meshes; never force thin lattices to 220k faces."""
import numpy as np
import trimesh
import manifold3d as manifold
from .mesh_memory import release_cache

DISPLAY_VERSION = 2
DIRECT_DISPLAY_FACE_THRESHOLD = 1_000_000


def as_manifold(mesh):
    data = manifold.Mesh64(np.ascontiguousarray(mesh.vertices, dtype=np.float64),
                           np.ascontiguousarray(mesh.faces, dtype=np.uint64))
    solid = manifold.Manifold(data)
    if solid.status() != manifold.Error.NoError or solid.is_empty():
        raise ValueError('网格不是有效的定向流形，不能安全减面；请提高采样精度重新生成')
    return solid


def as_trimesh(solid):
    data = solid.to_mesh64()
    return trimesh.Trimesh(np.array(data.vert_properties[:,:3]), np.array(data.tri_verts), process=False)


def clean_fragments(mesh, wall):
    release_cache(mesh)
    solid = as_manifold(mesh)
    if len(mesh.faces)>DIRECT_DISPLAY_FACE_THRESHOLD:
        del solid
        from .mesh_memory import clean_components_batched
        return clean_components_batched(mesh,wall)
    parts = solid.decompose()
    signed_volumes = np.array([p.volume() for p in parts])
    volumes = np.abs(signed_volumes)
    threshold = wall**3/10
    tiny = volumes < threshold
    tiny[np.argmax(volumes)] = False
    removed_volume = float(volumes[tiny].sum())
    if removed_volume > .005*volumes.sum():
        raise ValueError('孤立碎片过多（超过材料体积的 0.5%），不能自动修补；请提高采样精度或缩小设计区域')
    kept = [p for p,remove in zip(parts,tiny) if not remove]
    if tiny.any():
        # Disjoint pieces: concatenation preserves topology; no spatial weld.
        mesh = trimesh.util.concatenate([as_trimesh(p) for p in kept])
    return mesh, dict(components=len(kept), removed_fragments=int(tiny.sum()),
                      removed_volume=removed_volume, fragment_threshold_mm3=threshold,
                      volume=float(signed_volumes[~tiny].sum()))


def safe_display(mesh, wall, pitch):
    release_cache(mesh)
    solid = as_manifold(mesh)
    if len(mesh.faces) > DIRECT_DISPLAY_FACE_THRESHOLD:
        # Large strut networks can have thousands of tiny components. Avoid
        # expensive global simplification/decomposition just for rendering.
        # Retain the topology rather than target an arbitrary display budget.
        # Manifold has validated the original indices. Do not make a second
        # full-size mesh or build Trimesh's whole edge caches just for display.
        candidate = mesh
        return candidate, dict(version=DISPLAY_VERSION,method='original_large',
            tolerance_mm=0,watertight=True,winding_consistent=True,components=None,
            topology_preserved=True,volume_error=0,faces=len(candidate.faces))
    tolerance = min(wall/12, pitch/4, .05)
    simplified = solid.simplify(tolerance)
    original_components = len(solid.decompose())
    simplified_components = len(simplified.decompose())
    relative_volume_error = abs(simplified.volume()-solid.volume())/max(abs(solid.volume()),1e-12)
    accepted = (simplified.status() == manifold.Error.NoError and not simplified.is_empty()
                and original_components == simplified_components and relative_volume_error <= .02)
    candidate = as_trimesh(simplified if accepted else solid)
    if not candidate.is_watertight or not candidate.is_winding_consistent:
        candidate = as_trimesh(solid)
        accepted = False
    if not candidate.is_watertight or not candidate.is_winding_consistent:
        raise ValueError('显示网格质量检查未通过，已阻止发布破碎网格')
    return candidate, dict(version=DISPLAY_VERSION, method='manifold_tolerance' if accepted else 'original',
        tolerance_mm=tolerance if accepted else 0, watertight=True, winding_consistent=True,
        components=original_components, volume_error=relative_volume_error if accepted else 0,
        faces=len(candidate.faces))


def export_display(mesh, path, wall, pitch, centre):
    display, quality = safe_display(mesh, wall, pitch)
    # Keep float32 GPU coordinates near zero. The graph carries the placement.
    transform = np.eye(4)
    transform[:3,3] = centre
    scene = trimesh.Scene()
    scene.add_geometry(display, transform=transform)
    scene.export(path)
    # Verify the actual GLB without spatial welding (which can merge thin rods).
    check = trimesh.load_scene(path, process=False)
    for m in check.geometry.values():
        # Validate the actual serialized topology, without full edge-sort caches.
        solid=as_manifold(m)
        del solid
    return quality

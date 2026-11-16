"""核心流水线: 体素化 SDF → 皮壳 + 晶格布尔组合 → Marching Cubes。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh
from scipy import ndimage
from skimage import measure

from .lattices import lattice_phi

# 计算规模上限；低内存场分块不消除完整输出网格的内存需求。
MAX_VOXELS = 400_000_000


@dataclass
class FillParams:
    lattice: str = "gyroid"  # 晶格类型
    cell: float = 5.0        # 单元大小 (mm)
    wall: float = 0.8        # 晶格壁厚 (mm)
    shell: float = 1.0       # 保留的表面皮壳厚度 (mm)
    preserve_thicknesses: dict = field(default_factory=dict)  # CAD face overrides, persisted in checkpoints
    layer: float | None = None  # 层高 (mm), hexcell/hex3d 的 Z 向周期, 默认取 cell
    axis: str = "z"          # 晶格轴向 (蜂窝孔道方向)
    res: int = 8             # 每个单元的体素数 (分辨率)
    blur: float = 0.0        # 隐式场高斯平滑 sigma (体素数), 消除体素台阶
    smooth: int = 0          # Taubin 平滑迭代次数
    preview: str | None = None  # 若给定路径, 保存中间高度的切片预览 PNG
    stats: dict = field(default_factory=dict)  # 运行统计 (输出用)
    pitch: float | None = None  # GUI 可直接指定采样间距
    max_voxels: int = MAX_VOXELS
    region_depth: float = 5.0  # 所选面向实体内的近似距离带深度
    boundary_blur: float = 0.0  # 仅平滑实体裁切边界，不侵蚀晶格场
    unselected: str = "solid"  # solid=局部设计外实心; perforate=同一晶格贯穿至未选曲面
    surface_mode: str = 'volume'  # CLI compatibility; GUI defaults to perforated
    surface_thickness: float = .8
    surface_pattern: str = 'hex'
    surface_cell: float = 0.0  # 0 follows the interior cell
    surface_width: float = 0.0  # 0 follows the interior diameter
    pore_rounding: float = 0.0
    junction_blend: float = 0.0
    edge_width: float = 0.0
    surface_continuity: str = 'off'
    seam_style: str = 'rail'  # legacy CLI; GUI defaults to a shared adaptive surface web
    memory_mode: str = 'auto'  # auto switches large grids to the disk-backed path
    work_dir: str | None = None  # isolated worker directory for temporary fields


def _save_slice_preview(phi: np.ndarray, path: str) -> None:
    """把 4 个等距 Z 切片 (材料=白) 拼成一张 PNG, 便于快速目检。"""
    from PIL import Image

    nz = phi.shape[2]
    ks = [int(nz * f) for f in (0.25, 0.45, 0.55, 0.75)]
    tiles = []
    for k in ks:
        img = (phi[:, :, k] > 0).T[::-1]  # 转成常规图像取向
        tiles.append(np.where(img, 255, 40).astype(np.uint8))
    gap = np.full((tiles[0].shape[0], 4), 128, dtype=np.uint8)
    row = np.hstack([t if i == 0 else np.hstack([gap, t]) for i, t in enumerate(tiles)])
    scale = max(1, 700 // row.shape[1])
    im = Image.fromarray(row).resize(
        (row.shape[1] * scale, row.shape[0] * scale), Image.NEAREST
    )
    im.save(path)
    print(f"  切片预览已保存: {path}")


def _voxelize_sdf(mesh: trimesh.Trimesh, pitch: float):
    """体素化实体并返回 (sdf, origin)。

    sdf 为近似符号距离场 (mm), 内正外负; origin 为体素 (0,0,0) 中心的世界坐标。
    """
    vox = mesh.voxelized(pitch=pitch)
    vox = vox.fill()
    inside = np.asarray(vox.matrix, dtype=bool)
    origin = np.asarray(vox.transform[:3, 3], dtype=np.float64)

    # 四周补空体素, 保证距离变换与 marching cubes 在边界处正确闭合
    pad = 3
    inside = np.pad(inside, pad, mode="constant", constant_values=False)
    origin = origin - pad * pitch

    sdf = ndimage.distance_transform_edt(inside, sampling=pitch).astype(np.float32)
    sdf -= ndimage.distance_transform_edt(~inside, sampling=pitch).astype(np.float32)
    return sdf, origin


def _face_distance(mesh, shape, origin, pitch):
    """Approximate distance to ONLY the chosen CAD face tessellation.

    The raster has a half-voxel uncertainty. No flood fill is used for faces.
    """
    mask = np.ones(shape, dtype=bool)
    points = mesh.voxelized(pitch=pitch).points
    ijk = np.rint((points - origin) / pitch).astype(np.int64)
    valid = np.all((ijk >= 0) & (ijk < np.asarray(shape)), axis=1)
    ijk = ijk[valid]
    if not len(ijk):
        raise ValueError('所选面不在实体网格内')
    mask[tuple(ijk.T)] = False
    return ndimage.distance_transform_edt(mask, sampling=pitch).astype(np.float32)


def _collapse_pinholes(out: trimesh.Trimesh, max_loop: int = 10, rounds: int = 4) -> None:
    """闭合亚体素针孔: 把小的开放边环 (<= max_loop 顶点) 塌缩到质心。

    薄皮壳在体素采样极限附近会产生 1~2 体素尺度的针孔,
    fill_holes 对这类非简单孔常失效; 大孔洞不做处理。
    """
    for _ in range(rounds):
        if out.is_watertight:
            return
        edges_s = out.edges_sorted
        bidx = trimesh.grouping.group_rows(edges_s, require_count=1)
        if len(bidx) == 0:
            return
        # 并查集按连通性把开放边端点聚成环
        parent: dict[int, int] = {}

        def find(a: int) -> int:
            while parent.setdefault(a, a) != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for a, b in edges_s[bidx]:
            ra, rb = find(int(a)), find(int(b))
            if ra != rb:
                parent[ra] = rb
        groups: dict[int, list[int]] = {}
        for v in list(parent):
            groups.setdefault(find(v), []).append(v)
        verts = out.vertices.copy()
        collapsed = 0
        for g in groups.values():
            if len(g) <= max_loop:
                verts[g] = verts[g].mean(axis=0)
                collapsed += 1
        if collapsed == 0:
            return
        out.vertices = verts
        out.merge_vertices()
        out.update_faces(out.nondegenerate_faces())
        out.remove_unreferenced_vertices()


def fill_solid(mesh: trimesh.Trimesh, p: FillParams, *, design_faces=None,
               preserve_faces=None, preserve_regions=None, surface_field=None, progress=None) -> trimesh.Trimesh:
    """对单个实体做皮壳保留 + 晶格填充, 返回结果网格。"""
    def report(percent, message):
        if progress:
            progress(percent, message)

    pitch = p.pitch if p.pitch is not None else p.cell / p.res
    values = [p.cell, p.wall, pitch, p.region_depth]
    if not all(np.isfinite(v) and v > 0 for v in values):
        raise ValueError('单元、线径、采样间距与设计深度必须是有限正数')
    if p.wall >= p.cell:
        raise ValueError('线径/壁厚必须小于单元大小')
    if any(not np.isfinite(v) or v < 0 for v in [p.shell, p.blur, p.boundary_blur, p.smooth]):
        raise ValueError('皮壳与平滑参数不能为负数或非有限值')
    explicit_preserve = preserve_faces is not None or preserve_regions is not None
    groups = (list(preserve_regions) if preserve_regions is not None else
              [(preserve_faces,p.shell)] if preserve_faces is not None and p.shell>0 else [])
    if any(face is None or not np.isfinite(thickness) or thickness<=0 for face,thickness in groups):
        raise ValueError('单面保留区域需要有效曲面和正厚度')
    if p.preserve_thicknesses and preserve_regions is None:
        raise ValueError('单面保留厚度需要对应的曲面分组')
    if p.unselected not in ('solid', 'perforate'):
        raise ValueError('未选区域处理必须为 solid 或 perforate')
    if p.surface_mode not in ('volume', 'perforated'):
        raise ValueError('无效曲面处理模式')
    if p.surface_mode == 'perforated':
        if surface_field is None or not np.isfinite(p.surface_thickness) or p.surface_thickness <= 0:
            raise ValueError('曲面开孔需要有效的面映射和正面层厚度')
        if p.surface_pattern not in ('hex', 'square', 'circle'):
            raise ValueError('无效曲面孔型')
        if p.surface_continuity not in ('auto', 'off'):
            raise ValueError('无效曲面连续连接模式')
        if p.seam_style not in ('atlas', 'vertex', 'aligned', 'adaptive', 'stitch', 'rail'):
            raise ValueError('无效面间过渡模式')
        if p.seam_style in ('vertex','atlas') and (p.lattice!='bcc' or p.surface_continuity!='auto'):
            raise ValueError('顶点连接模式需使用 BCC 且开启自动曲面连续连接')
        if any(not np.isfinite(v) or v < 0 for v in [p.surface_cell,p.surface_width,p.pore_rounding,p.junction_blend,p.edge_width]):
            raise ValueError('曲面孔距、筋宽、圆角和实边不能为负数或非有限值')
        if (p.surface_width or p.wall) >= (p.surface_cell or p.cell):
            raise ValueError('曲面筋宽必须小于曲面孔距')
    ext = mesh.bounds[1] - mesh.bounds[0]
    n_est = np.prod(np.ceil(ext / pitch) + 8)
    if n_est > p.max_voxels:
        raise MemoryError(
            f"体素数约 {n_est:.2e}, 超过上限 {p.max_voxels:.0e}。"
            f"请增大 --cell、减小 --res, 或分实体处理。"
        )
    if p.memory_mode not in ('auto','chunked','dense'):
        raise ValueError('无效内存模式')
    from .lowmem import AUTO_THRESHOLD, fill_chunked
    if p.memory_mode=='chunked' or (p.memory_mode=='auto' and n_est>AUTO_THRESHOLD):
        return fill_chunked(mesh,p,design_faces,preserve_faces,surface_field,report,
                            preserve_regions=groups if explicit_preserve else None)
    p.stats['memory_mode']='dense'
    if pitch > p.wall / 2 and p.wall > 0:
        print(f"  [警告] 体素间距 {pitch:.3f}mm > 壁厚(杆径)/2, 晶格可能断连, 建议提高 --res")
    if pitch > p.shell / 2 and p.shell > 0:
        print(f"  [警告] 体素间距 {pitch:.3f}mm > 皮壳/2, 皮壳可能欠采样, 建议提高 --res")

    print(f"  体素化 (间距 {pitch:.3f} mm) ...")
    report(10, '采样实体边界')
    sdf, origin = _voxelize_sdf(mesh, pitch)
    body_vox = int(np.count_nonzero(sdf > 0))
    if p.boundary_blur > 0:
        ndimage.gaussian_filter(sdf, sigma=p.boundary_blur, output=sdf)
    if p.surface_mode == 'perforated':
        report(22, '建立原曲面开孔映射')
        surface_field.prepare(sdf.shape, origin, pitch, p)
    report(28, '建立面选择距离场')
    # Perforated mode continues ONE periodic lattice through selected and
    # unselected regions. The original body SDF still clips ALL outer/inner
    # surfaces, so nothing is extruded beyond the source outline. Only explicit
    # preserve_faces (or CLI global shell) can add a solid skin.
    region = (_face_distance(design_faces, sdf.shape, origin, pitch)
              if design_faces is not None and p.unselected == 'solid' else None)
    keep = None
    for face,thickness in groups:
        band = np.float32(thickness)-_face_distance(face,sdf.shape,origin,pitch)
        if keep is None:keep=band
        else:np.maximum(keep,band,out=keep)
    if groups:del band
    nx, ny, nz = sdf.shape
    print(f"  网格 {nx}x{ny}x{nz} = {sdf.size:,} 体素")

    # 世界坐标广播向量
    xs = (origin[0] + pitch * np.arange(nx, dtype=np.float32))[:, None, None]
    ys = (origin[1] + pitch * np.arange(ny, dtype=np.float32))[None, :, None]
    zs = (origin[2] + pitch * np.arange(nz, dtype=np.float32))[None, None, :]

    print(f"  计算晶格场 ({p.lattice}, 单元 {p.cell} mm, 壁厚 {p.wall} mm) ...")
    # 沿 Z 分块计算晶格场并与 SDF 组合, 控制内存峰值。
    # 组合规则:
    #   皮壳带: 0 < sdf < shell         → phi1 = min(sdf, shell - sdf)
    #   内部晶格: 实体内 ∩ 晶格材料      → phi2 = min(sdf, lat)
    #   最终: 两者并集, phi > 0 为材料
    phi = np.empty_like(sdf)
    shell32 = np.float32(p.shell)
    pad = 2  # TPMS 数值梯度需要的重叠边界
    chunk = max(4, int(2.0e6 // max(1, nx * ny)))
    for k0 in range(0, nz, chunk):
        k1 = min(nz, k0 + chunk)
        a0, a1 = max(0, k0 - pad), min(nz, k1 + pad)
        X, Y, Z = np.broadcast_arrays(xs, ys, zs[:, :, a0:a1])
        lat = lattice_phi(
            p.lattice, X, Y, Z, p.cell, p.wall, pitch, layer=p.layer, axis=p.axis
        )
        lat = lat[:, :, k0 - a0 : k0 - a0 + (k1 - k0)]
        s = sdf[:, :, k0:k1]
        if p.surface_mode == 'perforated':
            lat = surface_field.combine(lat, s, k0, p)
        if region is not None:
            # Outside the selected face-distance band, retain the original solid.
            lat = np.maximum(lat, region[:, :, k0:k1] - p.region_depth)
        if keep is not None:
            lat = np.maximum(lat, keep[:, :, k0:k1])
            ph = np.minimum(s, lat)
        elif p.shell > 0 and not explicit_preserve:
            ph = np.minimum(s, shell32 - s)
            np.maximum(ph, np.minimum(s, lat), out=ph)
        else:
            ph = np.minimum(s, lat)
        phi[:, :, k0:k1] = ph
        report(35 + int(35 * k1 / nz), '计算晶格与曲面裁切')
    del lat
    del region, keep

    # 统计相对密度 (材料体素 / 实体体素)
    solid_vox = int(np.count_nonzero(phi > 0))
    density = solid_vox / body_vox if body_vox else 0.0
    del sdf

    if p.blur > 0:
        # 场级平滑: 消除体素化/EDT 量化带来的表面台阶。sigma 以体素为单位,
        # 0.8~1.2 通常足够; 过大会侵蚀比 sigma*pitch 更薄的特征
        phi = ndimage.gaussian_filter(phi, sigma=p.blur)

    if p.preview:
        _save_slice_preview(phi, p.preview)

    print("  Marching Cubes 提取网格 ...")
    report(73, '提取三维网格')
    # 等值面从 0 微移: 避免解析距离场恰好在体素点上取零值时产生退化三角形
    level = float(np.float32(1.2345e-3) * pitch)
    if not phi.min() < level < phi.max():
        raise ValueError('当前参数没有可提取的晶格；请增加线径或提高采样精度')
    verts, faces, _, _ = measure.marching_cubes(phi, level=level, spacing=(pitch,) * 3)
    del phi
    verts = verts + origin
    out = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    return _finish_mesh(out,mesh,p,pitch,density,report)


def _finish_mesh(out,mesh,p,pitch,density,report,pretrim_ready=False):
    """Shared finishing semantics for dense and low-memory field backends."""
    low_memory=p.stats.get('memory_mode')=='chunked'
    if pretrim_ready:
        from .mesh_memory import release_cache
    elif low_memory:
        from .mesh_memory import release_cache, smooth_taubin_batched,remove_degenerate_batched
        from .mesh_quality import as_manifold
        remove_degenerate_batched(out)
        release_cache(out)
        # Validate orientation/topology without sorting all triangle edges.
        try:
            solid=as_manifold(out)
        except ValueError:
            # Removing numerically degenerate MC triangles can expose tiny
            # 3/4-edge pinholes. Keep the original bounded-loop repair semantics
            # rather than publishing them or silently discarding components.
            report(84,'检查并修复数值针孔')
            out.fill_holes()
            _collapse_pinholes(out)
            release_cache(out)
            solid=as_manifold(out)
        if solid.volume()<0:out.invert()
        del solid
    else:
        out.update_faces(out.nondegenerate_faces())
        if not out.is_watertight:
            out.fill_holes()
        _collapse_pinholes(out)
        # Our implicit field is positive INSIDE; marching cubes winds inward.
        if out.is_watertight and out.is_winding_consistent and out.volume < 0:
            out.invert()

    if p.smooth > 0 and not pretrim_ready:
        report(85, '平滑网格')
        print(f"  Taubin 平滑 x{p.smooth} ...")
        if low_memory:smooth_taubin_batched(out,p.smooth)
        else:trimesh.smoothing.filter_taubin(out, iterations=p.smooth)

    trimmed = False
    if p.surface_mode == 'perforated':
        from .surface import trim_to_source
        from .mesh_checkpoint import save_pretrim,CHECKPOINT_FACE_MIN
        if low_memory and p.work_dir and not pretrim_ready and len(out.faces)>=CHECKPOINT_FACE_MIN:
            report(87,'保存裁切前恢复检查点')
            save_pretrim(out,p,pitch,density)
        report(88, '按原曲面网格精确裁切开孔面层')
        out, trimmed = trim_to_source(out, mesh, low_memory=low_memory)

    if low_memory:
        from .mesh_quality import as_manifold
        release_cache(out)
        solid=as_manifold(out)
        closed=True;volume=float(abs(solid.volume()))
        del solid
    else:
        closed=bool(out.is_watertight)
        volume=float(abs(out.volume)) if closed else None

    p.stats.update(
        density=density,
        faces=len(out.faces),
        watertight=closed,
        unselected=p.unselected,
        surface_mode=p.surface_mode,
        surface_pattern=p.surface_pattern if p.surface_mode == 'perforated' else None,
        surface_cell=(p.surface_cell or p.cell) if p.surface_mode == 'perforated' else None,
        surface_width=(p.surface_width or p.wall) if p.surface_mode == 'perforated' else None,
        pore_rounding=p.pore_rounding,
        junction_blend=p.junction_blend,
        edge_width=p.edge_width,
        surface_continuity=p.surface_continuity,
        source_trimmed=trimmed,
        pitch=float(pitch),
        volume=volume,
        mesh_density=float(abs(volume / mesh.volume)) if closed and mesh.is_watertight and abs(mesh.volume) > 1e-9 else None,
    )
    print(
        f"  完成: {len(out.faces):,} 面, 相对密度 ~{density:.1%}, "
        f"水密: {'是' if closed else '否'}"
    )
    return out

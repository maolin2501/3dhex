"""STEP / 网格文件载入与实体枚举。

STEP 解析依赖 cascadio (OpenCASCADE 内核, trimesh 自动调用):
.step/.stp → 内部转为 GLB 场景, 每个 STEP 实体成为场景中一个具名几何体。
也直接支持 .stl/.obj/.ply/.glb 等网格格式 (作为单实体处理)。
"""

from __future__ import annotations

import trimesh


def load_solids(path: str, scale: float | None = None) -> list[tuple[str, trimesh.Trimesh]]:
    """载入文件, 返回 [(实体名, 已应用场景变换的网格), ...]。

    scale: 坐标缩放系数。None=自动: STEP 文件经 cascadio 转 GLB 后单位为米,
    自动乘 1000 恢复毫米; 其它格式不缩放。
    """
    obj = trimesh.load(path)
    if scale is None:
        is_step = path.lower().endswith((".step", ".stp"))
        scale = 1000.0 if is_step else 1.0
        if is_step:
            print("  (STEP 单位自动换算: m → mm, 如不符请用 --scale 指定)")
    solids: list[tuple[str, trimesh.Trimesh]] = []
    if isinstance(obj, trimesh.Scene):
        for node in obj.graph.nodes_geometry:
            transform, geom_name = obj.graph[node]
            geom = obj.geometry[geom_name]
            if not isinstance(geom, trimesh.Trimesh) or geom.is_empty:
                continue
            mesh = geom.copy()
            mesh.apply_transform(transform)
            if scale != 1.0:
                mesh.apply_scale(scale)
            solids.append((str(node), mesh))
    elif isinstance(obj, trimesh.Trimesh):
        if scale != 1.0:
            obj.apply_scale(scale)
        # 单网格文件: 按连通体拆分, 若只有一个就整体作为一个实体
        parts = obj.split(only_watertight=False)
        if len(parts) <= 1:
            solids.append(("solid_0", obj))
        else:
            for i, part in enumerate(parts):
                solids.append((f"solid_{i}", part))
    else:
        raise ValueError(f"无法从 {path} 载入网格几何体 (得到 {type(obj).__name__})")
    if not solids:
        raise ValueError(f"{path} 中没有找到任何实体")
    return solids


def describe_solids(solids: list[tuple[str, trimesh.Trimesh]]) -> str:
    """生成实体清单文本 (供 --list 输出)。"""
    lines = ["序号  名称                          包围盒尺寸 (mm)          面数      水密"]
    for i, (name, mesh) in enumerate(solids):
        ext = mesh.bounds[1] - mesh.bounds[0]
        size = f"{ext[0]:.1f} x {ext[1]:.1f} x {ext[2]:.1f}"
        wt = "是" if mesh.is_watertight else "否"
        lines.append(f"[{i:>2}]  {name:<28.28}  {size:<22}  {len(mesh.faces):>7,}  {wt}")
    return "\n".join(lines)


def select_solids(
    solids: list[tuple[str, trimesh.Trimesh]], spec: str | None
) -> list[tuple[int, str, trimesh.Trimesh]]:
    """按选取表达式过滤实体。

    spec 为 None 或 "all" → 全部; 否则为逗号分隔的序号或名称子串, 如 "0,2" 或 "Body1"。
    """
    if spec is None or spec.strip().lower() == "all":
        return [(i, n, m) for i, (n, m) in enumerate(solids)]
    picked: list[tuple[int, str, trimesh.Trimesh]] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if token.isdigit():
            i = int(token)
            if i >= len(solids):
                raise IndexError(f"实体序号 {i} 超出范围 (共 {len(solids)} 个)")
            picked.append((i, *solids[i]))
        else:
            matches = [
                (i, n, m) for i, (n, m) in enumerate(solids) if token.lower() in n.lower()
            ]
            if not matches:
                raise KeyError(f"没有名称包含 {token!r} 的实体")
            picked.extend(matches)
    # 去重并保持顺序
    seen: set[int] = set()
    out = []
    for item in picked:
        if item[0] not in seen:
            seen.add(item[0])
            out.append(item)
    return out

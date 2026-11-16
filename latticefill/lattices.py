"""隐式晶格场定义。

每种晶格类型给出一个标量场 phi(x, y, z)（单位 mm）:
  phi > 0  → 晶格材料内部
  phi = 0  → 晶格壁面
  phi < 0  → 空腔

TPMS 类型 (gyroid / schwarz / diamond / iwp) 为片状晶格: 取 |f| < 壁厚/2,
通过数值梯度把无量纲场 f 换算成近似真实距离, 使 --wall 参数接近实际壁厚。
hex(蜂窝) / grid(立方网格) 直接用解析距离, 壁厚精确。
"""

from __future__ import annotations

import numpy as np

# 可用晶格类型 → 中文说明（供 CLI 帮助与 README 引用）
LATTICE_TYPES = {
    "gyroid": "螺旋二十四面体极小曲面 (TPMS), 各向近似均匀, 自支撑性好",
    "schwarz": "Schwarz-P 极小曲面 (TPMS), 孔道沿三轴正交贯通",
    "diamond": "Schwarz-D 金刚石极小曲面 (TPMS), 刚度较高",
    "iwp": "Schoen I-WP 极小曲面 (TPMS)",
    "hex": "六边形蜂窝壁板, 沿 Z 轴挤出, 竖直孔道贯通 (面内封闭)",
    "grid": "正交立方网格壁板 (注意: 形成封闭立方腔)",
    "hexcell": "立体蜂窝壁板: 六边形棱柱壁 + 层间隔板, 蜂窝腔封闭, 壁厚/层高可调",
    "hex3d": "立体蜂窝连杆框架: 六边形顶点竖杆 + 每层六边形横杆框, 连杆直径可调",
    "bcc": "体心立方 (BCC) 斜连杆, 连杆直径可调",
    "cubic": "立方体边连杆, 连杆直径可调",
    "kelvin": "Kelvin 立体蜂窝: 截角八面体空间杆网, 含六边形和四边形孔面",
}

# 杆状晶格: 厚度参数含义为连杆直径 (其余类型为壁厚)
STRUT_TYPES = frozenset({"hex3d", "bcc", "cubic", "kelvin"})


def _tpms_f(kind: str, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """无量纲 TPMS 场, 输入为已按 2*pi/cell 缩放的坐标。"""
    sin, cos = np.sin, np.cos
    if kind == "gyroid":
        return sin(X) * cos(Y) + sin(Y) * cos(Z) + sin(Z) * cos(X)
    if kind == "schwarz":
        return cos(X) + cos(Y) + cos(Z)
    if kind == "diamond":
        return (
            sin(X) * sin(Y) * sin(Z)
            + sin(X) * cos(Y) * cos(Z)
            + cos(X) * sin(Y) * cos(Z)
            + cos(X) * cos(Y) * sin(Z)
        )
    if kind == "iwp":
        return 2.0 * (cos(X) * cos(Y) + cos(Y) * cos(Z) + cos(Z) * cos(X)) - (
            cos(2 * X) + cos(2 * Y) + cos(2 * Z)
        )
    raise ValueError(f"未知 TPMS 类型: {kind}")


def _tpms_phi(kind, x, y, z, cell, wall, pitch):
    w = np.float32(2.0 * np.pi / cell)
    f = _tpms_f(kind, w * x, w * y, w * z).astype(np.float32, copy=False)
    # 数值梯度归一化: dist ≈ |f| / |∇f|, 让 wall 以 mm 计
    gx, gy, gz = np.gradient(f, pitch)
    g = np.sqrt(gx * gx + gy * gy + gz * gz)
    del gx, gy, gz
    np.maximum(g, np.float32(1e-4), out=g)
    dist = np.abs(f)
    dist /= g
    del g
    return np.float32(wall / 2.0) - dist


def _grid_phi(x, y, z, cell, wall):
    """正交壁板: 到最近晶格平面 (坐标为 cell 整数倍的平面) 的距离。"""
    dx = np.abs(x - cell * np.round(x / cell))
    dy = np.abs(y - cell * np.round(y / cell))
    dz = np.abs(z - cell * np.round(z / cell))
    d = np.minimum(np.minimum(dx, dy), dz)
    return np.float32(wall / 2.0) - d


def _hex_local(x: np.ndarray, y: np.ndarray, cell: float):
    """把 XY 坐标折叠到最近的六边形蜂窝中心, 返回 (px, py, R)。

    cell = 相邻蜂窝中心距 = 六边形对边距 (across-flats), R = 外接圆半径。
    尖顶朝上 (pointy-top) 取向, 用立方坐标取整找最近中心。
    """
    R = cell / np.sqrt(3.0)
    # 像素 → 轴向坐标 (redblobgames pointy-top 约定)
    q = (np.sqrt(3.0) / 3.0 * x - y / 3.0) / R
    r = (2.0 / 3.0 * y) / R
    s = -q - r
    rq, rr, rs = np.round(q), np.round(r), np.round(s)
    dq, dr, ds = np.abs(rq - q), np.abs(rr - r), np.abs(rs - s)
    # 立方坐标取整修正: 误差最大的分量由另外两个反推
    fix_q = (dq > dr) & (dq > ds)
    fix_r = ~fix_q & (dr > ds)
    rq = np.where(fix_q, -rr - rs, rq)
    rr = np.where(fix_r, -rq - rs, rr)
    cx = np.sqrt(3.0) * R * (rq + rr / 2.0)
    cy = 1.5 * R * rr
    return x - cx, y - cy, R


def _hex_edge_dist(px: np.ndarray, py: np.ndarray, cell: float) -> np.ndarray:
    """局部坐标下到六边形边框 (蜂窝壁中面) 的精确距离。"""
    # 尖顶六边形 6 条边的外法向: 0°, ±60°, 180°, ±120° → 支撑函数
    m = np.maximum(
        np.abs(px),
        np.maximum(
            np.abs(0.5 * px + 0.8660254 * py),
            np.abs(0.5 * px - 0.8660254 * py),
        ),
    )
    apothem = cell / 2.0
    return np.abs(apothem - m)  # 蜂窝中心处 = apothem


def _hex_wall_dist(x: np.ndarray, y: np.ndarray, cell: float) -> np.ndarray:
    px, py, _ = _hex_local(x, y, cell)
    return _hex_edge_dist(px, py, cell)


def _hex_phi(x, y, cell, wall):
    return np.float32(wall / 2.0) - _hex_wall_dist(x, y, cell)


def _fold(x: np.ndarray, a: float) -> np.ndarray:
    """到最近 a 整数倍坐标的距离 (周期折叠)。"""
    return np.abs(x - a * np.round(x / a))


def _hexcell_dist(x, y, z, cell, layer):
    """立体蜂窝壁板: 六边形棱柱壁 + 每隔 layer 高度的水平隔板。"""
    d_wall = _hex_wall_dist(x, y, cell)
    dz = _fold(z, layer)
    return np.minimum(d_wall, dz)


def _hex3d_dist(x, y, z, cell, layer):
    """立体蜂窝连杆框架: 到最近连杆轴线的距离。

    竖杆: 沿 Z 连续, 位于六边形顶点 (蜂窝三叉结点);
    横杆: 每隔 layer 高度一层, 沿六边形边框布置。
    """
    px, py, R = _hex_local(x, y, cell)
    # 横杆框: 面内到六边形边框的距离 + Z 向到最近层的距离
    d_edge = _hex_edge_dist(px, py, cell)
    dz = _fold(z, layer)
    d_frame = np.sqrt(d_edge * d_edge + dz * dz)
    del d_edge, dz
    # 竖杆: 到最近六边形顶点的面内距离。利用对称性折叠后只剩 2 个代表顶点:
    # (√3/2 R, R/2) 与 (0, R)
    ax, ay = np.abs(px), np.abs(py)
    d1x = ax - 0.8660254 * R
    d1y = ay - 0.5 * R
    d2y = ay - R
    d_vert = np.minimum(
        np.sqrt(d1x * d1x + d1y * d1y),
        np.sqrt(ax * ax + d2y * d2y),
    )
    return np.minimum(d_frame, d_vert)


def _bcc_dist(x, y, z, cell):
    """BCC 斜连杆: 角点到体心的 8 根对角杆, 镜像折叠后归结为一根线段。"""
    h = cell / 2.0
    u = np.mod(x, cell)
    np.minimum(u, cell - u, out=u)
    v = np.mod(y, cell)
    np.minimum(v, cell - v, out=v)
    w = np.mod(z, cell)
    np.minimum(w, cell - w, out=w)
    # 折叠八分区内的对角线段 (0,0,0)→(h,h,h), 最近点参数 t 截断到线段上
    t = (u + v + w) / 3.0
    np.clip(t, 0.0, h, out=t)
    u -= t
    v -= t
    w -= t
    return np.sqrt(u * u + v * v + w * w)


def _cubic_dist(x, y, z, cell):
    """立方边连杆: 沿三个轴向、过晶格线的圆杆。"""
    dx, dy, dz = _fold(x, cell), _fold(y, cell), _fold(z, cell)
    dxy = np.sqrt(dx * dx + dy * dy)  # 沿 Z 的杆
    dyz = np.sqrt(dy * dy + dz * dz)  # 沿 X 的杆
    dxz = np.sqrt(dx * dx + dz * dz)  # 沿 Y 的杆
    return np.minimum(np.minimum(dxy, dyz), dxz)


def _kelvin_dist(x, y, z, cell):
    """Distance to edges of the BCC Voronoi tessellation (Kelvin cells).

    cell is the conventional cubic repeat, NOT the hexagon across-flats.
    Nearest BCC centre folds to a truncated octahedron with vertices
    permutations of (0, cell/4, cell/2). Symmetry reduces its 36 edges
    to the square/hexagon edge and hexagon/hexagon edge below.
    """
    coords = np.stack(np.broadcast_arrays(x, y, z), axis=-1)
    first = coords - cell * np.round(coords / cell)
    second = coords - cell / 2
    second -= cell * np.round(second / cell)
    use_second = np.sum(second * second, axis=-1) < np.sum(first * first, axis=-1)
    folded = np.sort(np.abs(np.where(use_second[..., None], second, first)), axis=-1)
    a, b, c = folded[..., 2], folded[..., 1], folded[..., 0]
    # Closest point on segment (cell/2, cell/4, 0) -> (cell/2, 0, cell/4)
    t = np.clip((c - b + cell / 4) / 2, 0, cell / 4)
    d1 = (a - cell / 2)**2 + (b - cell / 4 + t)**2 + (c - t)**2
    # Closest point on segment (cell/2, cell/4, 0) -> (cell/4, cell/2, 0)
    t = np.clip((b - a + cell / 4) / 2, 0, cell / 4)
    d2 = (a - cell / 2 + t)**2 + (b - cell / 4 - t)**2 + c*c
    return np.sqrt(np.minimum(d1, d2))


def lattice_phi(
    kind: str,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cell: float,
    wall: float,
    pitch: float,
    layer: float | None = None,
    axis: str = "z",
) -> np.ndarray:
    """返回晶格隐式场 (mm), phi > 0 为晶格材料。

    kind  : 晶格类型, 见 LATTICE_TYPES
    x/y/z : 世界坐标网格 (mm), 同形状数组
    cell  : 单元大小 (mm)
    wall  : 壁厚 (mm); 杆状晶格 (STRUT_TYPES) 时为连杆直径;
            hex/grid/杆状为精确值, TPMS 为近似值
    pitch : 体素间距 (mm), 用于 TPMS 数值梯度归一化
    layer : 层高 (mm), 用于 hexcell/hex3d 的 Z 向周期, 默认取 cell
    axis  : 晶格轴向 (蜂窝孔道/挤出方向), 'x'/'y'/'z', 默认 z
    """
    if kind not in LATTICE_TYPES:
        raise ValueError(f"未知晶格类型 {kind!r}, 可选: {', '.join(LATTICE_TYPES)}")
    if layer is None or layer <= 0:
        layer = cell
    # 坐标置换, 使晶格内部 Z 轴对准指定的世界轴
    if axis == "y":
        x, y, z = z, x, y
    elif axis == "x":
        x, y, z = y, z, x
    elif axis != "z":
        raise ValueError(f"axis 必须是 x/y/z, 得到 {axis!r}")
    if kind == "grid":
        return _grid_phi(x, y, z, cell, wall)
    if kind == "hex":
        return _hex_phi(x, y, cell, wall)
    if kind == "hexcell":
        return np.float32(wall / 2.0) - _hexcell_dist(x, y, z, cell, layer)
    if kind == "hex3d":
        return np.float32(wall / 2.0) - _hex3d_dist(x, y, z, cell, layer)
    if kind == "bcc":
        return np.float32(wall / 2.0) - _bcc_dist(x, y, z, cell)
    if kind == "cubic":
        return np.float32(wall / 2.0) - _cubic_dist(x, y, z, cell)
    if kind == "kelvin":
        return np.float32(wall / 2.0) - _kelvin_dist(x, y, z, cell)
    return _tpms_phi(kind, x, y, z, cell, wall, pitch)

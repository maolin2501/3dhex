"""Read-only preview of the new source STEP tessellation."""
from pathlib import Path
import importlib.util
import json
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("cad_render_helpers", HERE.parent / "waist_local_closure_20261112/render_design.py")
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def axes(draw, direction, up, origin=(118, 738)):
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    right = np.cross(up, direction)
    right /= np.linalg.norm(right)
    up = np.cross(direction, right)
    colors = ((177, 65, 53), (61, 136, 84), (47, 112, 168))
    for axis, name, color in zip(np.eye(3), "XYZ", colors):
        delta = np.array((axis @ right, -axis @ up)) * 65
        end = np.asarray(origin) + delta
        if np.linalg.norm(delta) < 4:
            draw.ellipse((origin[0]-6, origin[1]-6, origin[0]+6, origin[1]+6), outline=color, width=3)
            draw.text((origin[0]+11, origin[1]+14), name, font=r.font(21, True), fill=color)
        else:
            draw.line((*origin, *end), fill=color, width=4)
            unit = delta / np.linalg.norm(delta)
            normal = np.array((-unit[1], unit[0]))
            draw.polygon([tuple(end), tuple(end-11*unit+5*normal), tuple(end-11*unit-5*normal)], fill=color)
            draw.text(tuple(end+15*unit), name, font=r.font(21, True), fill=color, anchor="mm")


def panel(mesh, title, detail, direction, up):
    canvas = Image.new("RGB", (1160, 860), r.WHITE)
    canvas.paste(r.render([(mesh, r.FIXED)], (1100, 710), direction=direction, up=up), (30, 100))
    draw = ImageDraw.Draw(canvas)
    draw.text((30, 18), title, font=r.font(36, True), fill=r.INK)
    draw.text((32, 68), detail, font=r.font(23), fill=r.MUTED)
    draw.rounded_rectangle((26, 650, 229, 835), radius=12, fill=r.WHITE, outline=r.RULE, width=2)
    axes(draw, direction, up)
    return canvas


def main():
    data = json.loads((HERE / "source_inspection.json").read_text(encoding="utf-8"))["new_source"]
    mesh = r.read_stl(HERE / "source.stl")
    x, y, z = data["mesh_extent_mm"]
    top = panel(mesh, "俯视 · 从 +Z 观察", f"X {x:.3f} × Y {y:.3f} mm", (0, 0, 1), (0, 1, 0))
    iso = panel(mesh, "立体 · 原始开口与安装结构", "同一源模型；未添加晶格或瓣片", (0.85, -1, 0.8), (0, 0, 1))
    side = panel(mesh, "侧视 · 从 +X 观察", f"Y {y:.3f} × Z {z:.3f} mm", (1, 0, 0), (0, 0, 1))
    front = panel(mesh, "正视 · 从 −Y 观察", f"X {x:.3f} × Z {z:.3f} mm", (0, -1, 0), (0, 0, 1))
    result = Image.new("RGB", (2400, 1990), r.PAPER)
    draw = ImageDraw.Draw(result)
    draw.text((40, 27), "新 STEP 源模型 · 几何核对", font=r.font(56, True), fill=r.INK)
    draw.text((44, 108), f"网格外形尺寸 {x:.3f} × {y:.3f} × {z:.3f} mm  |  原始坐标 Z = 1239.000…1297.277 mm", font=r.font(29), fill=r.MUTED)
    for image, position in ((top, (30, 175)), (iso, (1210, 175)), (side, (30, 1055)), (front, (1210, 1055))):
        result.paste(image, position)
    draw = ImageDraw.Draw(result)
    draw.text((43, 1940), "直接渲染新 STEP 导出的表面网格；仅作显示坐标平移，几何未变形。尺寸来自网格包围盒。", font=r.font(25), fill=r.MUTED)
    for name, image in (("source_overview", result), ("source_top", top), ("source_iso", iso), ("source_x", side), ("source_y", front)):
        image.save(HERE / f"{name}.png")
        print(f"Rendered {name}.png {image.size}")


if __name__ == "__main__":
    main()

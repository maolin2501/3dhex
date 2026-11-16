"""Render the actual closed CAD and a contact-region illustration.

No leaf is moved, scaled, or deformed by this renderer. The response illustration
changes only the opacity/color of panels flagged by native CAD intersections.
Mesh coordinates are shifted by C.z only for display.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy


HERE = Path(__file__).resolve().parent
CZ = 1314.8968633141
FONT_DIR = Path("C:/Windows/Fonts")
PAPER = (236, 242, 239)
WHITE = (255, 255, 255)
INK = (23, 52, 49)
MUTED = (86, 109, 105)
FIXED = (73, 130, 123)
FLEXURE = (220, 144, 67)
CONTACT = (173, 88, 38)
PANEL = (156, 188, 172)
PANEL_ALT = (188, 208, 194)
ROD = (112, 145, 165)
AXIS = (88, 113, 128)
RULE = (213, 225, 220)


def font(size, bold=False):
    return ImageFont.truetype(str(FONT_DIR / ("msyhbd.ttc" if bold else "msyh.ttc")), size)


def read_stl(path):
    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(path))
    reader.Update()
    original = reader.GetOutput()
    if original.GetNumberOfPoints() == 0:
        raise ValueError(f"Empty or missing STL: {path}")
    transform = vtk.vtkTransform()
    transform.Translate(0, 0, -CZ)
    shift = vtk.vtkTransformPolyDataFilter()
    shift.SetInputData(original)
    shift.SetTransform(transform)
    shift.Update()
    mesh = vtk.vtkPolyData()
    mesh.DeepCopy(shift.GetOutput())
    return mesh


def triangle_mesh(vertices, faces):
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(np.asarray(vertices, dtype=float), deep=True))
    faces = np.asarray(faces, dtype=np.int64)
    cells = vtk.vtkCellArray()
    cells.SetData(numpy_to_vtkIdTypeArray(np.arange(len(faces) + 1, dtype=np.int64) * 3, deep=True),
                  numpy_to_vtkIdTypeArray(faces.ravel(), deep=True))
    mesh = vtk.vtkPolyData()
    mesh.SetPoints(points)
    mesh.SetPolys(cells)
    return mesh


def rod_mesh(theta, start=-95.0, end=125.0, radius=37.5):
    angle = np.deg2rad(theta)
    direction = np.array((0, np.sin(angle), -np.cos(angle)))
    cross = np.array((0, np.cos(angle), np.sin(angle)))
    n = 128
    phases = np.linspace(0, 2 * np.pi, n, endpoint=False)
    circle = radius * (np.cos(phases)[:, None] * (1, 0, 0) + np.sin(phases)[:, None] * cross)
    vertices = np.vstack((circle + start * direction, circle + end * direction,
                          start * direction, end * direction))
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.extend(((i, n + i, n + j), (i, n + j, j), (2 * n, i, j),
                      (2 * n + 1, n + j, n + i)))
    return triangle_mesh(vertices, faces)


def actor_for(mesh, color, opacity=1.0):
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(mesh)
    normals.SetFeatureAngle(36)
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(normals.GetOutputPort())
    if mesh.GetCellData().GetScalars() is not None:
        mapper.SetScalarModeToUseCellData()
        mapper.SetColorModeToDirectScalars()
    else:
        mapper.ScalarVisibilityOff()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetColor(*(v / 255 for v in color))
    prop.SetOpacity(opacity)
    prop.SetAmbient(0.38)
    prop.SetDiffuse(0.64)
    prop.SetSpecular(0.18)
    prop.SetSpecularPower(30)
    prop.SetInterpolationToPhong()
    return actor


def load_components():
    metadata = json.loads((HERE / "panel_metadata.json").read_text(encoding="utf-8"))
    records = metadata.get("panels", metadata) if isinstance(metadata, dict) else metadata
    if isinstance(records, dict):
        records = [dict(value, id=key) for key, value in records.items()]
    frame = read_stl(HERE / "fixed_frame.stl")
    hinges = read_stl(HERE / "hinges.stl")
    panels = {}
    for record in records:
        panel_id = str(record.get("id", record.get("panel_id")))
        panels[panel_id] = (read_stl(HERE / "panels" / f"{panel_id}.stl"), record)
    if len(panels) != 30:
        raise ValueError(f"Expected 30 independent panels; found {len(panels)}")
    return frame, hinges, panels


def closed_parts(frame, hinges, panels):
    parts = [(frame, FIXED), (hinges, FLEXURE)]
    for index, (mesh, record) in enumerate(panels.values()):
        tier = record.get("tier", record.get("radial_tier", index))
        try:
            tier = int(tier)
        except (ValueError, TypeError):
            tier = index
        parts.append((mesh, PANEL if tier % 2 == 0 else PANEL_ALT))
    return parts


def load_activation():
    data = json.loads((HERE / "activation.json").read_text(encoding="utf-8"))
    entries = data.get("angles", data.get("poses", data))
    if isinstance(entries, dict):
        entries = [dict(value, angle_deg=key) for key, value in entries.items()]
    result = {}
    for entry in entries:
        angle = int(float(entry.get("angle_deg", entry.get("angle"))))
        if "active_panel_ids" in entry:
            active = entry["active_panel_ids"]
        elif "panels" in entry:
            states = entry["panels"]
            active = ([key for key, value in states.items() if value.get("active", False)]
                      if isinstance(states, dict) else
                      [value.get("id", value.get("panel_id")) for value in states
                       if value.get("active", False)])
        else:
            raise ValueError(f"Unrecognized activation entry for {angle}")
        result[angle] = set(map(str, active))
    if not all(angle in result for angle in (-50, 0, 50)):
        raise ValueError("Activation must include −50°, 0°, +50°")
    return result


def add_axis(renderer, theta):
    angle = np.deg2rad(theta)
    direction = np.array((0, np.sin(angle), -np.cos(angle)))
    # Short separate line segments give a stable dashed axis in every view.
    for offset in np.arange(-110, 146, 10):
        line = vtk.vtkLineSource()
        line.SetPoint1(*(offset * direction))
        line.SetPoint2(*((offset + 5) * direction))
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(line.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*(v / 255 for v in AXIS))
        actor.GetProperty().SetLineWidth(2)
        renderer.AddActor(actor)
    point = vtk.vtkSphereSource()
    point.SetRadius(2.2)
    point.SetThetaResolution(20)
    point.SetPhiResolution(12)
    point.Update()
    renderer.AddActor(actor_for(point.GetOutput(), INK))


def render(parts, size, direction=(0.25, -0.35, -1.0), up=(0, 1, 0),
           rod_angle=None, rod_opacity=0.27, motion=False, framing=None):
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1, 1, 1)
    renderer.SetUseDepthPeeling(True)
    renderer.SetMaximumNumberOfPeels(100)
    renderer.SetOcclusionRatio(0.05)
    for part in parts:
        mesh, color = part[:2]
        opacity = part[2] if len(part) == 3 else 1.0
        renderer.AddActor(actor_for(mesh, color, opacity))
    meshes = [part[0] for part in parts]
    if rod_angle is not None:
        rod = rod_mesh(rod_angle)
        renderer.AddActor(actor_for(rod, ROD, rod_opacity))
        add_axis(renderer, rod_angle)
        if motion:
            meshes.append(rod)
    for position, strength in (((-0.6, 0.8, 1), 0.82), ((0.9, -0.3, 0.6), 0.35)):
        light = vtk.vtkLight()
        light.SetLightTypeToCameraLight()
        light.SetPosition(*position)
        light.SetFocalPoint(0, 0, 0)
        light.SetIntensity(strength)
        renderer.AddLight(light)

    camera = renderer.GetActiveCamera()
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    right = np.cross(up, direction)
    right /= np.linalg.norm(right)
    up = np.cross(direction, right)
    points = np.vstack([vtk_to_numpy(mesh.GetPoints().GetData()) for mesh in meshes])
    projection = points @ np.column_stack((right, up))
    lo, hi = projection.min(axis=0), projection.max(axis=0)
    middle = (lo + hi) / 2
    target = right * middle[0] + up * middle[1]
    scale = 0.55 * max(hi[1] - lo[1], (hi[0] - lo[0]) * size[1] / size[0])
    if motion:
        target = np.array((0, 0, -7))
        scale = 133
    if framing is not None:
        target, scale = framing
        target = np.asarray(target, dtype=float)
    camera.SetFocalPoint(*target)
    camera.SetPosition(*(target + 600 * direction))
    camera.SetViewUp(*up)
    camera.ParallelProjectionOn()
    camera.SetParallelScale(scale)
    renderer.ResetCameraClippingRange()

    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetAlphaBitPlanes(1)
    window.SetMultiSamples(0)
    window.SetSize(*size)
    window.AddRenderer(renderer)
    try:
        window.Render()
        capture = vtk.vtkWindowToImageFilter()
        capture.SetInput(window)
        capture.SetInputBufferTypeToRGB()
        capture.ReadFrontBufferOff()
        capture.Update()
        array = vtk_to_numpy(capture.GetOutput().GetPointData().GetScalars())
        return Image.fromarray(array.reshape(size[1], size[0], 3)[::-1].copy())
    finally:
        window.Finalize()


def chip(draw, xy, label, color=FIXED):
    x, y = xy
    draw.rounded_rectangle((x, y + 6, x + 17, y + 23), radius=4, fill=color)
    draw.text((x + 28, y), label, font=font(23), fill=MUTED)


def closed_panel(parts):
    panel = Image.new("RGB", (1600, 1430), WHITE)
    draw = ImageDraw.Draw(panel)
    draw.text((45, 26), "闭合参考形态 · 搭接覆盖", font=font(49, True), fill=INK)
    draw.text((48, 99), "14 组对合瓣 + 2 端瓣，共 30 片独立活动瓣", font=font(29), fill=MUTED)
    draw.text((48, 149), "外侧视图｜原始 CAD 几何，未施加杆体位移", font=font(25), fill=MUTED)
    panel.paste(render(parts, (1520, 1060), direction=(0.12, -0.16, -1.0)), (40, 197))
    draw = ImageDraw.Draw(panel)
    draw.line((48, 1278, 1552, 1278), fill=RULE, width=2)
    draw.text((48, 1300), "瓣厚 0.8 mm · 层间隙 0.5 mm · R85 基准行间搭接 1.4 mm · 中缝搭接 3 mm", font=font(27), fill=INK)
    draw.text((48, 1353), "“闭合”指投影搭接覆盖；实际回弹能力与开启行程尚未验证。", font=font(25), fill=MUTED)
    return panel


def underside_panel(parts):
    panel = Image.new("RGB", (1800, 1430), WHITE)
    draw = ImageDraw.Draw(panel)
    draw.text((45, 26), "内侧结构 · 分段曲梁根部", font=font(49, True), fill=INK)
    draw.text((48, 99), "56 条主瓣 S 曲梁 + 4 条端瓣曲梁｜30 片活动瓣独立连接", font=font(29), fill=MUTED)
    draw.text((48, 149), "内侧整体与侧根局部，均从实际闭合 CAD 网格渲染", font=font(25), fill=MUTED)
    panel.paste(render(parts, (1210, 1040), direction=(0.12, -0.25, 1.0)), (20, 202))
    transparent_parts = [(part[0], part[1], 1.0 if part[1] == FLEXURE else 0.10)
                         for part in parts]
    panel.paste(render(transparent_parts, (530, 690), direction=(0.15, -0.1, 1.0),
                       framing=((40, 0, -68), 25)), (1240, 300))
    draw = ImageDraw.Draw(panel)
    draw.rounded_rectangle((1240, 300, 1770, 990), radius=16, outline=RULE, width=3)
    draw.text((1257, 233), "侧根局部 · 透明观察", font=font(31, True), fill=INK)
    draw.text((1257, 1030), "橙色：曲梁", font=font(26), fill=FLEXURE)
    draw.text((1257, 1078), "青绿：固定框架", font=font(26), fill=FIXED)
    draw.text((1257, 1126), "覆盖件透明显示", font=font(26), fill=MUTED)
    draw.line((48, 1278, 1752, 1278), fill=RULE, width=2)
    draw.text((48, 1300), "设计目标：受杆影响的分段局部响应，其余区域保留覆盖；未计算材料变形。", font=font(28), fill=INK)
    draw.text((48, 1353), "层间为真实几何间隙；打印公差、回弹力及疲劳寿命需后续验证。", font=font(25), fill=MUTED)
    return panel


def local_response_panel(frame, hinges, panels, activation):
    panel = Image.new("RGB", (2520, 1430), WHITE)
    draw = ImageDraw.Draw(panel)
    draw.text((45, 26), "局部受压区域随杆位置改变", font=font(50, True), fill=INK)
    draw.text((48, 102), "受压区域示意，未计算实际变形", font=font(34, True), fill=CONTACT)
    draw.text((48, 162), "Ø75 杆轴始终通过球心 C，绕 X 在 YZ 面内摆动；0° 沿 −Z", font=font(27), fill=MUTED)
    for index, angle in enumerate((-50, 0, 50)):
        left = 30 + index * 830
        active = activation[angle]
        unknown = active.difference(panels)
        if unknown:
            raise ValueError(f"Unknown activated panels: {sorted(unknown)}")
        parts = [(frame, FIXED), (hinges, FIXED)]
        for panel_id, (mesh, record) in panels.items():
            parts.append((mesh, FLEXURE, 0.13) if panel_id in active else (mesh, PANEL, 1.0))
        panel.paste(render(parts, (800, 630), direction=(1.0, -0.16, -0.32), up=(0, 0, 1),
                           rod_angle=angle, rod_opacity=0.42, motion=True), (left, 220))
        # The bottom view isolates the intersection mask from the oblique rod view.
        mask_parts = [(frame, FIXED), (hinges, FIXED)] + [
            (mesh, FLEXURE if panel_id in active else PANEL)
            for panel_id, (mesh, record) in panels.items()]
        panel.paste(render(mask_parts, (440, 410), direction=(0, 0, -1.0)), (left + 180, 850))
        draw = ImageDraw.Draw(panel)
        draw.text((left + 414, 506), "C", font=font(25, True), fill=INK,
                  stroke_width=2, stroke_fill=WHITE)
        label = "0°" if angle == 0 else ("−50°" if angle < 0 else "+50°")
        draw.text((left + 400, 240), label, font=font(36, True), fill=INK, anchor="mm")
        draw.text((left + 400, 1283), f"初始相交瓣片 {len(active)} / 30", font=font(29, True), fill=INK, anchor="mm")
    draw.line((45, 1320, 2475, 1320), fill=RULE, width=2)
    draw.text((48, 1340), "橙色为初始几何与杆相交的瓣片；透明仅为显示方式。其余瓣片保留闭合参考位置，未计算真实开启与回弹。", font=font(27), fill=MUTED)
    return panel


def compose_overview(closed, underside, response):
    canvas = Image.new("RGB", (2600, 2740), PAPER)
    draw = ImageDraw.Draw(canvas)
    draw.text((46, 30), "Ø75 球心摆杆 · 局部闭合瓣片方案", font=font(65, True), fill=INK)
    draw.text((50, 126), "以分段搭接瓣片覆盖开口，曲梁分别连接；杆接触区域随摆动位置改变。", font=font(32), fill=MUTED)
    canvas.paste(closed.resize((1200, 1073), Image.Resampling.LANCZOS), (40, 204))
    canvas.paste(underside.resize((1320, 1049), Image.Resampling.LANCZOS), (1240, 204))
    canvas.paste(response, (40, 1290))
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", action="store_true", help="Render closed CAD diagnostic views only")
    parser.add_argument("--closed-only", action="store_true", help="Render closed and underside views before activation exists")
    args = parser.parse_args()
    frame, hinges, panels = load_components()
    parts = closed_parts(frame, hinges, panels)
    if args.draft:
        closed = render(parts, (800, 650))
        underside = render(parts, (800, 650), direction=(0.12, -0.25, 1.0))
        canvas = Image.new("RGB", (1600, 650), WHITE)
        canvas.paste(closed, (0, 0))
        canvas.paste(underside, (800, 0))
        canvas.save(HERE / "render_draft.png")
        print("Rendered render_draft.png")
        return
    closed = closed_panel(parts)
    underside = underside_panel(parts)
    pictures = [("closed_view", closed), ("underside_view", underside)]
    if not args.closed_only:
        response = local_response_panel(frame, hinges, panels, load_activation())
        pictures += [("local_response", response),
                     ("overview", compose_overview(closed, underside, response))]
    for name, picture in pictures:
        picture.save(HERE / f"{name}.png")
        print(f"Rendered {name}.png: {picture.width} x {picture.height}")


if __name__ == "__main__":
    main()

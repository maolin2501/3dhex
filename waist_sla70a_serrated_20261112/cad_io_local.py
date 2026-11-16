"""OpenCASCADE face-preserving import for the visual editor (millimetres)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import trimesh


@dataclass
class CadBody:
    name: str
    mesh: trimesh.Trimesh
    face_ids: np.ndarray  # one original CAD face ID per triangle, stable on this import
    face_info: list[dict]

    def selected_mesh(self, ids: list[int]) -> trimesh.Trimesh:
        if not ids or not set(ids) <= {f['id'] for f in self.face_info}:
            raise ValueError('请选择当前实体中有效的 CAD 面')
        return self.mesh.submesh([np.flatnonzero(np.isin(self.face_ids, ids))], append=True)


def tessellate_shape(shape, name='实体', tolerance=0.12) -> list[CadBody]:
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_SOLID, TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    BRepMesh_IncrementalMesh(shape, tolerance, False, 0.2, True)
    explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    shapes = []
    while explorer.More():
        shapes.append(explorer.Current())
        explorer.Next()
    if not shapes:
        raise ValueError('STEP 中没有封闭实体；请先在 CAD 中将曲面缝合成实体')
    bodies = []
    for index, solid in enumerate(shapes):
        vertices, triangles, ids, info = [], [], [], []
        exp = TopExp_Explorer(solid, TopAbs_FACE)
        offset, fid = 0, 0
        while exp.More():
            face = TopoDS.Face_s(exp.Current())
            loc = TopLoc_Location()
            tri = BRep_Tool.Triangulation_s(face, loc)
            if tri is None or tri.NbTriangles() == 0:
                raise ValueError(f'实体 {index + 1} 面 {fid + 1} 三角化失败')
            transform = loc.Transformation()
            pts = np.array([tri.Node(i).Transformed(transform).Coord()
                            for i in range(1, tri.NbNodes() + 1)])
            faces = np.array([tri.Triangle(i).Get()
                              for i in range(1, tri.NbTriangles() + 1)], dtype=np.int64) - 1
            if face.Orientation() == TopAbs_REVERSED:
                faces = faces[:, [0, 2, 1]]
            prop = GProp_GProps()
            BRepGProp.SurfaceProperties_s(face, prop)
            adaptor = BRepAdaptor_Surface(face)
            surface = str(adaptor.GetType()).split('_')[-1]
            item = dict(id=fid, label=f'面 {fid + 1:03d}', type=surface,
                        area=round(prop.Mass(), 3), triangles=len(faces))
            if surface in ('Plane', 'Cylinder', 'Torus'):
                analytic = getattr(adaptor, surface)()
                frame = analytic.Position()
                item['chart'] = dict(kind=surface.lower(), origin=list(frame.Location().Coord()),
                                     u=list(frame.XDirection().Coord()), v=list(frame.YDirection().Coord()))
                if surface == 'Cylinder':
                    item['chart'].update(axis=list(frame.Direction().Coord()), radius=analytic.Radius())
                elif surface == 'Torus':
                    vmid = (adaptor.FirstVParameter()+adaptor.LastVParameter())/2
                    signed_radius = analytic.MajorRadius()+analytic.MinorRadius()*np.cos(vmid)
                    item['chart'].update(axis=list(frame.Direction().Coord()),
                        major_radius=analytic.MajorRadius(), minor_radius=analytic.MinorRadius(),
                        v_mid=vmid, radial_sign=1 if signed_radius >= 0 else -1,
                        radius=abs(signed_radius),
                        v_periodic=adaptor.LastVParameter()-adaptor.FirstVParameter() >= 2*np.pi-1e-6)
            info.append(item)
            vertices.append(pts)
            triangles.append(faces + offset)
            ids.extend([fid] * len(faces))
            offset += len(pts)
            fid += 1
            exp.Next()
        mesh = trimesh.Trimesh(np.vstack(vertices), np.vstack(triangles), process=False)
        # Merge shared vertices without changing triangle order / CAD face mapping.
        mesh.merge_vertices(digits_vertex=7)
        face_ids = np.asarray(ids, dtype=np.int32)
        valid = mesh.nondegenerate_faces()
        mesh.update_faces(valid)
        face_ids = face_ids[valid]
        mesh.remove_unreferenced_vertices()
        for item in info:
            item['triangles'] = int(np.count_nonzero(face_ids == item['id']))
        bodies.append(CadBody(f'{name} · 实体 {index + 1}', mesh, face_ids, info))
    return bodies


def load_cad(path: str) -> list[CadBody]:
    from OCP.STEPControl import STEPControl_Reader
    from OCP.IFSelect import IFSelect_RetDone
    path = Path(path)
    if path.suffix.lower() not in {'.step', '.stp'}:
        raise ValueError('可视化面选择目前支持 STEP / STP 文件')
    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise ValueError('STEP 文件读取失败')
    reader.SetSystemLengthUnit(1.0)  # OCCT target unit in mm; honour source STEP unit
    if reader.TransferRoots() == 0:
        raise ValueError('STEP 没有可转换的几何体')
    return tessellate_shape(reader.OneShape(), path.stem)


def demo_bodies() -> list[CadBody]:
    """A curved shell with separate inner, outer and end CAD faces."""
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.gp import gp_Ax2, gp_Pnt, gp_Dir
    import math
    ax = gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
    outer = BRepPrimAPI_MakeCylinder(ax, 24, 36, math.pi * 1.3).Shape()
    inner = BRepPrimAPI_MakeCylinder(ax, 19, 36, math.pi * 1.3).Shape()
    return tessellate_shape(BRepAlgoAPI_Cut(outer, inner).Shape(), '示例曲壳')


def save_bodies(bodies: list[CadBody], directory: Path):
    import json
    metadata = []
    for i, body in enumerate(bodies):
        np.savez_compressed(directory / f'body_{i}.npz', vertices=body.mesh.vertices,
                            faces=body.mesh.faces, face_ids=body.face_ids)
        metadata.append(dict(name=body.name, faces=body.face_info))
    (directory / 'bodies.json').write_text(json.dumps(metadata, ensure_ascii=False), encoding='utf-8')


def read_body(directory: Path, index: int) -> CadBody:
    import json
    data = json.loads((directory / 'bodies.json').read_text(encoding='utf-8'))[index]
    with np.load(directory / f'body_{index}.npz') as arrays:
        return CadBody(data['name'], trimesh.Trimesh(arrays['vertices'], arrays['faces'], process=False),
                       arrays['face_ids'], data['faces'])


def upgrade_face_charts(body: CadBody, directory: Path, index: int):
    """Old caches keep their IDs/mesh. Only reuse new charts after exact checks."""
    import json
    if not any(f['type'] == 'Torus' and not f.get('chart') for f in body.face_info):
        return
    request = json.loads((directory/'request.json').read_text(encoding='utf-8'))
    fresh = (demo_bodies() if request.get('demo') else load_cad(request['path']))[index]
    if (len(fresh.face_info) != len(body.face_info)
        or not np.array_equal(fresh.face_ids, body.face_ids)
        or fresh.mesh.vertices.shape != body.mesh.vertices.shape
        or not np.allclose(fresh.mesh.vertices, body.mesh.vertices, atol=1e-6, rtol=0)):
        raise ValueError('旧缓存与 STEP 面映射不一致，请重新导入后选择曲面；不会猜测旧面编号')
    for old, new in zip(body.face_info, fresh.face_info):
        if new.get('chart'):
            old['chart'] = new['chart']

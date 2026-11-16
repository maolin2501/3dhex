"""Paired actual CAD-seam crop, then a new full-shoulder result (v8)."""
import json
import secrets
import sys
from pathlib import Path
import numpy as np
from latticefill.studio_jobs import atomic_json, compute, body_metadata
from latticefill.cad_io import read_body, tessellate_shape, save_bodies
from latticefill.continuity import MeshProbe
from .inspect_reference import read_triangles, render
from .render_continuity_comparison import crop

BASELINE = '4d042acdecebcf921d869666'
FOLDER = Path('reference_meshes/shoulder_v1/stitch')


def crop_model(root, previous, centre=None, half=12.):
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
    from OCP.gp import gp_Pnt
    reader = STEPControl_Reader(); reader.ReadFile(str(root/previous['model_id']/'input.step'))
    reader.SetSystemLengthUnit(1.); reader.TransferRoots()
    explorer = TopExp_Explorer(reader.OneShape(), TopAbs_SOLID)
    centre = np.array([-254., -15., 1618.375] if centre is None else centre)
    cutter = BRepPrimAPI_MakeBox(gp_Pnt(*(centre-half)), 2*half, 2*half, 2*half).Shape()
    bodies = tessellate_shape(BRepAlgoAPI_Common(explorer.Current(), cutter).Shape(),
                             '右肩分界线 · 24 mm 对照', tolerance=.035)
    original = read_body(root/previous['model_id'], previous['source_body'])
    probe = MeshProbe(original.mesh, original.face_ids)
    active = set(f['id'] for f in original.face_info)-set(previous['config']['preserve_faces'])
    design = []
    for face in bodies[0].face_info:
        mesh = bodies[0].selected_mesh([face['id']])
        votes = []
        for point, normal in zip(mesh.triangles_center[::max(1,len(mesh.faces)//30)],
                                 mesh.face_normals[::max(1,len(mesh.faces)//30)]):
            nearest, n, fid = probe.closest(point)
            votes.append(np.linalg.norm(nearest-point) < .1 and np.dot(n,normal) > .85 and fid in active)
        if np.mean(votes) > .8: design.append(face['id'])
    model = root/secrets.token_hex(12); model.mkdir(); save_bodies(bodies, model)
    atomic_json(model/'status.json', dict(state='done', progress=100, message='局部分界线对照模型',
        result=dict(filename='真实右肩分界线 · 24 mm 局部', bodies=[body_metadata(b,i) for i,b in enumerate(bodies)])))
    config = previous['config'] | dict(design_faces=design,
        preserve_faces=[f['id'] for f in bodies[0].face_info if f['id'] not in design])
    print('CROP_MODEL', model.name, 'design', design, flush=True)
    return model, config


def main():
    root = Path('.lattice-studio').resolve(); FOLDER.mkdir(exist_ok=True)
    previous = json.loads((root/BASELINE/'design.json').read_text(encoding='utf-8'))
    mode = sys.argv[1] if len(sys.argv)>1 else 'crop'
    if mode == 'render':
        for name, path in dict(before=root/BASELINE/'result.stl', after=root/sys.argv[2]/'result.stl',
            reference=Path('reference_meshes/shoulder_v1/722机器人-右肩晶格v1.stl')).items():
            tris = read_triangles(path)
            for part, centre in [('seam',[-254,-15,1618.375]), ('struts',[-250,-22,1640])]:
                centre = np.array(centre)
                render(crop(tris,centre,12), FOLDER/f'{part}_{name}.png', [-1,-.4,.2], focal=centre, scale=14)
        return
    model, cfg = (crop_model(root, previous) if mode == 'crop' else
                  (root/previous['model_id'], previous['config']))
    for style in (('rail','stitch') if mode == 'crop' else ('stitch',)):
        job = root/secrets.token_hex(12); job.mkdir()
        config = cfg | dict(surface_continuity='auto', seam_style=style, memory_mode='chunked')
        request = dict(action='generate', model_dir=str(model), body=0, config=config)
        atomic_json(job/'request.json', request)
        atomic_json(FOLDER/f'{mode}_{style}_job.json', dict(job=job.name, baseline=BASELINE))
        print('STITCH_JOB', mode, style, job.name, flush=True)
        try: compute(job, request)
        except Exception as exc:
            atomic_json(job/'status.json', dict(state='failed', progress=0, message=str(exc)))
            raise
        result = json.loads((job/'design.json').read_text(encoding='utf-8'))
        atomic_json(FOLDER/f'{mode}_{style}_result.json', result)
        if mode == 'crop': render(read_triangles(job/'result.stl'), FOLDER/f'crop_{style}.png', [-1,-.4,.2])
        print('STITCH_DONE', job.name, result['stats'], result['performance'], flush=True)


if __name__ == '__main__': main()

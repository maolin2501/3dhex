"""New jobs only: real shoulder comparison without changing saved user settings."""
import json
import secrets
import sys
from pathlib import Path
from latticefill.studio_jobs import atomic_json, compute
from .inspect_reference import read_triangles, render


def main():
    root = Path('.lattice-studio').resolve()
    mode = sys.argv[1] if len(sys.argv)>1 else 'crop'
    baseline = 'a7affebbef2cba1b41d6848e' if mode == 'full' else 'affe8203d6ee4a88bb0bf301'
    result = json.loads((root/baseline/'design.json').read_text(encoding='utf-8'))
    config = result['config'] | dict(surface_continuity='auto', memory_mode='chunked')
    job = root/secrets.token_hex(12); job.mkdir()
    request = dict(action='generate', model_dir=str(root/result['model_id']), body=result['source_body'], config=config)
    atomic_json(job/'request.json', request)
    folder = Path('reference_meshes/shoulder_v1/continuity'); folder.mkdir(exist_ok=True)
    atomic_json(folder/f'{mode}_job.json', dict(job=job.name, baseline=baseline))
    print('CONTINUITY_JOB', mode, job.name, flush=True)
    try:
        compute(job, request)
    except Exception as exc:
        atomic_json(job/'status.json', dict(state='failed', progress=0, message=str(exc)))
        raise
    result = json.loads((job/'design.json').read_text(encoding='utf-8'))
    atomic_json(folder/f'{mode}_result.json', result)
    if mode == 'crop':
        render(read_triangles(job/'result.stl'), folder/'crop_after.png', [-1,-.4,.2])
        render(read_triangles(root/baseline/'result.stl'), folder/'crop_before.png', [-1,-.4,.2])
    print('CONTINUITY_DONE', job.name, result['stats'], result['performance'], flush=True)


if __name__ == '__main__': main()

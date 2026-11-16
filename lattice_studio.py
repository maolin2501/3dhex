"""Local-only browser UI. Start with .lattice-env/Scripts/python lattice_studio.py."""
from __future__ import annotations
import argparse
import atexit
import json
import mimetypes
import secrets
import subprocess
import sys
import threading
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from latticefill.studio_jobs import (atomic_json, estimate, validate_design,
                                    PREVIEW_VOXEL_LIMIT, FINAL_VOXEL_LIMIT)

ROOT = Path(__file__).resolve().parent
WEB = ROOT / 'lattice_studio'
DATA = ROOT / '.lattice-studio'
JOB_ROOT = None  # Optional user-selected output disk; DATA retains old jobs.
JOBS = {}
LOCK = threading.RLock()
TOKEN = secrets.token_urlsafe(32)


class LocalServer(ThreadingHTTPServer):
    # Windows SO_REUSEADDR otherwise permits two servers on the same port.
    allow_reuse_address = False

    def server_bind(self):
        if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def job_path(job_id):
    if len(job_id) != 24 or any(c not in '0123456789abcdef' for c in job_id):
        raise ValueError('无效任务编号')
    for root in (DATA, JOB_ROOT):
        if root is not None and (root / job_id).is_dir():
            return root / job_id
    raise ValueError('任务不存在')


def get_status(job_id):
    with LOCK:
        directory = job_path(job_id)
        status = json.loads((directory / 'status.json').read_text(encoding='utf-8'))
        job = JOBS.get(job_id)
        if job and job['process'].poll() is not None:
            # The worker may have committed done between our first read and
            # poll. Once exited, reread its final atomic publication.
            status=json.loads((directory/'status.json').read_text(encoding='utf-8'))
            if status['state']=='running':
                code=job['process'].returncode
                try:checkpoint=checkpoint_source(directory)
                except ValueError:checkpoint=None
                status = dict(state='failed', progress=0, exit_code=code, checkpoint=checkpoint is not None,
                    checkpoint_job=checkpoint.name if checkpoint is not None else None,
                    message=f'计算进程异常退出（代码 {code}）。'+('裁切前检查点已保留，可恢复后处理。' if checkpoint else '请查看任务日志。'))
                atomic_json(directory/'status.json',status)
        return status


def checkpoint_source(directory):
    seen=set();current=Path(directory).resolve()
    for _ in range(64):
        if current in seen:raise ValueError('恢复检查点引用存在循环')
        seen.add(current)
        if (current/'pretrim/ready.json').is_file():return current
        request_file=current/'request.json'
        if not request_file.is_file():return None
        previous=json.loads(request_file.read_text(encoding='utf-8')).get('resume_from')
        if not previous:return None
        target=Path(previous).resolve()
        if job_path(target.name).resolve()!=target:raise ValueError('恢复检查点不在任务目录中')
        current=target
    raise ValueError('恢复检查点引用过深')


def start_job(request, upload=None):
    with LOCK:
        for old_id, old in JOBS.items():
            if old['process'].poll() is None:
                if get_status(old_id)['state'] in ('done', 'failed', 'cancelled'):
                    # Worker commits output before Python/OCCT module teardown completes.
                    try:
                        old['process'].wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        raise ValueError('上一任务正在释放内存，请稍后重试')
                else:
                    raise ValueError('已有任务正在运行，请等待或点击停止')
        job_id = secrets.token_hex(12)
        directory = (JOB_ROOT or DATA) / job_id
        directory.mkdir(parents=True)
        if upload is not None:
            source = directory / 'input.step'
            source.write_bytes(upload)
            request['path'] = str(source)
        atomic_json(directory / 'request.json', request)
        atomic_json(directory / 'status.json', dict(state='running', progress=0, message='启动计算'))
        log = (directory / 'run.log').open('wb')
        try:
            proc = subprocess.Popen([sys.executable, '-X', 'faulthandler', '-u', '-m', 'latticefill.studio_jobs', str(directory)],
                                    cwd=ROOT, stdout=log, stderr=log,
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        finally:
            log.close()
        JOBS[job_id] = dict(process=proc)
        return dict(job=job_id)


def cancel_job(job_id):
    with LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise ValueError('当前会话中没有该任务')
        proc = job['process']
        if proc.poll() is None:
            if sys.platform=='win32':
                # A Windows venv launcher starts the real Python as a child.
                # Killing only the launcher leaves a worker and locked memmaps.
                stopped=subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],
                    capture_output=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                if stopped.returncode and proc.poll() is None:
                    raise OSError('无法停止该计算进程树，请稍后重试')
            else:proc.terminate()
            proc.wait(timeout=10)
            from latticefill.lowmem import cleanup_job_scratch
            message='已停止计算'
            try:cleanup_job_scratch(job_path(job_id))
            except OSError:message+='；临时文件仍被占用，关闭软件后可清理该任务的 scratch-lowmem 目录'
            atomic_json(job_path(job_id) / 'status.json', dict(state='cancelled', progress=0, message=message))
    return get_status(job_id)


def stop_all():
    for job_id in list(JOBS):
        cancel_job(job_id)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def send_bytes(self, data, content_type, filename=None):
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        if filename:
            self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(data)

    def json(self, data):
        self.send_bytes(json.dumps(data, ensure_ascii=False, allow_nan=False).encode(), 'application/json; charset=utf-8')

    def valid_host(self):
        return self.headers.get('Host') in (f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}')

    def do_GET(self):
        if not self.valid_host():
            self.send_error(403)
            return
        try:
            url = urlparse(self.path)
            q = parse_qs(url.query)
            if url.path == '/api/bootstrap':
                return self.json(dict(token=TOKEN, version='1.10', app='lattice-studio',
                                      sample_limits=dict(preview=PREVIEW_VOXEL_LIMIT, final=FINAL_VOXEL_LIMIT)))
            if url.path == '/api/status':
                return self.json(get_status(q['job'][0]))
            if url.path == '/api/body':
                from latticefill.cad_io import read_body
                body = read_body(job_path(q['model'][0]), int(q['body'][0]))
                # Per-face groups preserve CAD IDs. Indexed rendering avoids JSON expansion.
                return self.json(dict(vertices=body.mesh.vertices.ravel().tolist(),
                                      indices=body.mesh.faces.ravel().tolist(), face_ids=body.face_ids.tolist()))
            if url.path.startswith('/artifacts/'):
                parts = url.path.split('/')
                if len(parts) != 4 or parts[3] not in ('result.stl', 'display.glb', 'design.json', 'run.log'):
                    raise ValueError('文件不可访问')
                directory = job_path(parts[2])
                if get_status(parts[2])['state'] != 'done' and parts[3] != 'run.log':
                    raise ValueError('任务未完成，不能下载不完整结果')
                return self.file(directory / parts[3], download=parts[3] != 'display.glb')
            if url.path.startswith('/vendor/'):
                base = WEB / 'node_modules' / 'three'
                target = (base / url.path.removeprefix('/vendor/')).resolve()
            else:
                base = WEB
                target = (WEB / ('index.html' if url.path == '/' else url.path.lstrip('/'))).resolve()
            if not target.is_relative_to(base.resolve()) or not target.is_file():
                self.send_error(404)
                return
            self.file(target)
        except (ConnectionResetError, BrokenPipeError):
            pass
        except (ValueError, KeyError, IndexError, OSError) as exc:
            self.send_error(400, str(exc).encode('ascii', 'replace').decode())

    def file(self, path, download=False):
        kind = {'.js': 'text/javascript', '.glb': 'model/gltf-binary', '.stl': 'application/octet-stream'}.get(path.suffix)
        kind = kind or mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
        # Stream STL instead of allocating another hundreds-of-MB byte buffer.
        self.send_response(200)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(path.stat().st_size))
        self.send_header('X-Content-Type-Options', 'nosniff')
        if download:
            self.send_header('Content-Disposition', f'attachment; filename="lattice_{path.name}"')
        self.end_headers()
        with path.open('rb') as stream:
            while chunk := stream.read(1024 * 1024):
                self.wfile.write(chunk)

    def do_POST(self):
        if not self.valid_host() or self.headers.get('X-Lattice-Token') != TOKEN:
            self.send_error(403)
            return
        origin = self.headers.get('Origin')
        if origin and origin not in (f'http://127.0.0.1:{self.server.server_port}', f'http://localhost:{self.server.server_port}'):
            self.send_error(403)
            return
        try:
            size = int(self.headers.get('Content-Length', 0))
            if not 0 < size <= 200 * 1024**2:
                raise ValueError('文件/请求大小需在 0–200 MB 之间')
            raw = self.rfile.read(size)
            url = urlparse(self.path)
            if url.path == '/api/upload':
                return self.json(start_job(dict(action='load'), upload=raw))
            data = json.loads(raw)
            if url.path == '/api/load':
                if data.get('demo'):
                    return self.json(start_job(dict(action='load', demo=True)))
                path = Path(data['path']).resolve()
                if not path.is_file() or path.suffix.lower() not in ('.step', '.stp'):
                    raise ValueError('请选择已有的 STEP / STP 文件')
                return self.json(start_job(dict(action='load', path=str(path))))
            if url.path in ('/api/generate', '/api/estimate'):
                cfg = validate_design(data['config'])
                model_dir = job_path(data['model'])
                state = get_status(data['model'])
                if state['state'] != 'done' or 'bodies' not in state['result']:
                    raise ValueError('模型还未载入完成')
                body = state['result']['bodies'][int(data['body'])]
                valid_faces = {f['id'] for f in body['faces']}
                if not (set(cfg['design_faces']) | set(cfg['preserve_faces'])) <= valid_faces:
                    raise ValueError('存在不属于当前实体的面')
                if url.path == '/api/estimate':
                    return self.json(estimate(body['bounds'], cfg))
                return self.json(start_job(dict(action='generate', config=cfg,
                                               model_dir=str(model_dir), body=int(data['body']))))
            if url.path == '/api/resume':
                source_dir=job_path(data['job'])
                checkpoint=checkpoint_source(source_dir)
                if checkpoint is None:
                    raise ValueError('该任务没有完整裁切检查点')
                original=json.loads((source_dir/'request.json').read_text(encoding='utf-8'))
                original['resume_from']=str(checkpoint)
                return self.json(start_job(original))
            if url.path == '/api/cancel':
                return self.json(cancel_job(data['job']))
            if url.path == '/api/redisplay':
                source_dir = job_path(data['job'])
                status = get_status(data['job'])
                if status['state'] != 'done' or not (source_dir/'result.stl').is_file():
                    raise ValueError('仅能修复已完成结果的显示副本')
                return self.json(start_job(dict(action='redisplay',source_dir=str(source_dir))))
            if url.path == '/api/shutdown':
                stop_all()
                self.json(dict(stopped=True))
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
            self.send_error(404)
        except (ValueError, KeyError, IndexError, TypeError, OSError) as exc:
            self.json(dict(error=str(exc)))


def main():
    global JOB_ROOT
    parser = argparse.ArgumentParser(description='Lattice Studio 本地三维晶格设计')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--job-root', type=Path, help='新计算与结果的保存目录，原有实例仍从项目目录读取')
    args = parser.parse_args()
    if not (WEB / 'node_modules/three/build/three.module.js').is_file():
        raise SystemExit('缺少显示组件，请先运行 npm install --prefix lattice_studio')
    DATA.mkdir(exist_ok=True)
    storage_file=DATA/'storage.json'
    if args.job_root is not None:
        JOB_ROOT=args.job_root.resolve()
    elif storage_file.is_file():
        JOB_ROOT=Path(json.loads(storage_file.read_text(encoding='utf-8'))['job_root']).resolve()
    try:
        server = LocalServer(('127.0.0.1', args.port), Handler)
    except OSError:
        if args.job_root is not None:
            raise SystemExit('保存位置尚未生效：请先关闭正在运行的软件服务，再使用 --job-root 重启。')
        from urllib.request import urlopen
        # A second double-click reopens this app, not an unrelated local service.
        try:
            with urlopen(f'http://127.0.0.1:{args.port}/api/bootstrap', timeout=2) as response:
                existing = json.load(response)
            if existing.get('app') != 'lattice-studio':
                raise ValueError('port in use')
        except Exception:
            raise SystemExit(f'端口 {args.port} 已占用，请使用 --port 指定其他端口。')
        if not args.no_browser:
            import webbrowser
            webbrowser.open(f'http://127.0.0.1:{args.port}')
        return
    if JOB_ROOT is not None:
        JOB_ROOT.mkdir(parents=True,exist_ok=True)
        if args.job_root is not None:
            atomic_json(storage_file,dict(job_root=str(JOB_ROOT)))
    atexit.register(stop_all)
    url = f'http://127.0.0.1:{server.server_port}'
    print(f'Lattice Studio: {url}', flush=True)
    if not args.no_browser:
        import webbrowser
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        stop_all()


if __name__ == '__main__':
    main()

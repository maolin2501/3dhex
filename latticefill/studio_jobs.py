"""Isolated compute jobs; safe to terminate without freezing the viewer."""
from __future__ import annotations
import json
import math
from pathlib import Path
import sys
import traceback
import time
import numpy as np
import trimesh
from .cad_io import load_cad, demo_bodies, save_bodies, read_body, upgrade_face_charts
from .core import FillParams, fill_solid
from .lattices import LATTICE_TYPES
from .surface import PerforatedSurface
from .mesh_quality import clean_fragments, export_display

PREVIEW_VOXEL_LIMIT = 32_000_000
FINAL_VOXEL_LIMIT = 400_000_000


def physical_memory():
    """Read-only physical RAM snapshot; unavailable platforms return None.

    This is a warning signal, not a new lower sample-count cap. The mesh and
    Boolean stages can exceed the simple per-voxel memory estimate.
    """
    if sys.platform != 'win32':
        return None
    try:
        import ctypes
        from ctypes import wintypes
        class MemoryStatus(ctypes.Structure):
            _fields_ = [('length', wintypes.DWORD), ('load', wintypes.DWORD)] + [
                (name, ctypes.c_ulonglong) for name in ('total', 'available', 'total_page',
                'available_page', 'total_virtual', 'available_virtual', 'extended')]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return dict(total_gb=round(status.total / 1e9, 2), available_gb=round(status.available / 1e9, 2))
    except (OSError, AttributeError):
        return None


def process_memory():
    """Worker-only measured high-water mark, not a system memory estimate."""
    if sys.platform!='win32':return None
    try:
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_=[('cb',wintypes.DWORD),('faults',wintypes.DWORD)]+[
                (k,ctypes.c_size_t) for k in ('peak_rss','rss','peak_paged','paged',
                  'peak_nonpaged','nonpaged','pagefile','peak_pagefile','private')]
        c=Counters();c.cb=ctypes.sizeof(c)
        fn=ctypes.windll.psapi.GetProcessMemoryInfo
        fn.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
        if not fn(wintypes.HANDLE(-1),ctypes.byref(c),c.cb):return None
        return dict(rss_gb=round(c.rss/1e9,3),peak_rss_gb=round(c.peak_rss/1e9,3),
                    private_gb=round(c.private/1e9,3),peak_commit_gb=round(c.peak_pagefile/1e9,3))
    except (OSError,AttributeError):return None


def atomic_json(path: Path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    # On Windows a concurrent HTTP reader can briefly prevent replace/delete.
    for attempt in range(30):
        try:
            temp.replace(path)
            break
        except PermissionError:
            if attempt == 29:
                raise
            time.sleep(.02)


def body_metadata(body, index):
    return dict(id=index, name=body.name, faces=body.face_info,
                bounds=body.mesh.bounds.tolist(), triangles=len(body.mesh.faces),
                watertight=bool(body.mesh.is_watertight))


def validate_design(data):
    cfg = dict(data)
    if cfg.get('lattice') not in LATTICE_TYPES:
        raise ValueError('请选择有效晶格类型')
    for key, default, low, high in [('cell', 4, .2, 100), ('wall', .6, .05, 50),
                                  ('depth', 6, .1, 500), ('shell', 0, 0, 50),
                                  ('pitch', .3, .05, 5), ('layer', 4, .2, 100),
                                  ('surface_thickness', .8, .1, 50),
                                  ('surface_cell',0,0,100),('surface_width',0,0,50),
                                  ('pore_rounding',0,0,10),('junction_blend',0,0,5),('edge_width',0,0,20)]:
        cfg[key] = float(cfg.get(key, default))
        if not math.isfinite(cfg[key]) or not low <= cfg[key] <= high:
            raise ValueError(f'{key} 必须在 {low}–{high} mm 之间')
    if cfg['wall'] >= cfg['cell']:
        raise ValueError('线径/壁厚必须小于单元大小')
    if cfg.get('axis') not in ('x', 'y', 'z'):
        raise ValueError('请选择 X / Y / Z 轴向')
    if cfg.get('scope') not in ('body', 'faces'):
        raise ValueError('无效的设计范围')
    cfg['unselected'] = cfg.get('unselected', 'perforate')
    if cfg['unselected'] not in ('solid', 'perforate'):
        raise ValueError('请选择未选区域的处理方式：沿曲面开孔或保持实心')
    cfg['surface_mode'] = cfg.get('surface_mode', 'perforated')
    if cfg['surface_mode'] not in ('volume', 'perforated'):
        raise ValueError('请选择曲面开孔或体积杆网模式')
    cfg['surface_pattern'] = cfg.get('surface_pattern', 'hex')
    if cfg['surface_pattern'] not in ('hex', 'square', 'circle'):
        raise ValueError('请选择六边形、方形或圆形曲面孔型')
    if (cfg['surface_width'] or cfg['wall']) >= (cfg['surface_cell'] or cfg['cell']):
        raise ValueError('曲面筋宽必须小于曲面孔距')
    if cfg['junction_blend'] > min(cfg['wall'],cfg['surface_width'] or cfg['wall'])/2:
        raise ValueError('连接圆滑半径不能超过最小杆径/筋宽的一半')
    cfg['memory_mode']=cfg.get('memory_mode','auto')
    if cfg['memory_mode'] not in ('auto','chunked'):
        raise ValueError('内存模式必须为 auto 或 chunked')
    cfg['surface_continuity']=cfg.get('surface_continuity','auto')
    if cfg['surface_continuity'] not in ('auto','off'):
        raise ValueError('连续连接模式必须为 auto 或 off')
    cfg['seam_style'] = cfg.get('seam_style', 'aligned')
    if cfg['seam_style'] not in ('atlas', 'vertex', 'aligned', 'adaptive', 'stitch', 'rail'):
        raise ValueError('无效的面间过渡模式')
    if cfg['seam_style'] in ('atlas','vertex') and cfg['surface_mode']=='perforated' and (cfg['lattice']!='bcc' or cfg['surface_continuity']!='auto'):
        raise ValueError('顶点连接模式需使用 BCC 且开启自动曲面连续连接')
    if cfg['seam_style']=='atlas' and cfg['surface_mode']=='perforated' and cfg['surface_pattern'] not in ('hex','square'):
        raise ValueError('曲面分区铺网支持六边形或方形孔型')
    cfg['design_version'] = 12
    for key in ('design_faces', 'preserve_faces'):
        ids = cfg.get(key, [])
        if not isinstance(ids, list) or any(type(x) is not int or x < 0 for x in ids):
            raise ValueError('无效面编号')
        cfg[key] = sorted(set(ids))
    if cfg['scope'] == 'faces' and not cfg['design_faces']:
        raise ValueError('请先点击选择要晶格化的外观面')
    if set(cfg['design_faces']) & set(cfg['preserve_faces']):
        raise ValueError('同一张面不能同时指定为设计面与保留面')
    if cfg['preserve_faces'] and cfg['shell'] <= 0:
        raise ValueError('选中保留面后，请设置大于 0 的保留厚度')
    overrides=cfg.get('preserve_thicknesses',{})
    if not isinstance(overrides,dict):raise ValueError('单面保留厚度必须是面编号与厚度的映射')
    normalized={}
    for key,value in overrides.items():
        if (not isinstance(key,str) or not key.isascii() or not key.isdigit() or
                str(int(key))!=key or int(key) not in cfg['preserve_faces']):
            raise ValueError('单面厚度只能用于当前橙色保留面')
        if isinstance(value,bool):raise ValueError('单面保留厚度必须为 0.1–50 mm')
        try:thickness=float(value)
        except (ValueError,TypeError):raise ValueError('单面保留厚度必须为 0.1–50 mm')
        if not math.isfinite(thickness) or not .1<=thickness<=50:
            raise ValueError('单面保留厚度必须为 0.1–50 mm')
        normalized[key]=thickness
    cfg['preserve_thicknesses']=normalized
    cfg['smooth'] = int(cfg.get('smooth', 2))
    if not 0 <= cfg['smooth'] <= 12:
        raise ValueError('平滑次数必须在 0–12 之间')
    cfg['quality'] = cfg.get('quality', 'preview')
    if cfg['quality'] not in ('preview', 'final'):
        raise ValueError('无效精度模式')
    return cfg


def estimate(bounds, cfg):
    extent = np.diff(np.asarray(bounds), axis=0)[0]
    pitch = cfg['pitch']
    limit = PREVIEW_VOXEL_LIMIT if cfg['quality'] == 'preview' else FINAL_VOXEL_LIMIT
    count = lambda h: int(np.prod(np.ceil(extent / h) + 8))
    requested = count(pitch)
    if cfg['quality'] == 'preview':
        while count(pitch) > limit:
            pitch *= 1.05
    warnings = []
    if cfg['surface_mode'] == 'perforated' and pitch > cfg['surface_thickness']/3:
        warnings.append('开孔面层不足 3 个采样间距，请减小采样间距或增加面层厚度。')
    if cfg['surface_mode']=='perforated' and pitch > (cfg['surface_width'] or cfg['wall'])/3+1e-8:
        warnings.append('曲面筋宽不足 3 个采样间距，请减小采样间距。')
    if pitch > cfg['wall'] / 3 + 1e-8:
        warnings.append('线径不足 3 个采样间距：细杆可能变细或断连，请减小采样间距。')
    retained=[cfg.get('preserve_thicknesses',{}).get(str(fid),cfg['shell']) for fid in cfg['preserve_faces']]
    min_retained=min(retained) if retained else cfg['wall']
    if retained and pitch > min_retained / 3 + 1e-8:
        warnings.append('保留层采样偏粗，可能出现孔洞或厚度误差。')
    if pitch > cfg['pitch'] * 1.001:
        warnings.append('大模型预览已自动降低采样精度；仅用于布局查看，不能作为制造结果。')
    from .lowmem import AUTO_THRESHOLD
    chunked=cfg.get('memory_mode','auto')=='chunked' or count(pitch)>AUTO_THRESHOLD
    legacy_memory_gb=round(count(pitch)*60/1e9,2)
    # Include large output topology/Boolean stages, not just the ~3 GB field
    # stage: the full 320M-point case can produce ~24M triangles.
    memory_gb=round(2+count(pitch)*40/1e9,2) if chunked else legacy_memory_gb
    scratch_gb=round((count(pitch)*(6+len(set(retained))+
        (cfg['scope']=='faces' and cfg['unselected']=='solid'))+1e9)/1e9,2) if chunked else 0
    ram = physical_memory()
    memory_risk = ram is not None and memory_gb > ram['available_gb'] * .8
    if memory_risk:
        warnings.append(f"粗估需 {memory_gb:.2f} GB，当前可用物理内存 {ram['available_gb']:.2f} GB；接近或超过可用内存，可能大量换页、卡顿或内存不足。")
    if count(pitch) > 80_000_000:
        warnings.append('高采样量任务：网格提取、曲面布尔和导出可能使峰值内存显著超过粗估值。')
    if chunked:
        warnings.append(f'已启用低内存分块：保持采样间距，临时磁盘预留约 {scratch_gb:g} GB；布尔和连通性检查仍随输出网格规模占用内存。')
    if ram is None:
        warnings.append('无法读取可用物理内存；请自行确认内存余量，粗估值不是峰值上限。')
    feature = min(cfg['wall'], cfg['surface_thickness'] if cfg['surface_mode']=='perforated' else cfg['wall'],
                  (cfg['surface_width'] or cfg['wall']) if cfg['surface_mode']=='perforated' else cfg['wall'],
                  min_retained)
    resolved = pitch <= feature/2.5 + 1e-8
    blocked = ''
    if count(pitch) > limit:
        blocked = f'超过本模式 {limit/1e6:g} 百万采样点上限，请缩小设计实体或增大采样间距。'
    elif not resolved:
        blocked = f'实际采样间距 {pitch:.3f} mm 不能解析 {feature:g} mm 的最小特征，已阻止生成碎片；请按设定精度生成（间距 ≤ {feature/2.5:.3f} mm），或使用较小实体。'
    return dict(pitch=round(float(pitch), 6), voxels=count(pitch), requested_voxels=requested,
                limit=limit, allowed=not blocked, allowed_by_count=count(pitch)<=limit,
                feature_resolved=resolved, blocked_reason=blocked,
                memory_gb=memory_gb, physical_memory=ram, memory_risk=memory_risk, warnings=warnings,
                memory_mode='chunked' if chunked else 'dense',scratch_gb=scratch_gb,legacy_memory_gb=legacy_memory_gb)


def compute(job_dir: Path, request):
    started=time.perf_counter()
    last_progress = None
    def progress(percent, message):
        nonlocal last_progress
        if last_progress == (percent, message):
            return
        last_progress = (percent, message)
        atomic_json(job_dir / 'status.json', dict(state='running', progress=percent, message=message,memory=process_memory()))
    progress(2, '准备数据')
    if request['action'] == 'load':
        bodies = demo_bodies() if request.get('demo') else load_cad(request['path'])
        save_bodies(bodies, job_dir)
        result = dict(bodies=[body_metadata(b, i) for i, b in enumerate(bodies)],
                      filename='示例曲壳' if request.get('demo') else Path(request['path']).name)
    elif request['action'] == 'redisplay':
        import shutil
        source_dir = Path(request['source_dir'])
        result = json.loads((source_dir/'design.json').read_text(encoding='utf-8'))
        progress(10, '读取完整 STL，旧结果保持不变')
        out = trimesh.load(source_dir/'result.stl', force='mesh')
        centre = out.bounds.mean(axis=0)
        out.apply_translation(-centre)
        progress(45, '重建通过流形检查的显示副本')
        quality = export_display(out, job_dir/'display.glb', result['config']['wall'], result['stats']['pitch'], centre)
        progress(95, '保存修复副本与检查记录')
        shutil.copyfile(source_dir/'result.stl', job_dir/'result.stl')
        result.update(display_quality=quality, display_faces=quality['faces'], redisplay_of=source_dir.name)
        result['warnings'].append('仅修复显示副本；原 STL 未改变。旧版排孔映射或欠采样造成的几何缺陷仍需重新生成。')
        atomic_json(job_dir/'design.json', result)
    else:
        cfg = validate_design(request['config'])
        model_dir = Path(request['model_dir'])
        body = read_body(model_dir, int(request['body']))
        selected = body.selected_mesh(cfg['design_faces']) if cfg['scope'] == 'faces' else None
        protected = body.selected_mesh(cfg['preserve_faces']) if cfg['preserve_faces'] else None
        retained_groups={}
        for fid in cfg['preserve_faces']:
            thickness=cfg['preserve_thicknesses'].get(str(fid),cfg['shell'])
            retained_groups.setdefault(thickness,[]).append(fid)
        preserve_regions=[(body.selected_mesh(ids),thickness) for thickness,ids in sorted(retained_groups.items())]
        est = estimate(body.mesh.bounds, cfg)
        if not est['allowed']:
            raise ValueError(est['blocked_reason'])
        warnings = est['warnings'][:]
        if not body.mesh.is_watertight:
            warnings.append('源实体网格未闭合：内部判定、几何尺寸和密度仅供参考。')
        # Work near the origin; avoid float32 precision loss at large STEP offsets.
        centre = body.mesh.bounds.mean(axis=0)
        surface_field = None
        if cfg['surface_mode'] == 'perforated':
            upgrade_face_charts(body, model_dir, int(request['body']))
            active = ({f['id'] for f in body.face_info} if cfg['scope'] == 'body' or cfg['unselected'] == 'perforate'
                      else set(cfg['design_faces'])) - set(cfg['preserve_faces'])
            surface_field = PerforatedSurface(body, active, translation=-centre, chart_mode=cfg['seam_style'])
            if surface_field.approximate:
                warnings.append(f'{len(surface_field.approximate)} 张自由曲面使用通过畸变检查的局部投影，孔距仍可能变化，请检查面间接缝。')
            if cfg['seam_style'] == 'adaptive':
                warnings.append('自适应模式在全部设计曲面上统一取点，使用不规则多边形孔网；孔距为平均设计尺度，不再逐面套用固定孔型。')
            if cfg['seam_style'] in ('aligned','vertex'):
                warnings.append('定向模式使用跨面规则孔列及共享相位，曲率变化处允许局部改变孔边数；轴向曲壳和近平面适用，不是通用自由曲面参数化。孔距为目标尺度。')
            if cfg['seam_style']=='atlas':
                warnings.append('曲面分区铺网沿局部切平面排布六边形/方形，检查整片设计面覆盖；曲率变化、修剪边界和分区接合处允许过渡多边形，并非全局等边孔网。')
            if protected is not None:
                warnings.append('橙色保留面仍为实心层；如需内外贯通开孔，请取消对应内侧面的保留标记。')
        for mesh in (body.mesh, selected, protected):
            if mesh is not None:
                mesh.apply_translation(-centre)
        for face,_ in preserve_regions:face.apply_translation(-centre)
        p = FillParams(lattice=cfg['lattice'], cell=cfg['cell'], wall=cfg['wall'],
                       shell=cfg['shell'] if protected is not None else 0,
                       preserve_thicknesses=cfg['preserve_thicknesses'],
                       layer=cfg['layer'], axis=cfg['axis'], pitch=est['pitch'],
                       region_depth=cfg['depth'], max_voxels=est['limit'] + 100000,
                       boundary_blur=.6, smooth=cfg['smooth'], unselected=cfg['unselected'],
                       surface_mode=cfg['surface_mode'], surface_pattern=cfg['surface_pattern'],
                       surface_thickness=cfg['surface_thickness'],surface_cell=cfg['surface_cell'],
                       surface_width=cfg['surface_width'],pore_rounding=cfg['pore_rounding'],
                       junction_blend=cfg['junction_blend'],edge_width=cfg['edge_width'],
                       surface_continuity=cfg['surface_continuity'],
                       seam_style=cfg['seam_style'],
                       memory_mode=cfg['memory_mode'],work_dir=str(job_dir))
        if request.get('resume_from'):
            source_dir=Path(request['resume_from'])
            original=json.loads((source_dir/'request.json').read_text(encoding='utf-8'))
            if (validate_design(original['config'])!=cfg or Path(original['model_dir']).resolve()!=model_dir.resolve()
                    or int(original['body'])!=int(request['body'])):
                raise ValueError('恢复检查点的模型、选面或参数不一致')
            from .mesh_checkpoint import load_pretrim
            from .core import _finish_mesh
            out,pitch,density=load_pretrim(source_dir,p)
            progress(87,'已恢复裁切前网格，不重复铺网或平滑')
            out=_finish_mesh(out,body.mesh,p,pitch,density,progress,pretrim_ready=True)
        else:
            out = fill_solid(body.mesh, p, design_faces=selected, preserve_faces=protected,
                             preserve_regions=preserve_regions if protected is not None else None,
                             surface_field=surface_field, progress=progress)
        p.stats['preserve_thicknesses']=dict(cfg['preserve_thicknesses'])
        if cfg['preserve_thicknesses']:
            warnings.append('单独保留厚度：'+ '，'.join(f'面 {int(fid)+1:03d} = {thickness:g} mm' for fid,thickness in cfg['preserve_thicknesses'].items())+'；未单独设置的保留面仍用默认厚度。厚度向实体内部延伸，并受原实体厚度限制。')
        if cfg['seam_style'] in ('atlas','vertex') and cfg['surface_mode']=='perforated':
            warnings.append(f"顶点模式替换 {p.stats['removed_old_outer_members']} 根旧外层杆，{p.stats['vertex_connected_nodes']} 个多边形顶点通过短柱和内侧直杆接入；{p.stats['vertex_posts_unconnected']} 个顶点未加内接柱。非顶点落点为 0；薄壁处允许相邻顶点内接。这是拓扑调整，制造前须验证局部刚度。")
        elif p.stats.get('omitted_boundary_members',0):
            warnings.append(f"直杆匹配为避免回折或交叉，局部省略 {p.stats['omitted_boundary_members']} / {p.stats['boundary_members_considered']} 根边界杆；这属于拓扑调整，不是数值碎片清理，制造前须验证局部刚度。")
        if cfg['surface_mode'] == 'perforated' and not p.stats['source_trimmed']:
            warnings.append('源网格或中间网格不是闭合正体积，未执行原曲面布尔裁切；当前仅为体素近似，不保证原曲面精度。')
        progress(89, '检查连通性与清理数值碎片')
        feature=min(cfg['wall'],cfg['surface_width'] or cfg['wall'],cfg['surface_thickness']) if cfg['surface_mode']=='perforated' else cfg['wall']
        out, topology = clean_fragments(out, feature)
        p.stats.update(topology, faces=len(out.faces), watertight=True,
                       mesh_density=float(topology['volume']/body.mesh.volume) if body.mesh.is_volume else None)
        if topology['removed_fragments']:
            warnings.append(f"已清理 {topology['removed_fragments']} 个亚线径数值碎片，体积 {topology['removed_volume']:.6f} mm³。")
        if topology['components'] > 1:
            warnings.append(f"结果仍有 {topology['components']} 个独立连通体；未擅自删除较大结构，制造前须检查连接。")
        progress(91, '按误差容限生成流形显示网格')
        display_quality = export_display(out, job_dir/'display.glb', feature, est['pitch'], centre)
        progress(97, '保存完整网格')
        from .mesh_memory import export_stl_batched,release_cache
        release_cache(out)
        output_bounds=out.bounds+centre
        result = dict(stats=p.stats, warnings=warnings, config=cfg, estimate=est,
                      bounds=output_bounds.tolist(), display_faces=display_quality['faces'], display_quality=display_quality,
                      bytes=84+50*len(out.faces),
                      source_body=int(request['body']), model_id=model_dir.name)
        # A failed STL write must not lose computed statistics or the validated
        # display geometry. This checkpoint is never exposed as a done result.
        atomic_json(job_dir/'export_checkpoint.json',result)
        export_stl_batched(out,job_dir/'result.stl',translation=centre)
        result['bytes']=(job_dir/'result.stl').stat().st_size
        atomic_json(job_dir / 'design.json', result)
    result['performance']=dict(elapsed_seconds=round(time.perf_counter()-started,2),memory=process_memory(),
        scope='postprocess_only' if result.get('stats',{}).get('resumed_from') else 'full')
    if 'config' in result:atomic_json(job_dir/'design.json',result)
    atomic_json(job_dir / 'status.json', dict(state='done', progress=100, message='完成', result=result))


def main():
    job_dir = Path(sys.argv[1]).resolve()
    try:
        request = json.loads((job_dir / 'request.json').read_text(encoding='utf-8'))
        compute(job_dir, request)
    except Exception as exc:
        traceback.print_exc()
        atomic_json(job_dir / 'status.json', dict(state='failed', progress=0, message=str(exc)))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

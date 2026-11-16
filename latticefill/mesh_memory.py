"""Bounded mesh post-processing: no whole-mesh edge/triangle working copies."""
import struct
import numpy as np

FACE_BATCH = 65_536


def release_cache(mesh):
    # Trimesh caches triangles, edges and adjacency, often much larger than the
    # actual indexed mesh. Geometry and visual attributes are not changed.
    mesh._cache.clear()


def remove_degenerate_batched(mesh):
    from trimesh.triangles import nondegenerate
    keep=np.empty(len(mesh.faces),bool)
    for k in range(0,len(mesh.faces),FACE_BATCH):
        keep[k:k+FACE_BATCH]=nondegenerate(mesh.vertices[mesh.faces[k:k+FACE_BATCH]])
    mesh.update_faces(keep)


def smooth_taubin_batched(mesh, iterations, lamb=.5, nu=.5):
    """Same equal-weight directed-edge Taubin passes on an oriented manifold.

    Each directed edge occurs once on a consistently wound manifold; accumulate
    its neighbor without constructing the full sparse adjacency operator.
    """
    release_cache(mesh)
    vertices=np.array(mesh.vertices)
    faces=mesh.faces.view(np.ndarray)
    degree=np.zeros(len(vertices),np.uint32)
    for k in range(0,len(faces),FACE_BATCH):
        np.add.at(degree,faces[k:k+FACE_BATCH].ravel(),1)
    divisor=np.maximum(degree,1)[:,None]
    for iteration in range(iterations):
        delta=np.zeros_like(vertices)
        for k in range(0,len(faces),FACE_BATCH):
            f=faces[k:k+FACE_BATCH]
            for a,b in ((0,1),(1,2),(2,0)):
                np.add.at(delta,f[:,a],vertices[f[:,b]])
        delta/=divisor
        delta-=vertices
        delta[degree==0]=0
        delta*=lamb if iteration%2==0 else -nu
        vertices+=delta
    mesh.vertices=vertices
    return mesh


def export_stl_batched(mesh,path,translation=None):
    """Space-check and atomically publish STL using bounded triangle batches.

    The old destination stays intact until a flushed, synced sibling partial
    has the exact binary STL length and header. Only our own partial is cleaned
    on failure; unrelated partials and historical outputs are never removed.
    """
    import errno
    import os
    import shutil
    import tempfile
    from pathlib import Path

    face_count=len(mesh.faces)
    if face_count>0xffffffff:
        raise ValueError('STL 三角面数超过格式上限')
    destination=Path(os.path.abspath(os.fspath(path)))
    expected_bytes=84+50*face_count
    # Keep at least 64 MiB, or 5% for large outputs, beyond the new file.
    # Existing destinations still occupy disk while the partial is written.
    reserve=max(64*1024**2,(expected_bytes+19)//20)
    free=shutil.disk_usage(destination.parent).free
    if free<expected_bytes+reserve:
        raise OSError(errno.ENOSPC,
            f'STL export needs {expected_bytes} bytes plus {reserve} bytes reserve; '
            f'only {free} bytes free on destination disk',str(destination))
    shift=np.zeros(3) if translation is None else np.asarray(translation)
    dtype=np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attribute','<u2')])
    header=b'Lattice Studio streamed STL'.ljust(80,b'\0')+struct.pack('<I',face_count)
    fd,partial=tempfile.mkstemp(prefix=f'.{destination.name}.',suffix='.partial',
                               dir=destination.parent)
    try:
        try:
            stream=os.fdopen(fd,'wb')
        except BaseException:
            os.close(fd)
            raise
        with stream:
            if stream.write(header)!=len(header):
                raise OSError(errno.EIO,'Short write in STL header',partial)
            for k in range(0,face_count,FACE_BATCH):
                triangles=mesh.vertices[mesh.faces[k:k+FACE_BATCH]]
                normal=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
                length=np.linalg.norm(normal,axis=1)
                np.divide(normal,length[:,None],out=normal,where=length[:,None]>0)
                records=np.zeros(len(triangles),dtype)
                records['normal']=normal
                records['vertices']=triangles+shift
                data=records.tobytes()
                if stream.write(data)!=len(data):
                    raise OSError(errno.EIO,'Short write in STL triangles',partial)
            stream.flush()
            os.fsync(stream.fileno())
        # Reopen after close so buffered/close failures cannot publish a final.
        with open(partial,'rb') as check:
            actual_bytes=os.fstat(check.fileno()).st_size
            if actual_bytes!=expected_bytes or check.read(84)!=header:
                raise OSError(errno.EIO,
                    f'Incomplete STL: expected {expected_bytes} bytes, got {actual_bytes}',partial)
        os.replace(partial,destination)
    except BaseException as error:
        try:
            os.unlink(partial)
        except FileNotFoundError:
            pass
        except OSError as cleanup_error:
            error.add_note(f'Could not remove own STL partial {partial}: {cleanup_error}')
        raise


def clean_components_batched(mesh,wall):
    """Connectivity labels and signed volumes, not a copy of every component.

    Caller has already validated the oriented manifold. The sparse graph uses
    32-bit indices when possible; each triangle contributes its directed cycle.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    import trimesh
    dtype=np.int32 if len(mesh.vertices)<2**31 else np.int64
    f=mesh.faces.astype(dtype)
    graph=coo_matrix((np.ones(f.size,bool),(f.ravel(),np.roll(f,-1,axis=1).ravel())),
                     shape=(len(mesh.vertices),)*2).tocsr()
    del f
    count,labels=connected_components(graph,directed=False)
    del graph
    volumes=np.zeros(count,np.float64);has_faces=np.zeros(count,bool)
    centre=mesh.bounds.mean(axis=0)
    for k in range(0,len(mesh.faces),FACE_BATCH):
        faces=mesh.faces[k:k+FACE_BATCH]
        part=labels[faces[:,0]]
        triangles=mesh.vertices[faces]-centre
        v=np.einsum('ij,ij->i',triangles[:,0],np.cross(triangles[:,1],triangles[:,2]))/6
        np.add.at(volumes,part,v);has_faces[part]=True
    threshold=wall**3/10
    tiny=(np.abs(volumes)<threshold)&has_faces
    tiny[np.argmax(np.abs(volumes))]=False
    removed_volume=float(np.abs(volumes[tiny]).sum())
    if removed_volume>.005*np.abs(volumes).sum():
        raise ValueError('孤立碎片过多（超过材料体积的 0.5%），不能自动修补；请提高采样精度或缩小设计区域')
    if tiny.any():
        keep=np.empty(len(mesh.faces),bool)
        used=np.zeros(len(mesh.vertices),bool)
        for k in range(0,len(mesh.faces),FACE_BATCH):
            faces=mesh.faces[k:k+FACE_BATCH]
            mask=~tiny[labels[faces[:,0]]]
            keep[k:k+FACE_BATCH]=mask;used[faces[mask].ravel()]=True
        remap=np.cumsum(used,dtype=dtype)-1
        mesh=trimesh.Trimesh(mesh.vertices[used],remap[mesh.faces[keep]],process=False)
    return mesh,dict(components=int(np.count_nonzero(has_faces&~tiny)),
        removed_fragments=int(tiny.sum()),removed_volume=removed_volume,
        fragment_threshold_mm3=threshold,volume=float(volumes[~tiny].sum()))

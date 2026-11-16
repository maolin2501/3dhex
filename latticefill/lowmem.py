"""Disk-backed lattice fields, bounded surface rasterization and slab meshing.

No full-mesh subdivision, full-grid EDT or full-grid coordinate/index arrays.
All geometry stays on ONE global grid: blocks share samples, not capped solids.
"""
from pathlib import Path
import tempfile
import shutil
import time
import re
import numpy as np
import trimesh
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage import measure
from .lattices import lattice_phi

AUTO_THRESHOLD = 8_000_000
FIELD_BLOCK_POINTS = 4_000_000
# Bound each EDT including its halo. Deep routing bands are tiled in XY before
# falling back to the global KD tree; increasing the band must not force millions
# of slow point queries when a pair of overlapping EDT tiles suffices.
EDT_BLOCK_POINTS = 32_000_000


class Scratch:
    def __init__(self, parent=None):
        self.temp=tempfile.TemporaryDirectory(prefix='scratch-lowmem-',dir=parent)
        self.path=Path(self.temp.name)
        self.maps=[]

    def array(self,name,shape,dtype):
        arr=np.memmap(self.path/name,mode='w+',dtype=dtype,shape=shape,order='F')
        self.maps.append(arr)
        return arr

    def __enter__(self):return self

    def __exit__(self,*exc):
        for arr in self.maps:
            arr.flush();arr._mmap.close()
        self.maps.clear()
        self.temp.cleanup()


def cleanup_job_scratch(directory):
    """After an isolated worker stops, delete only recognized scratch fields."""
    root=Path(directory).resolve()
    allowed={'surface.bin','preserve.bin','design.bin','field.bin','filtered.bin'}
    for child in root.glob('scratch-lowmem-*'):
        if child.is_symlink() or child.is_junction() or not child.is_dir():continue
        resolved=child.resolve()
        if resolved.parent!=root:continue
        entries=list(resolved.iterdir())
        if any((p.name not in allowed and re.fullmatch(r'preserve-[0-9]+\.bin',p.name) is None)
               or not p.is_file() or p.is_symlink() for p in entries):continue
        for attempt in range(20):
            try:
                shutil.rmtree(resolved)
                break
            except PermissionError:
                if attempt==19:raise
                # Windows can retain a killed process's mappings briefly.
                time.sleep(.1)


def grid_spec(bounds,pitch):
    # Put the longest axis last, bounding slab cross sections even for plates.
    lo=np.floor(np.asarray(bounds[0])/pitch).astype(np.int64)-3
    hi=np.ceil(np.asarray(bounds[1])/pitch).astype(np.int64)+3
    shape=hi-lo+1
    last=int(np.argmax(shape))
    order=[i for i in range(3) if i!=last]+[last]
    return tuple(int(v) for v in shape[order]),lo[order]*pitch,np.array(order)


def projected_samples(low,high):
    cols=min(int(high[1]-low[1]+1),65536)
    rows=max(1,65536//cols)
    for i in range(int(low[0]),int(high[0])+1,rows):
        for j in range(int(low[1]),int(high[1])+1,cols):
            u,v=np.meshgrid(np.arange(i,min(i+rows,high[0]+1)),
                            np.arange(j,min(j+cols,high[1]+1)),indexing='ij')
            yield u.ravel(),v.ravel()


def rasterize(mesh,output,origin,pitch,order=(0,1,2),value=1):
    """Conservative dominant-plane triangle raster, bounded per-triangle rows.

    The half-cell coverage avoids losing narrow CAD strips. This is a surface
    seed raster only: solid occupancy comes separately from VTK scan conversion.
    """
    shape=np.asarray(output.shape)
    vertices=(mesh.vertices[:,order]-origin)/pitch
    for face in mesh.faces:
        t=vertices[face]
        normal=np.cross(t[1]-t[0],t[2]-t[0])
        major=int(np.argmax(abs(normal)))
        if abs(normal[major])<1e-12:continue
        axes=[i for i in range(3) if i!=major]
        p=t[:,axes]
        e1,e2=p[1]-p[0],p[2]-p[0]
        det=e1[0]*e2[1]-e1[1]*e2[0]
        if abs(det)<1e-12:continue
        low=np.maximum(np.ceil(p.min(axis=0)-.5).astype(int),0)
        high=np.minimum(np.floor(p.max(axis=0)+.5).astype(int),shape[axes]-1)
        if np.any(high<low):continue
        # At most 65k projected samples, irrespective of triangle edge length.
        for u,v in projected_samples(low,high):
            valid=np.ones(len(u),dtype=bool)
            for a,b in zip(p,np.roll(p,-1,axis=0)):
                e=b-a
                side=(e[0]*(v-a[1])-e[1]*(u-a[0]))*np.sign(det)
                valid &= side>=-.5*(abs(e[0])+abs(e[1]))-1e-9
            u,v=u[valid],v[valid]
            w=t[0,major]-(normal[axes[0]]*(u-t[0,axes[0]])+normal[axes[1]]*(v-t[0,axes[1]]))/normal[major]
            w=np.rint(np.clip(w,t[:,major].min(),t[:,major].max())).astype(np.int64)
            valid=(w>=0)&(w<shape[major])
            index=[None]*3;index[axes[0]]=u[valid];index[axes[1]]=v[valid];index[major]=w[valid]
            output[tuple(index)]=value


class SolidStencil:
    """Closed triangle surface -> exact scan-line occupancy, one slab at a time."""
    def __init__(self,mesh,shape,origin,pitch,order):
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk,numpy_to_vtkIdTypeArray
        self.shape=shape
        points=vtk.vtkPoints();points.SetData(numpy_to_vtk(np.ascontiguousarray(mesh.vertices[:,order]),deep=True))
        cells=vtk.vtkCellArray()
        cells.SetData(numpy_to_vtkIdTypeArray(np.arange(len(mesh.faces)+1,dtype=np.int64)*3,deep=True),
                      numpy_to_vtkIdTypeArray(mesh.faces.ravel().astype(np.int64),deep=True))
        poly=vtk.vtkPolyData();poly.SetPoints(points);poly.SetPolys(cells)
        self.stencil=vtk.vtkPolyDataToImageStencil();self.stencil.SetInputData(poly)
        self.stencil.SetOutputOrigin(*origin);self.stencil.SetOutputSpacing(pitch,pitch,pitch)
        self.stencil.SetTolerance(1e-6)
        self.image=vtk.vtkImageStencilToImage();self.image.SetInputConnection(self.stencil.GetOutputPort())
        self.image.SetInsideValue(1);self.image.SetOutsideValue(0);self.image.SetOutputScalarTypeToUnsignedChar()

    def inside(self,k0,k1):
        from vtk.util.numpy_support import vtk_to_numpy
        nx,ny,_=self.shape
        self.stencil.SetOutputWholeExtent(0,nx-1,0,ny-1,k0,k1-1)
        self.image.Update()
        return vtk_to_numpy(self.image.GetOutput().GetPointData().GetScalars()).reshape((nx,ny,k1-k0),order='F').astype(bool)


class SeedField:
    def __init__(self,seeds,pitch):
        self.seeds=seeds;self.pitch=pitch;self.tree=None;self.values=None

    def _tree(self):
        points=[];values=[]
        step=max(1,FIELD_BLOCK_POINTS//(self.seeds.shape[0]*self.seeds.shape[1]))
        for k in range(0,self.seeds.shape[2],step):
            slab=self.seeds[:,:,k:k+step]
            ijk=np.argwhere(slab!=0)
            if len(ijk):
                values.append(np.array(slab[tuple(ijk.T)]));ijk[:,2]+=k
                points.append(ijk*self.pitch)
        if not points:raise ValueError('曲面栅格为空，无法计算距离')
        self.tree=cKDTree(np.vstack(points));self.values=np.concatenate(values)

    def query(self,k0,k1,limit,nearest=False):
        halo=int(np.ceil(limit/self.pitch))+1
        a=max(0,k0-halo);b=min(self.seeds.shape[2],k1+halo)
        shape=(*self.seeds.shape[:2],k1-k0)
        if np.prod(self.seeds.shape[:2])*(b-a)<=EDT_BLOCK_POINTS:
            seeds=np.asarray(self.seeds[:,:,a:b])
            if not np.any(seeds):
                return np.full(shape,limit,np.float32),np.zeros(shape,self.seeds.dtype) if nearest else None
            if nearest:
                distance,indices=ndimage.distance_transform_edt(seeds==0,sampling=self.pitch,return_indices=True)
                cut=indices[:,:,:,k0-a:k1-a]
                labels=np.array(seeds[tuple(cut)])
                del cut,indices
            else:
                distance=ndimage.distance_transform_edt(seeds==0,sampling=self.pitch);labels=None
            d=np.array(distance[:,:,k0-a:k1-a],dtype=np.float32)
            if labels is not None:labels[d>=limit]=0
            np.minimum(d,limit,out=d)
            return d,labels
        # Exact truncated distances: every retained voxel has a full limit-sized
        # halo in XYZ (or reaches the global boundary). Split core XY rectangles,
        # NOT disconnected solids, and copy only the non-overlapping cores back.
        # If even the smallest practical halo cannot fit, retain the KD fallback.
        stack=[(0,shape[0],0,shape[1])];tiles=[];possible=EDT_BLOCK_POINTS>0
        while stack and possible:
            x0,x1,y0,y1=stack.pop()
            xa,xb=max(0,x0-halo),min(shape[0],x1+halo)
            ya,yb=max(0,y0-halo),min(shape[1],y1+halo)
            if (xb-xa)*(yb-ya)*(b-a)<=EDT_BLOCK_POINTS:
                tiles.append((x0,x1,y0,y1,xa,xb,ya,yb));continue
            if max(x1-x0,y1-y0)<=max(16,halo):possible=False;break
            if x1-x0>=y1-y0:
                mid=(x0+x1)//2;stack.extend([(x0,mid,y0,y1),(mid,x1,y0,y1)])
            else:
                mid=(y0+y1)//2;stack.extend([(x0,x1,y0,mid),(x0,x1,mid,y1)])
        if possible:
            d=np.full(shape,limit,np.float32)
            labels=np.zeros(shape,self.seeds.dtype) if nearest else None
            for x0,x1,y0,y1,xa,xb,ya,yb in tiles:
                seeds=np.asarray(self.seeds[xa:xb,ya:yb,a:b])
                if not np.any(seeds):continue
                cut=(slice(x0-xa,x1-xa),slice(y0-ya,y1-ya),slice(k0-a,k1-a))
                if nearest:
                    distance,indices=ndimage.distance_transform_edt(seeds==0,sampling=self.pitch,return_indices=True)
                    tile_labels=seeds[tuple(indices[(slice(None),)+cut])]
                    labels[x0:x1,y0:y1]=tile_labels
                    del tile_labels,indices
                else:distance=ndimage.distance_transform_edt(seeds==0,sampling=self.pitch)
                d[x0:x1,y0:y1]=distance[cut]
                del distance
            if nearest:labels[d>=limit]=0
            np.minimum(d,limit,out=d)
            return d,labels
        # Very deep local-design bands must not expand a halo to the full grid.
        if self.tree is None:self._tree()
        count=int(np.prod(shape));d=np.empty(count,np.float32)
        labels=np.zeros(count,self.seeds.dtype) if nearest else None
        for start in range(0,count,65536):
            stop=min(count,start+65536)
            pts=np.column_stack(np.unravel_index(np.arange(start,stop),shape)).astype(float)
            pts[:,2]+=k0
            dist,idx=self.tree.query(pts*self.pitch,distance_upper_bound=limit)
            valid=np.isfinite(dist);d[start:stop]=np.minimum(dist,limit)
            if nearest:labels[start:stop][valid]=self.values[idx[valid]]
        return d.reshape(shape),labels.reshape(shape) if nearest else None


def marching_blocks(phi,pitch,origin,order,block_cells=8,progress=None):
    """March disjoint cell slabs and weld ONLY their common grid plane."""
    level=float(np.float32(1.2345e-3)*pitch)
    vertices=[];faces=[];previous={};offset=0
    for k0 in range(0,phi.shape[2]-1,block_cells):
        k1=min(phi.shape[2]-1,k0+block_cells)
        slab=np.ascontiguousarray(phi[:,:,k0:k1+1])
        if not slab.min()<level<slab.max():previous={};continue
        v,f,_,_=measure.marching_cubes(slab,level=level,spacing=(pitch,)*3)
        bottom=np.flatnonzero(abs(v[:,2])<pitch*1e-5)
        top=np.flatnonzero(abs(v[:,2]-(k1-k0)*pitch)<pitch*1e-5)
        ids=np.full(len(v),-1,np.int64)
        for i in bottom:
            key=tuple(np.round(v[i,:2]/pitch,5))
            if key in previous:ids[i]=previous[key]
        fresh=ids<0
        ids[fresh]=np.arange(offset,offset+np.count_nonzero(fresh));offset+=np.count_nonzero(fresh)
        previous={tuple(np.round(v[i,:2]/pitch,5)):int(ids[i]) for i in top}
        v[:,2]+=k0*pitch
        world=(v[fresh]+origin)[:,np.argsort(order)]
        vertices.append(world);faces.append(ids[f])
        if progress:progress(73+int(10*k1/(phi.shape[2]-1)),'分块提取网格并拼接共享边界')
    if not faces:raise ValueError('当前参数没有可提取的晶格')
    # The output mesh (unlike temporary volume grids) necessarily scales with
    # geometric complexity. No full-grid int64 edge table is created here.
    return trimesh.Trimesh(np.vstack(vertices),np.vstack(faces),process=False)


def fill_chunked(mesh,p,design_faces,preserve_faces,surface_field,progress,preserve_regions=None):
    from .core import _finish_mesh,_save_slice_preview
    if not mesh.is_volume:
        raise ValueError('低内存扫描需要闭合且朝向一致的源实体，请先修复源 STEP')
    pitch=p.pitch if p.pitch is not None else p.cell/p.res
    shape,origin,order=grid_spec(mesh.bounds,pitch)
    count=int(np.prod(shape));nxy=shape[0]*shape[1]
    dtype=np.uint16 if surface_field is None or max(surface_field.meshes)<65534 else np.uint32
    explicit_preserve=preserve_faces is not None or preserve_regions is not None
    groups=(list(preserve_regions) if preserve_regions is not None else
            [(preserve_faces,p.shell)] if preserve_faces is not None and p.shell>0 else [])
    required=count*(np.dtype(dtype).itemsize+4+len(groups)+
        bool(design_faces is not None and p.unselected=='solid')+4*bool(p.blur))+1_000_000_000
    disk=shutil.disk_usage(p.work_dir or tempfile.gettempdir())
    if disk.free<required:
        raise ValueError(f'低内存计算临时磁盘空间不足：至少需要约 {required/1e9:.1f} GB')
    block=max(1,min(32,FIELD_BLOCK_POINTS//nxy))
    with Scratch(p.work_dir) as scratch:
        labels=scratch.array('surface.bin',shape,dtype)
        progress(10,'逐面栅格化边界，不细分整网格')
        if surface_field is not None:
            for i,(fid,face) in enumerate(surface_field.meshes.items()):
                rasterize(face,labels,origin,pitch,order,fid+1)
                progress(10+int(12*(i+1)/len(surface_field.meshes)),'逐面栅格化边界')
            surface_field.origin=origin;surface_field.pitch=pitch
            surface_field.grid_order=order
            progress(23,'构建曲面接缝与连续连杆转接')
            surface_field.prepare_edges(pitch,p)
        else:rasterize(mesh,labels,origin,pitch,order)
        source=SeedField(labels,pitch)
        keep=[];region=None
        for index,(face,thickness) in enumerate(groups):
            seeds=scratch.array(f'preserve-{index}.bin',shape,np.uint8)
            rasterize(face,seeds,origin,pitch,order);keep.append((SeedField(seeds,pitch),thickness))
        if design_faces is not None and p.unselected=='solid':
            seeds=scratch.array('design.bin',shape,np.uint8)
            rasterize(design_faces,seeds,origin,pitch,order);region=SeedField(seeds,pitch)
        progress(28,'准备低内存实体扫描与磁盘场')
        stencil=SolidStencil(mesh,shape,origin,pitch,order)
        phi=scratch.array('field.bin',shape,np.float32)
        cap=max([p.wall,p.shell,p.surface_thickness,p.edge_width]+[t for _,t in groups])+4*pitch
        if surface_field is not None:
            cap=max(cap,surface_field.depth_limit(p)+4*pitch)
        gaussian_halo=int(np.ceil(4*p.boundary_blur))
        halo=max(2,gaussian_halo)
        body_vox=solid_vox=0
        axes=[origin[i]+pitch*np.arange(shape[i],dtype=np.float32) for i in range(3)]
        for k0 in range(0,shape[2],block):
            k1=min(shape[2],k0+block);a=max(0,k0-halo);b=min(shape[2],k1+halo)
            d,nearest=source.query(a,b,cap,nearest=surface_field is not None)
            inside=stencil.inside(a,b)
            s=np.where(inside,d+pitch/2,-d-pitch/2).astype(np.float32)
            body_vox+=int(np.count_nonzero(inside[:,:,k0-a:k1-a]))
            del d,inside
            if p.boundary_blur:ndimage.gaussian_filter(s,p.boundary_blur,output=s)
            grid=np.broadcast_arrays(axes[0][:,None,None],axes[1][None,:,None],axes[2][None,None,a:b])
            world=[None]*3
            for i in range(3):world[order[i]]=grid[i]
            lat=lattice_phi(p.lattice,*world,p.cell,p.wall,pitch,layer=p.layer,axis=p.axis)
            lat=lat[:,:,k0-a:k1-a];s=s[:,:,k0-a:k1-a]
            if surface_field is not None:
                lat=surface_field.combine_nearest(lat,s,k0,p,nearest[:,:,k0-a:k1-a])
            if region is not None:
                dist,_=region.query(k0,k1,p.region_depth+cap)
                lat=np.maximum(lat,dist-p.region_depth)
            for keep_field,thickness in keep:
                dist,_=keep_field.query(k0,k1,thickness+cap)
                lat=np.maximum(lat,thickness-dist)
            if not explicit_preserve and p.shell>0:lat=np.maximum(lat,p.shell-s)
            ph=np.minimum(s,lat).astype(np.float32)
            phi[:,:,k0:k1]=ph;solid_vox+=int(np.count_nonzero(ph>0))
            progress(30+int(40*k1/shape[2]),'低内存分块计算晶格与曲面')
        # Close field intermediates before the output mesh grows.
        if keep:del keep_field
        del lat,s,ph,nearest,source,keep,region,stencil
        phi.flush()
        if p.blur:
            filtered=scratch.array('filtered.bin',shape,np.float32)
            h=int(np.ceil(4*p.blur))
            for k0 in range(0,shape[2],block):
                k1=min(shape[2],k0+block);a=max(0,k0-h);b=min(shape[2],k1+h)
                filtered[:,:,k0:k1]=ndimage.gaussian_filter(np.array(phi[:,:,a:b]),p.blur)[:,:,k0-a:k1-a]
            phi=filtered
        if p.preview:_save_slice_preview(np.transpose(phi,np.argsort(order)),p.preview)
        out=marching_blocks(phi,pitch,origin,order,block_cells=block,progress=progress)
    p.stats.update(memory_mode='chunked',grid_shape=list(shape),grid_axes=order.tolist(),
                   grid_points=count,block_slices=block,scratch_estimate_gb=required/1e9)
    return _finish_mesh(out,mesh,p,pitch,solid_vox/max(body_vox,1),progress)

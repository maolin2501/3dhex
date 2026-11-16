"""Native CAD helpers for the SLA radial-lattice prototype."""
from pathlib import Path
import json, math, time, sys
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from OCP.gp import gp_Pnt, gp_Vec, gp_Ax2, gp_Dir, gp_Circ
from OCP.BRep import BRep_Builder
from OCP.TopoDS import TopoDS_Shape, TopoDS_Compound
from OCP.BRepTools import BRepTools
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepPrimAPI import BRepPrimAPI_MakeSphere, BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakePrism, BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCone
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace
from OCP.BRepOffsetAPI import BRepOffsetAPI_MakePipe
from OCP.GeomAPI import GeomAPI_Interpolate
from OCP.TColgp import TColgp_HArray1OfPnt
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.TopAbs import TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopTools import TopTools_ListOfShape
from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs, STEPControl_Reader

ROOT=Path(__file__).resolve().parent
C=1314.8968633141
from cad_io_local import tessellate_shape
P={'mesh_linear_deflection':.04}

def solids(s):
    ex=TopExp_Explorer(s,TopAbs_SOLID); out=[]
    while ex.More(): out.append(ex.Current()); ex.Next()
    return out

def compound(ss):
    b=BRep_Builder(); c=TopoDS_Compound(); b.MakeCompound(c)
    for s in ss: b.Add(c,s)
    return c

def boolean(cls,a,b):
    aa=TopTools_ListOfShape(); aa.Append(a); bb=TopTools_ListOfShape()
    for s in (b if isinstance(b,list) else [b]): bb.Append(s)
    op=cls(); op.SetArguments(aa); op.SetTools(bb); op.SetNonDestructive(True)
    op.SetFuzzyValue(1e-7); op.SetRunParallel(False); op.Build()
    if not op.IsDone(): raise RuntimeError(f'{cls}: failed')
    s=op.Shape()
    if not BRepCheck_Analyzer(s).IsValid(): raise RuntimeError(f'{cls}: invalid result')
    return s

def read(path):
    s=TopoDS_Shape(); BRepTools.Read_s(s,str(path),BRep_Builder()); return s

def cached(name,fn):
    path=ROOT/(name+'.brep'); path.parent.mkdir(parents=True,exist_ok=True)
    # Rebuild by default so changed geometry cannot silently reuse stale files.
    # --reuse-cache is only for resuming the identical configuration.
    if '--reuse-cache' in sys.argv and path.exists(): return read(path)
    s=fn(); BRepTools.Write_s(s,str(path)); print('Built',name,'solids',len(solids(s)),flush=True); return s

def box(x0,x1,y0,y1,z0=1200.,z1=1340.):
    return BRepPrimAPI_MakeBox(gp_Pnt(x0,y0,z0),x1-x0,y1-y0,z1-z0).Shape()

def cylinder(x,y,r):
    return BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x,y,1200),gp_Dir(0,0,1)),r,150).Shape()

def shell(ri,ro):
    return boolean(BRepAlgoAPI_Cut,BRepPrimAPI_MakeSphere(gp_Pnt(0,0,C),ro).Shape(),BRepPrimAPI_MakeSphere(gp_Pnt(0,0,C),ri).Shape())

def spline(points,closed=False):
    arr=TColgp_HArray1OfPnt(1,len(points))
    for i,p in enumerate(points): arr.SetValue(i+1,gp_Pnt(*map(float,p)))
    fit=GeomAPI_Interpolate(arr,closed,1e-7); fit.Perform()
    if not fit.IsDone(): raise RuntimeError('Spline interpolation failed')
    return fit.Curve()

def pipe(points,r=.7):
    crv=spline(np.array(points,float)); edge=BRepBuilderAPI_MakeEdge(crv).Edge()
    spine=BRepBuilderAPI_MakeWire(edge).Wire()
    p=gp_Pnt(); d=gp_Vec(); crv.D1(crv.FirstParameter(),p,d)
    circle=gp_Circ(gp_Ax2(p,gp_Dir(d)),r)
    profile=BRepBuilderAPI_MakeFace(BRepBuilderAPI_MakeWire(BRepBuilderAPI_MakeEdge(circle).Edge()).Wire()).Face()
    op=BRepOffsetAPI_MakePipe(spine,profile); op.Build()
    if not op.IsDone(): raise RuntimeError('End petal pipe failed')
    return op.Shape()

def export(s,name,region=None,step=False):
    path=ROOT/name; path.parent.mkdir(parents=True,exist_ok=True)
    BRepTools.Write_s(s,str(path.with_suffix('.brep')))
    bodies=tessellate_shape(s,name,tolerance=P['mesh_linear_deflection'])
    mesh=trimesh.util.concatenate([b.mesh for b in bodies]); mesh.export(path.with_suffix('.stl'))
    arrays=dict(vertices=mesh.vertices,faces=mesh.faces)
    if region is not None: arrays['region']=np.full(len(mesh.faces),region,np.int8)
    np.savez_compressed(str(path)+'_mesh.npz',**arrays)
    stats=dict(native_valid=bool(BRepCheck_Analyzer(s).IsValid()),solid_count=len(solids(s)),
               mesh_watertight=bool(mesh.is_watertight),winding_consistent=bool(mesh.is_winding_consistent),
               mesh_volume_mm3=float(mesh.volume),vertices=len(mesh.vertices),triangles=len(mesh.faces),bounds_mm=mesh.bounds.tolist())
    if step:
        # Rounded node primitives are already split into nonperiodic sectors
        # before Boolean construction. Transfer directly; splitting multiply
        # trimmed node faces after fusion can change their trimming regions.
        from OCP.BRepGProp import BRepGProp
        from OCP.GProp import GProp_GProps
        export_shape=s
        assert BRepCheck_Analyzer(export_shape).IsValid()
        assert len(solids(export_shape))==len(solids(s))
        writer=STEPControl_Writer(); writer.Transfer(export_shape,STEPControl_AsIs)
        assert writer.Write(str(path.with_suffix('.step')))==1
        rd=STEPControl_Reader(); assert rd.ReadFile(str(path.with_suffix('.step')))==1
        rd.TransferRoots(); imported=rd.OneShape()
        stats.update(step_roundtrip_valid=bool(BRepCheck_Analyzer(imported).IsValid()),step_roundtrip_solids=len(solids(imported)))
        original_prop=GProp_GProps(); BRepGProp.VolumeProperties_s(s,original_prop)
        step_prop=GProp_GProps(); BRepGProp.VolumeProperties_s(imported,step_prop)
        fresh=trimesh.util.concatenate([b.mesh for b in tessellate_shape(imported,'STEP roundtrip',tolerance=.025)])
        difference=abs(float(mesh.volume)-float(fresh.volume))/abs(float(fresh.volume))
        stats.update(native_volume_mm3=original_prop.Mass(),step_native_volume_mm3=step_prop.Mass(),
                     step_mesh_volume_mm3=float(fresh.volume),step_mesh_watertight=bool(fresh.is_watertight),
                     step_mesh_winding_consistent=bool(fresh.is_winding_consistent),
                     step_stl_relative_volume_difference=difference,
                     step_export_preparation='Rounded sphere node primitives have four nonperiodic sectors before Boolean construction; direct STEP transfer without after-fusion face splitting')
        assert original_prop.Mass()>0 and step_prop.Mass()>0 and fresh.volume>0
        assert fresh.is_watertight and fresh.is_winding_consistent and difference<.003
    print('Exported',name,stats,flush=True)
    return mesh,stats


"""Retain the source slot; cut sixteen rounded waves into EACH flank."""
from pathlib import Path
import sys,json,time
import numpy as np

OUT=Path(__file__).resolve().parent
OLD=OUT.parent/'waist_wave_teeth_20261112'
sys.path.insert(0,str(OUT.parent/'waist_sla70a_serrated_20261112'))
import cad_kernel as k
from OCP.Geom import Geom_BezierCurve
from OCP.TColgp import TColgp_Array1OfPnt
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
k.ROOT=OUT;k.P['mesh_linear_deflection']=.015

PARAMS=dict(depth_mm=20.,pitch_mm=6.5,waves_per_side=16,total_waves=32,active_y_range_mm=[-52,52],
            original_straight_slot_width_mm=15.,root_abs_x_mm=7.5,valley_abs_x_mm=27.5,
            side='both, mirrored',operation='remove material only; retain full original voids',
            profile='Two cubic Bezier segments per period, tangent and curvature continuous',
            original_coordinates_preserved=True)

def volume(s):
    g=GProp_GProps();BRepGProp.VolumeProperties_s(s,g);return g.Mass()

def bezier(points):
    a=TColgp_Array1OfPnt(1,len(points))
    for i,p in enumerate(points):a.SetValue(i+1,k.gp_Pnt(*p))
    return k.BRepBuilderAPI_MakeEdge(Geom_BezierCurve(a)).Edge()

def tooth(a,side):
    h=PARAMS['pitch_mm']/2;z=1200.;x0=side*7.5;x1=side*27.5
    e1=bezier([(x0,a,z),(x0,a+h/3,z),(x1,a+2*h/3,z),(x1,a+h,z)])
    e2=bezier([(x1,a+h,z),(x1,a+4*h/3,z),(x0,a+5*h/3,z),(x0,a+2*h,z)])
    w=k.BRepBuilderAPI_MakeWire();w.Add(e1);w.Add(e2)
    points=[(x0,a+2*h,z),(0.,a+2*h,z),(0.,a,z),(x0,a,z)]
    for p,q in zip(points[:-1],points[1:]):w.Add(k.BRepBuilderAPI_MakeEdge(k.gp_Pnt(*p),k.gp_Pnt(*q)).Edge())
    assert w.IsDone()
    s=k.BRepPrimAPI_MakePrism(k.BRepBuilderAPI_MakeFace(w.Wire()).Face(),k.gp_Vec(0,0,120)).Shape()
    assert k.BRepCheck_Analyzer(s).IsValid()
    return s

def main():
    base=k.read(OLD/'source.brep')
    tools=[tooth(-52+6.5*i,side) for side in [-1,1] for i in range(16)]
    for i,t in enumerate(tools):k.BRepTools.Write_s(t,str(OUT/f'tool_{i}.brep'))
    t=time.time()
    result=k.boolean(k.BRepAlgoAPI_Cut,base,tools)
    assert len(k.solids(result))==1
    print('native result valid',volume(result),'seconds',time.time()-t,flush=True)
    mesh,stats=k.export(result,'bilateral_dense_wave_20mm',step=True)
    stats.pop('step_export_preparation',None)
    # Binary STL float precision is improved by translating Z close to zero.
    mesh.apply_translation([0,0,-mesh.bounds[0,2]])
    mesh.export(OUT/'bilateral_dense_wave_20mm_print.stl')
    check=k.trimesh.load_mesh(OUT/'bilateral_dense_wave_20mm_print.stl')
    assert check.is_watertight and check.is_winding_consistent and len(check.split())==1
    PARAMS['stats']=stats
    (OUT/'parameters.json').write_text(json.dumps(PARAMS,indent=2),encoding='utf8')
    print('DONE',flush=True)

if __name__=='__main__':main()

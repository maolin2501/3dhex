from pathlib import Path
import sys,json,hashlib
import numpy as np

OUT=Path(__file__).resolve().parent
OLD=OUT.parent/'waist_wave_teeth_20261112'
sys.path.insert(0,str(OUT.parent/'waist_sla70a_serrated_20261112'))
import cad_kernel as k
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.TopAbs import TopAbs_IN,TopAbs_ON

def volume(s):
    p=GProp_GProps();BRepGProp.VolumeProperties_s(s,p);return p.Mass()

def inside(s,x,y,z):
    return BRepClass3d_SolidClassifier(s,k.gp_Pnt(float(x),float(y),float(z)),1e-6).State()==TopAbs_IN

def main():
    rd=k.STEPControl_Reader();assert rd.ReadFile(str(OUT/'bilateral_dense_wave_20mm.step'))==1
    assert rd.TransferRoots()>0;s=rd.OneShape();base=k.read(OLD/'source.brep')
    assert k.BRepCheck_Analyzer(s).IsValid() and len(k.solids(s))==1
    stats=dict(step_valid=True,solid_count=1)
    extra=k.boolean(k.BRepAlgoAPI_Cut,s,base)
    stats['material_added_to_original_mm3']=volume(extra)
    assert abs(stats['material_added_to_original_mm3'])<.01
    print('PASS whole-model subset: original voids preserved',flush=True)
    # Compare the actual outer frame outside both 20 mm tooth bands.
    frame_base=k.boolean(k.BRepAlgoAPI_Cut,base,k.box(-27.501,27.501,-100,100))
    frame_final=k.boolean(k.BRepAlgoAPI_Cut,s,k.box(-27.501,27.501,-100,100))
    loss=k.boolean(k.BRepAlgoAPI_Cut,frame_base,frame_final)
    stats['removed_from_outer_frame_mm3']=volume(loss)
    assert abs(stats['removed_from_outer_frame_mm3'])<.01
    # Measure the actual STEP boundary by bisection, independent of tool bounds.
    def boundary(y,side):
        lo=5.;hi=32.;z=1250.
        assert not inside(s,side*lo,y,z) and inside(s,side*hi,y,z)
        for _ in range(35):
            mid=(lo+hi)/2
            if inside(s,side*mid,y,z):hi=mid
            else:lo=mid
        return side*(lo+hi)/2
    stats['measured_sides']={}
    for side,label in [(-1,'left'),(1,'right')]:
        valleys=[boundary(y,side) for y in [-29.25,-16.25,-3.25,3.25,16.25,29.25]]
        tips=[boundary(y,side) for y in [-26.,-13.,-6.5,6.5,13.,26.]]
        depth=float(side*(np.mean(valleys)-np.mean(tips)))
        stats['measured_sides'][label]=dict(valley_x_mm=valleys,tip_x_mm=tips,depth_mm=depth)
        assert abs(depth-20)<.005
        assert all(abs(x-side*7.5)<.005 for x in tips)
        assert all(abs(x-side*27.5)<.005 for x in valleys)
    for y in np.linspace(-40,40,41):
        for x in [-7.49,0.,7.49]:assert not inside(s,x,y,1250)
    stats['clear_straight_channel_width_mm']=15.
    stats['clear_channel_checks']=123
    for x in np.linspace(8,60,20):assert not inside(s,x,0,1270)
    stats['right_radial_slit_preserved']=True
    m=k.trimesh.load_mesh(OUT/'bilateral_dense_wave_20mm_print.stl')
    assert m.is_watertight and m.is_winding_consistent and len(m.split())==1
    stats['printing_stl']=dict(watertight=True,winding_consistent=True,connected_components=1,
                              z_min_mm=float(m.bounds[0,2]),triangles=len(m.faces))
    source=json.loads((OLD/'inspection.json').read_text(encoding='utf8'))
    original=Path(source['file'])
    stats['original_import_sha256']=source['sha256']
    stats['cached_source_brep_sha256']=hashlib.sha256((OLD/'source.brep').read_bytes()).hexdigest()
    if original.exists():
        assert hashlib.sha256(original.read_bytes()).hexdigest()==source['sha256']
        stats['external_source_hash_reverified']=True
    else:
        stats['external_source_hash_reverified']=False
        stats['source_note']='Original temporary attachment path expired; geometric comparison uses the preserved original BREP import.'
    (OUT/'verification.json').write_text(json.dumps(stats,indent=2),encoding='utf8')
    print(json.dumps(stats,indent=2),flush=True)

if __name__=='__main__':main()

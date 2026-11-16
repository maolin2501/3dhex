"""User-confirmed revision: open tooth walls, 6 mm cells and 1 mm struts."""
from pathlib import Path
import json,shutil,hashlib
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent
PREVIOUS=ROOT/'waist_bilateral_lattice_ribs_20261113'
doc=json.loads((PREVIOUS/'design.json').read_text(encoding='utf8'))
old_faces=doc['preserved_face_ids']
assert len(old_faces)==64 and doc['rib_count']==35
doc.update(wave_face_ids=old_faces,preserved_face_ids=[],preserved_face_count=0,
           preserved_thickness_mm=0.,revision_of=str(PREVIOUS),
           note='User confirmed 6 mm cells and 1.0 mm struts, with no solid tooth-sidewall preservation; existing 0.8 mm ribs and original gaps retained.')
doc['config'].update(cell=6.,wall=1.,layer=6.,shell=0.,preserve_faces=[],preserve_thicknesses={},
                     surface_cell=0.,surface_width=0.)
shutil.copytree(PREVIOUS/'model',OUT/'model',dirs_exist_ok=True)
shutil.copy2(PREVIOUS/'rib_midplanes.npz',OUT/'rib_midplanes.npz')
doc['source_mesh_sha256']=hashlib.sha256((OUT/'model/body_0.npz').read_bytes()).hexdigest()
(OUT/'design.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2),encoding='utf8')
print('PREPARED: 64 tooth faces opened, cell 6 mm, strut 1 mm, 35 ribs retained',flush=True)

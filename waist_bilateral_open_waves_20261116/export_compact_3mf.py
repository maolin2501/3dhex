"""Stream an indexed 3MF with explicit millimeter units and ZIP64 support.

Core XML/package layout follows the installed trimesh 3MF exporter. Unlike
that bytes-returning API, this writes directly to disk in bounded batches.
"""
from pathlib import Path
import json,time,zipfile,hashlib,xml.parsers.expat
import numpy as np
OUT=Path(__file__).resolve().parent;started=time.time()
ply=OUT/'open_wave_lattice_indexed.ply'
with ply.open('rb') as f:
    while True:
        line=f.readline().decode('ascii').strip()
        if line.startswith('element vertex'):nv=int(line.split()[-1])
        if line.startswith('element face'):nf=int(line.split()[-1])
        if line=='end_header':break
    offset=f.tell()
vertices=np.memmap(ply,dtype='<f4',offset=offset,shape=(nv,3),mode='r')
faces=np.memmap(ply,dtype=[('count','u1'),('v','<i4',(3,))],offset=offset+12*nv,shape=(nf,),mode='r')
archive=OUT/'open_wave_lattice.3mf';digest=hashlib.sha256()
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as z:
    z.writestr('[Content_Types].xml','<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>')
    z.writestr('_rels/.rels','<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')
    with z.open('3D/3dmodel.model','w',force_zip64=True) as f:
        def write(text):
            data=text.encode('ascii');f.write(data);digest.update(data)
        write('<?xml version="1.0" encoding="UTF-8"?><model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"><resources><object id="1" type="model" name="Open wave lattice 6mm cell 1mm strut"><mesh><vertices>\n')
        for i in range(0,nv,8192):
            batch=vertices[i:i+8192]
            # Nine significant digits preserve every finite float32 coordinate.
            fragment=('<vertex x="{:.9g}" y="{:.9g}" z="{:.9g}"/>\n'*len(batch)).format(*batch.ravel())
            write(fragment)
            if i%1048576==0:print('VERTICES',i,'/',nv,flush=True)
        write('</vertices><triangles>\n')
        for i in range(0,nf,16384):
            batch=faces['v'][i:i+16384]
            write(('<triangle v1="{}" v2="{}" v3="{}"/>\n'*len(batch)).format(*batch.ravel()))
            if i%2097152==0:print('TRIANGLES',i,'/',nf,flush=True)
        write('</triangles></mesh></object></resources><build><item objectid="1"/></build></model>')
print('EXPORTED',archive.stat().st_size,flush=True)
# Full decompression checks CRC, payload hash and all geometry record counts.
check=hashlib.sha256();vertex_count=triangle_count=0;tail=b''
parser=xml.parsers.expat.ParserCreate()
with zipfile.ZipFile(archive) as z:
    assert set(z.namelist())=={'[Content_Types].xml','_rels/.rels','3D/3dmodel.model'}
    with z.open('3D/3dmodel.model') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):
            check.update(chunk);parser.Parse(chunk,False);data=tail+chunk;lines=data.split(b'\n');tail=lines.pop()
            block=b'\n'.join(lines)
            vertex_count+=block.count(b'<vertex ');triangle_count+=block.count(b'<triangle ')
    assert tail.endswith(b'</build></model>')
parser.Parse(b'',True)
assert vertex_count==nv and triangle_count==nf,(vertex_count,triangle_count,nv,nf)
assert check.hexdigest()==digest.hexdigest()
report=dict(format='3MF Core',units='millimeter',vertices=nv,triangles=nf,bytes=archive.stat().st_size,
            indexed_topology_preserved=True,coordinate_precision='9 significant decimal digits (lossless float32 roundtrip)',
            crc_and_payload_sha256_verified=True,xml_well_formed=True,model_xml_sha256=check.hexdigest(),elapsed_seconds=time.time()-started)
(OUT/'3mf_verification.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print('3MF VERIFIED',json.dumps(report),flush=True)

from pathlib import Path
import sys
import numpy as np
from PIL import Image,ImageDraw

OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parent/'waist_tpu70a_20261112'))
import render_source as rs
r=rs.r

def panel(mesh,title,subtitle,direction,up,size=(1160,920)):
    im=Image.new('RGB',size,r.WHITE);d=ImageDraw.Draw(im)
    d.text((32,20),title,font=r.font(36,True),fill=r.INK)
    d.text((34,76),subtitle,font=r.font(23),fill=r.MUTED)
    im.paste(r.render([(mesh,r.FIXED)],(1100,780),direction=direction,up=up),(30,123))
    return im

def main():
    mesh=r.read_stl(OUT/'bilateral_dense_wave_20mm.stl')
    top=panel(mesh,'双边对称波浪齿 · 直通缝保留','每侧 16 个波谷 · 齿距 6.5 mm · 齿深 20 mm',(0,0,1),(0,1,0))
    iso=panel(mesh,'立体视图','实际 STEP 几何 · 中央直通缝及右侧细缝保留',(.75,-1,.8),(0,0,1))
    # Independent drawing of the actual cubic profile, with dimension labels.
    drawing=Image.new('RGB',(2320,540),r.WHITE);d=ImageDraw.Draw(drawing)
    d.text((40,20),'双边齿形尺寸（单位 mm）',font=r.font(34,True),fill=r.INK)
    scale=7.;ox=390;oy=290
    def point(y,x):return (ox+(y+52)*scale,oy-x*scale)
    for side,color in [(-1,(193,104,48)),(1,r.FIXED)]:
        curve=[]
        for a in np.arange(-52,52,6.5):
            for t in np.linspace(0,1,41):
                x=side*(7.5+20*(3*t*t-2*t*t*t));curve.append(point(a+3.25*t,x))
            for t in np.linspace(0,1,41):
                x=side*(27.5-20*(3*t*t-2*t*t*t));curve.append(point(a+3.25+3.25*t,x))
        d.line(curve,fill=color,width=4)
        d.line([point(-52,side*7.5),point(52,side*7.5)],fill=(162,185,177),width=2)
    d.text((1260,155),'双边对称：每侧 16 个波谷，共 32 个',font=r.font(28),fill=r.INK)
    d.text((1260,225),'齿距：6.5 mm，单侧密度再加倍',font=r.font(28),fill=r.INK)
    d.text((1260,295),'两侧齿深：各 20 mm（齿尖至齿谷）',font=r.font(28),fill=r.INK)
    d.text((1260,365),'原竖直通缝：15 mm，完整保留',font=r.font(28),fill=r.INK)
    d.text((1260,435),'原有右侧 0.5 mm 细缝继续保留',font=r.font(26),fill=r.MUTED)
    for x,label,y0,y1 in [(330,'15',237.5,342.5),(280,'20',97.5,237.5),(280,'20',342.5,482.5)]:
        d.line((x,y0,x,y1),fill=r.INK,width=2)
        for yy in [y0,y1]:d.line((x-8,yy,x+8,yy),fill=r.INK,width=2)
        d.text((x-40,(y0+y1)/2),label,font=r.font(25),fill=r.INK,anchor='mm')
    result=Image.new('RGB',(2380,1580),r.PAPER)
    d=ImageDraw.Draw(result)
    d.text((36,24),'中央长槽 · 双边高密度波浪锯齿',font=r.font(48,True),fill=r.INK)
    d.text((40,90),'双侧齿距 6.5 mm · 齿深 20 mm · 中央原有 15 mm 直通缝保留',font=r.font(27),fill=r.MUTED)
    result.paste(top,(20,140));result.paste(iso,(1200,140));result.paste(drawing,(30,1040))
    result.save(OUT/'preview.png');top.save(OUT/'top_view.png');iso.save(OUT/'isometric_view.png')

if __name__=='__main__':main()

"""Bounded spatial hash for unintended straight-member centreline crossings."""
import itertools
import numpy as np


def segment_distance(a,b,c,d):
    """Closest points on two finite segments, including parallel segments."""
    u=b-a;v=d-c;w=a-c
    aa=np.dot(u,u);bb=np.dot(u,v);cc=np.dot(v,v)
    dd=np.dot(u,w);ee=np.dot(v,w);den=aa*cc-bb*bb
    if min(aa,cc)<1e-16:raise ValueError('Zero length member')
    s=np.clip((bb*ee-cc*dd)/den,0.,1.) if den>1e-12*aa*cc else 0.
    t=(bb*s+ee)/cc
    if t<0:s=np.clip(-dd/aa,0.,1.);t=0.
    elif t>1:s=np.clip((bb-dd)/aa,0.,1.);t=1.
    p=a+s*u;q=c+t*v
    return float(np.linalg.norm(p-q)),p,q


class SegmentGuard:
    def __init__(self,cell,diameter):
        self.cell=cell;self.diameter=diameter
        self.edges=[];self.bins={};self.rejections=0

    def _keys(self,a,b):
        lo=np.floor((np.minimum(a,b)-self.diameter)/self.cell).astype(int)
        hi=np.floor((np.maximum(a,b)+self.diameter)/self.cell).astype(int)
        return itertools.product(*(range(x,y+1) for x,y in zip(lo,hi)))

    def add(self,a,b):
        i=len(self.edges);self.edges.append((a.copy(),b.copy()))
        for key in self._keys(a,b):self.bins.setdefault(key,[]).append(i)

    def clear(self,a,b):
        nearby=set()
        for key in self._keys(a,b):nearby.update(self.bins.get(key,()))
        for index in sorted(nearby):
            c,d=self.edges[index]
            ends=np.linalg.norm(np.array([a,b])[:,None]-np.array([c,d])[None],axis=2)
            # Common BCC nodes and merged sheet contacts are intended joints.
            if ends.min()<self.diameter:continue
            distance,p,q=segment_distance(a,b,c,d)
            if distance<self.diameter*.8:
                self.rejections+=1
                return False
        return True

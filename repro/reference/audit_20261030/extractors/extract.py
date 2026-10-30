# Independent extraction of curve polylines from a matplotlib PDF.
import re, zlib, sys, json
def streams(data):
    out=[]
    for m in re.finditer(rb'stream\r?\n', data):
        s=m.end(); e=data.find(b'endstream', s)
        raw=data[s:e]
        try: out.append(zlib.decompress(raw))
        except Exception:
            try: out.append(zlib.decompressobj().decompress(raw))
            except Exception: pass
    return out
def paths(content):
    toks=content.split()
    stack=[]; cur=[]; res=[]; color=None; lw=None
    for t in toks:
        try:
            stack.append(float(t)); continue
        except ValueError: pass
        if t==b'm': cur=[(stack[-2],stack[-1])]
        elif t==b'l': cur.append((stack[-2],stack[-1]))
        elif t==b'RG': color=tuple(stack[-3:])
        elif t==b'w': lw=stack[-1]
        elif t==b're': res.append(('re',stack[-4:],color,lw))
        elif t in (b'S',b'n',b'f',b'B'):
            if cur: res.append(('path',cur,color,lw,t.decode()))
            cur=[]
        stack=[] if t not in (b'm',b'l') else []
    return res
data=open(sys.argv[1],'rb').read()
allp=[]
for c in streams(data):
    if b' l' in c or b' m' in c:
        allp+=paths(c)
rects=[p for p in allp if p[0]=='re']
long=[p for p in allp if p[0]=='path' and len(p[1])>50]
print('rects',[ (r[1]) for r in rects][:5])
res={}
for p in long:
    xs=[q[0] for q in p[1]]; 
    print('color',p[2],'lw',p[3],'npts',len(p[1]),'x0',xs[0],'x1',xs[-1])

ax=rects[0][1]; X0,Y0,W,H=ax
xl=(-180,3780); yl=(float(sys.argv[2]) if len(sys.argv)>2 else 30, float(sys.argv[3]) if len(sys.argv)>3 else 34)
def tx(x): return xl[0]+(x-X0)/W*(xl[1]-xl[0])
def ty(y): return yl[0]+(y-Y0)/H*(yl[1]-yl[0])
names={ (0.1215686275, 0.4666666667, 0.7058823529):'gk0',(1.0, 0.4980392157, 0.0549019608):'gk1',(0.1725490196, 0.6274509804, 0.1725490196):'gk2',(0.8392156863, 0.1529411765, 0.1568627451):'gk3'}
out={}
for p in long:
    nm=names.get(p[2],str(p[2]))
    pts=[(tx(x),ty(y)) for x,y in p[1]]
    out[nm]=pts
    def at(t):
        for (a,b),(c,d) in zip(pts,pts[1:]):
            if a<=t<=c: return b+(d-b)*(t-a)/(c-a) if c>a else d
        return None
    def cross(th):
        for (a,b),(c,d) in zip(pts,pts[1:]):
            if b<th<=d: return a+(c-a)*(th-b)/(d-b)
            if b>=th and a==pts[0][0]: return a
        return None
    print(nm,'final %.4f'%pts[-1][1],'t@32%%=%s'%(None if cross(32) is None else '%.1f'%cross(32)),'v@1000=%.3f'%at(1000),'v@100=%.3f'%at(100), 'v@500=%.3f'%at(500))
json.dump(out,open(sys.argv[1]+'.curves.json','w'))

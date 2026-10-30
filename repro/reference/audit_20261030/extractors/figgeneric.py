"""Generic extractor for matplotlib comparison_3x3 PDFs: python3 figgeneric.py file.pdf out.json"""
import re, zlib, sys, json
pdf=sys.argv[1]
data=open(pdf,'rb').read()
streams=[]
for num,body in re.findall(rb'(\d+) 0 obj(.*?)endobj',data,re.S):
    m=re.search(rb'stream\r?\n(.*?)\r?\nendstream',body,re.S)
    if m and b'/Subtype' not in body[:300]:
        try: streams.append(zlib.decompress(m.group(1)).decode('latin1'))
        except Exception: pass
s=max(streams,key=len)
toks=re.findall(r'\[[^\]]*\]|\([^)]*\)|/[A-Za-z0-9_.+-]+|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|[A-Za-z*\'"]+',s)
def mul(m1,m2):
    a,b,c,d,e,f=m1; A,B,C,D,E,F=m2
    return (a*A+b*C,a*B+b*D,c*A+d*C,c*B+d*D,e*A+f*C+E,e*B+f*D+F)
ctm=(1,0,0,1,0,0); st=[]; clip=None; gs=None; cur=[]; ops=[]
markers=[]; texts=[]; paths=[]; glyphs=[]; dashes=[]; dash=None; stroke=None
for t in toks:
    if re.match(r'^[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?$',t) or t[0] in '/[(':
        ops.append(t); continue
    if t=='q': st.append((ctm,clip,gs,dash,stroke))
    elif t=='Q': ctm,clip,gs,dash,stroke=st.pop()
    elif t=='cm': ctm=mul(tuple(float(x) for x in ops[-6:]),ctm)
    elif t=='re': cur=[('re',tuple(float(x) for x in ops[-4:]))]
    elif t=='W':
        if cur and cur[0][0]=='re': clip=cur[0][1]
    elif t=='d': dash=ops[-2] if len(ops)>=2 else None
    elif t=='RG': stroke=tuple(float(x) for x in ops[-3:])
    elif t=='G': stroke=(float(ops[-1]),)*3
    elif t=='m': cur=[('m',float(ops[-2])*ctm[0]+ctm[4],float(ops[-1])*ctm[3]+ctm[5])]
    elif t=='l': cur.append(('l',float(ops[-2])*ctm[0]+ctm[4],float(ops[-1])*ctm[3]+ctm[5]))
    elif t in ('S','B','f','n'):
        if cur and cur[0][0]=='m': paths.append((clip,cur,t,dash,stroke))
        cur=[]
    elif t=='gs': gs=ops[-1]
    elif t=='Do':
        nm=ops[-1]
        if clip and nm.startswith('/M'): markers.append((nm,clip,ctm[4],ctm[5],gs))
        else: glyphs.append((nm,ctm[4],ctm[5]))
    elif t=='TJ':
        texts.append((ctm[4],ctm[5],''.join(re.findall(r'\(([^)]*)\)',ops[-1]))))
    ops=[]
boxes=sorted(set(m[1] for m in markers),key=lambda b:(-b[1],b[0]))
rows=sorted(set(round(b[1],3) for b in boxes),reverse=True)
cols=sorted(set(round(b[0],3) for b in boxes))
def axis(b):
    x0,y0,w,h=b
    ty=sorted(set(round(p[1][0][2],4) for p in paths if len(p[1])==2 and abs(p[1][0][1]-x0)<1e-2 and abs(p[1][1][1]-(x0-3.5))<1e-2 and abs(p[1][0][2]-p[1][1][2])<1e-6 and y0-1<=p[1][0][2]<=y0+h+1))
    lab=sorted([(y,t,x) for x,y,t in texts if x0-90<x<x0-1 and y0-20<y<y0+h and re.match(r'^[0-9.]+$',t)])
    vals=[]
    for y,t,x in lab:
        neg=any(g[0].endswith('minus') and abs(g[2]-y)<3 and x-25<g[1]<x+1 for g in glyphs)
        vals.append((-1 if neg else 1)*float(t))
    return ty,vals
maps={}
for b in boxes:
    ty,vals=axis(b)
    if vals and len(vals)==len(ty): maps[b]=(ty,vals)
for b in boxes:
    if b not in maps:
        ty,_=axis(b)
        cand=[bb for bb in maps if abs(bb[1]-b[1])<1e-3 and maps[bb][0]==ty]
        if cand: maps[b]=maps[min(cand,key=lambda bb:abs(bb[0]-b[0]))]
out={}
series={}
for nm,clip,x,y,g in markers: series.setdefault(nm,[]).append((clip,x,y,g))
colors={}
for nm,v in series.items():
    b=v[0][0]; ty,vals=maps.get(b,(None,None))
    if ty is None: print('nomap',nm); continue
    (ya,va),(yb,vb)=(ty[0],vals[0]),(ty[-1],vals[-1])
    ys=[va+(y-ya)*(vb-va)/(yb-ya) for _,x,y,_ in v]
    key=f"r{rows.index(round(b[1],3))}c{cols.index(round(b[0],3))}|{v[0][3]}|{nm}"
    out[key]=ys
    print(key,len(ys),round(sum(ys)/len(ys),5))
# dashed/horizontal lines inside axes
hl=[]
for clip,p,op,d,stc in paths:
    if clip and len(p)==2 and abs(p[0][2]-p[1][2])<1e-6 and d and d!='[]':
        b=clip; ty,vals=maps.get(tuple(b),(None,None))
        if ty: hl.append((f"r{rows.index(round(b[1],3))}c{cols.index(round(b[0],3))}",d,stc,round(vals[0]+(p[0][2]-ty[0])*(vals[-1]-vals[0])/(ty[-1]-ty[0]),5)))
for h in hl: print('dashed',h)
json.dump(out,open(sys.argv[2],'w'))
for x,y,t in texts:
    if not re.match(r'^[0-9.]+$',t): pass
print('titles',[t for x,y,t in texts if not re.match(r'^[0-9.]+$',t)][:40])

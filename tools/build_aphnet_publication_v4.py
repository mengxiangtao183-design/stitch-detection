"""Code-audited M6 publication figures, native editable Visio and SVG.

Geometry is shared between SVG and VSDX exports. No bitmap shapes are used.
Only the selected M6 architecture is illustrated; unused centerline outputs
and topology/boundary auxiliary losses are intentionally omitted.
"""
from pathlib import Path
from html import escape
import json
import math
import zipfile
import xml.etree.ElementTree as ET

OUT = Path(__file__).resolve().parents[1] / 'outputs' / 'aphnet_publication_v4'
OUT.mkdir(parents=True, exist_ok=True)
COL = {'ink':'#263445','enc':'#F8DFCF','fpn':'#F7EDBA','edge':'#F5D6DF',
       'loc':'#E5DDF2','hr':'#DDEEDB','prior':'#D8EAF6','white':'#FFFFFF',
       'muted':'#697785','line':'#526477','accent':'#377DAB'}

class Scene:
    def __init__(self, name, w, h):
        self.name, self.w, self.h, self.items = name, w, h, []
        self.nodes = {}
    def node(self, key, x,y,w,h,label,fill='white',kind='box',fs=17,stroke=None,dash=False):
        item=dict(kind=kind,id=key,x=x,y=y,w=w,h=h,text=label,
                  fill=COL.get(fill,fill),stroke=stroke or COL['line'],fs=fs,dash=dash)
        self.items.append(item); self.nodes[key]=item
        return key
    def label(self,x,y,text,fs=17,w=500,align='left',color=None):
        self.items.append(dict(kind='text',id=f'text{len(self.items)}',x=x,y=y,w=w,h=fs*1.4,
                               text=text,fs=fs,align=align,fill='none',stroke=color or COL['ink']))
    def point(self,key,side='r'):
        n=self.nodes[key]; x,y,w,h=n['x'],n['y'],n['w'],n['h']
        return {'l':(x,y+h/2),'r':(x+w,y+h/2),'t':(x+w/2,y),'b':(x+w/2,y+h)}[side]
    def route(self,points,color=None,dash=False,start=None,end=None,arrow=True):
        self.items.append(dict(kind='line',id=f'wire{len(self.items)}',points=points,
                               stroke=color or COL['line'],dash=dash,start=start,end=end,arrow=arrow))
    def link(self,a,b,sa='r',sb='l',via=(),color=None,dash=False):
        self.route([self.point(a,sa),*via,self.point(b,sb)],color,dash,(a,sa),(b,sb))
    def dot(self,x,y,color=None):
        self.node(f'dot{len(self.items)}',x-3,y-3,6,6,'',color or COL['line'],'dot')
    def frame(self,x,y,w,h,title,fill='white'):
        self.node(f'frame{len(self.items)}',x,y,w,h,'',fill,'frame',stroke='#98A5AE',dash=True)
        self.label(x+16,y+9,title,21,w-32)
    def plus(self,key,x,y,symbol='+'):
        return self.node(key,x-17,y-17,34,34,symbol,'white','circle',22)
    def port(self,key,x,y,label,fill='fpn',w=90):
        return self.node(key,x,y,w,46,label,fill,'port',17)
    def paste(self,other,x,y,prefix):
        for original in other.items:
            item=dict(original); item['id']=prefix+item['id']
            if item['kind']=='line':
                item['points']=[(xx+x,yy+y) for xx,yy in item['points']]
                for tag in ('start','end'):
                    if item.get(tag): item[tag]=(prefix+item[tag][0],item[tag][1])
            else: item['x']+=x; item['y']+=y
            self.items.append(item)
            if item['kind'] not in ('line','text'): self.nodes[item['id']]=item

def block(s,k,x,y,w,label,fill='white',h=58,fs=17):
    return s.node(k,x,y,w,h,label,fill,fs=fs)

def overview():
    s=Scene('01_APHNet_overall',2520,910)
    s.frame(8,8,2504,894,'(a) APHNet / M6 — complete feature routes')
    s.label(48,55,'MobileNetV4 Hybrid Medium',20,470)
    s.label(565,55,'Mobile-FPN / lightweight multiscale reconstruction',20,570)
    s.label(1210,55,'Boundary and contextual localization pathways',20,950)
    block(s,'rgb',48,102,120,'RGB\n320×320×3','enc')
    block(s,'stem',225,102,178,'Stem: Conv 3×3 / s2\nBN → ReLU; 32 ch','enc',64,16)
    s.link('rgb','stem')
    ys=[218,363,508,653]
    desc=['Stage 2\nFused inverted block','Stage 3\n2 UIB blocks','Stage 4\n8 UIB + 4 MQA','Stage 5\n12 UIB + 4 MQA\nConv 1×1: 256→960']
    shapes=['80²×48','40²×80','20²×160','10²×960']
    for i,(cy,label,sh) in enumerate(zip(ys,desc,shapes)):
        n=i+2
        block(s,f's{n}',225,cy-34,178,label,'enc',76 if n==5 else 68,16)
        s.node(f'c{n}',439,cy-26,100,52,f'C{n}\n{sh}','enc',kind='tensor',fs=16)
        block(s,f'l{n}',580,cy-30,146,f'L{n}: Conv 1×1\nGN → ReLU\n→192 ch','fpn',60,16)
        block(s,f'ds{n}',822,cy-29,137,'DW 3×3 → PW 1×1\nGN/ReLU after each','hr',58,15)
        s.node(f'p{n}',1002,cy-26,95,52,f'P{n}\n{sh.split("×")[0]}×192','fpn',kind='tensor',fs=16)
        s.link(f's{n}',f'c{n}'); s.link(f'c{n}',f'l{n}')
        if n<5:
            s.plus(f'add{n}',770,cy)
            s.link(f'l{n}',f'add{n}');s.link(f'add{n}',f'ds{n}')
        else:s.link(f'l{n}',f'ds{n}')
        s.link(f'ds{n}',f'p{n}')
        if i:s.link(f's{n-1}',f's{n}','b','t')
    s.link('stem','s2','b','t')
    for n,cy in [(5,653),(4,508),(3,363)]:
        dest=cy-145
        s.link(f'p{n}',f'add{n-1}',via=[(1125,cy),(1125,cy-72),(770,cy-72)],sb='b',color=COL['accent'])
        s.label(885,cy-98,'Bilinear ↑2',15,160)
    # Same-name ports refer to the completed P maps without crossing the sheet.
    s.port('ep3',1220,155,'P3');block(s,'eup',1345,151,115,'Bilinear ↑2')
    s.port('ep2',1220,272,'P2');s.plus('ec',1500,238,'C')
    s.link('ep3','eup');s.link('eup','ec',via=[(1475,180),(1475,221)],sb='t')
    s.link('ep2','ec',via=[(1475,295),(1475,255)],sb='b')
    block(s,'ep',1550,205,170,'Conv 3×3\nGN → ReLU\n384→128','edge',66)
    block(s,'er',1770,205,198,'Edge enhancement (b)\nContext / Sobel / Lap.\nChannel attention + skip','edge',66,16)
    s.port('fe',2015,215,'F_edge','edge',120)
    s.link('ec','ep');s.link('ep','er');s.link('er','fe')
    s.port('lp4',1220,430,'P4');block(s,'seu',1350,418,145,'Semantic enhancement\nSEU (f); 192 ch','loc',68,15)
    block(s,'icg',1545,418,160,'Image-context guidance\nICG (d); 192 ch','loc',68,15)
    s.port('rgb2',1528,300,'RGB','enc',100);block(s,'resize',1680,292,130,'Resize\n20×20','enc',58)
    s.link('rgb2','resize');s.link('resize','icg','b','t',via=[(1745,382),(1625,382)])
    block(s,'up40',1748,418,118,'Bilinear ↑2\n40×40','loc',68,16)
    block(s,'up80',1905,418,118,'Bilinear ↑2\n80×80','loc',68,16)
    s.plus('lpadd',2080,452)
    s.port('lp2',2035,540,'P2');s.link('lp2','lpadd','t','b')
    block(s,'lr',2130,414,180,'Localization refine\nConv 3×3: 192→128\nDW 3×3 → PW 1×1','loc',76,15)
    s.port('fl',2350,429,'F_loc','loc',110)
    for a,b in [('lp4','seu'),('seu','icg'),('icg','up40'),('up40','up80'),('up80','lpadd'),('lpadd','lr'),('lr','fl')]:s.link(a,b)
    s.label(1220,603,'Fusion and full-resolution prediction',20,1100)
    s.port('afe',1220,683,'F_edge','edge',115);s.port('afl',1220,792,'F_loc','loc',115)
    block(s,'afm',1390,715,165,'Attentive fusion\nAFM (c)\n80×80×1 logits','prior',82)
    s.link('afe','afm',via=[(1355,706),(1355,737)],sb='l')
    s.link('afl','afm',via=[(1370,815),(1472,815)],sb='b')
    s.port('z',1600,732,'Z_afm','prior',110);s.link('afm','z')
    block(s,'head',1815,726,440,'AFM-guided HighRes head (e)\nIndependent projections → 5-way concat → 1×1 fusion\nDepthwise refinement → Conv 1×1 → final logits','hr',96,18)
    s.link('z','head')
    for k,x,label,fill in [('he',1825,'F_edge','edge'),('hl',1975,'F_loc','loc'),('hi',2125,'RGB','enc')]:
        s.port(k,x,648,label,fill,115)
        s.route([s.point(k,'b'),(x+57.5,726)],start=(k,'b'),end=('head','t'))
    block(s,'sig',2300,732,154,'Sigmoid → threshold\n320×320 mask','prior',82,16)
    s.link('head','sig')
    s.label(48,821,'C2–C5: stage outputs; P2–P5: completed pyramid features.',17,1100)
    s.label(48,851,'Matching named ports denote the same tensor. ↑: bilinear resize (align_corners=False).',17,1150)
    s.label(1220,859,'Main mask shown; localization and edge auxiliary outputs are detailed in (d), (e).',16,1200)
    return s

def edge():
    s=Scene('02_edge_enhancement',1230,470);s.frame(8,8,1214,454,'(b) Edge enhancement — 80×80×128','edge')
    s.port('x',30,172,'X','edge',65)
    rows=[(95,'DW 3×3\n128→128','PW 1×1\n128→128'),(200,'Grouped Conv 3×3\nSobel X/Y init; 128→256','PW 1×1\n256→32'),(305,'DW 3×3\nLaplacian init; 128→128',None)]
    for i,(y,a,b) in enumerate(rows):
        block(s,f'a{i}',160,y,220,a,'white',58,16)
        block(s,f'n{i}',410,y,100,'GN → ReLU','white',58,15)
        s.link('x',f'a{i}',via=[(120,195),(120,y+29)])
        s.link(f'a{i}',f'n{i}')
        if b:
            block(s,f'b{i}',545,y,165,b+'\nGN → ReLU','white',66,16);s.link(f'n{i}',f'b{i}')
        end=f'b{i}' if b else f'n{i}'
        s.route([s.point(end,'r'),(755,y+29),(795,y+29)],end=('cat','l'))
    s.node('cat',795,95,42,277,'C\no\nn\nc\na\nt','white',fs=17)
    block(s,'mix',875,171,120,'PW 1×1\n288→128\nGN → ReLU','white',78,16)
    block(s,'eca',1025,177,150,'ECA (f) → GN','white',66,16)
    s.link('cat','mix');s.link('mix','eca')
    s.plus('sum',1100,373);s.link('eca','sum','b','t')
    s.link('x','sum','b','l',via=[(62,373)])
    block(s,'act',905,397,110,'ReLU','white',42);s.link('sum','act','b','r',via=[(1100,418)])
    s.label(160,414,'Sobel / Laplacian kernels are trainable after initialization.',16,720)
    return s

def afm():
    s=Scene('03_AFM',1230,470);s.frame(8,8,1214,454,'(c) Attentive fusion — localization-guided coarse logits','prior')
    s.port('fe',35,89,'F_edge','edge',108);block(s,'pe',178,82,160,'Conv 1×1\n128→128; GN → ReLU','edge',62,15)
    s.port('e',375,90,'E','edge',68);s.plus('mul',900,113,'×');s.port('eg',959,90,'E_g','edge',83)
    for a,b in [('fe','pe'),('pe','e'),('e','mul'),('mul','eg')]:s.link(a,b,color='#B55F79')
    s.port('fl',35,212,'F_loc','loc',108);block(s,'pl',178,205,160,'Conv 1×1\n128→128; GN → ReLU','loc',62,15)
    s.port('l',375,213,'L','loc',68)
    block(s,'cg',488,205,155,'Conv 3×3\n128→64; GN → ReLU','loc',62,15)
    block(s,'cg1',680,211,123,'Conv 1×1\n64→1','loc',50)
    block(s,'sg',840,211,120,'Sigmoid → A','loc',50,16)
    for a,b in [('fl','pl'),('pl','l'),('l','cg'),('cg','cg1'),('cg1','sg')]:s.link(a,b,color='#81669F')
    s.link('sg','mul','t','b',color='#81669F')
    s.plus('cat',540,368,'C')
    s.link('l','cat','b','l',via=[(409,368)],color='#81669F')
    s.link('eg','cat','r','t',via=[(1150,113),(1150,306),(540,306)],color='#B55F79')
    block(s,'ref',601,337,225,'Conv 3×3: 256→128\nGN → ReLU (one block)','prior',62)
    block(s,'pred',860,343,143,'Conv 1×1\n128→1','prior',50)
    s.port('out',1040,345,'Z_afm','prior',110)
    for a,b in [('cat','ref'),('ref','pred'),('pred','out')]:s.link(a,b)
    s.label(35,415,'A: 80×80×1, broadcast over E channels. Z_afm: 80×80×1, before Sigmoid.',16,1160)
    return s

def icg():
    s=Scene('04_ICG',1230,890);s.frame(8,8,1214,874,'(d) Image-based context guidance — explicit contrast and shared gates','loc')
    s.port('rgb',35,85,'RGB','enc',85);block(s,'rs',150,78,120,'Bilinear resize\n20×20','enc',60,16)
    block(s,'cnn1',300,78,157,'Conv 3×3: 3→16\nGN → ReLU','enc',60,16)
    block(s,'cnn2',490,78,157,'Conv 3×3: 16→32\nGN → ReLU','enc',60,16)
    s.port('app',680,85,'V: 32 ch','enc',110)
    for a,b in [('rgb','rs'),('rs','cnn1'),('cnn1','cnn2'),('cnn2','app')]:s.link(a,b)
    s.port('sem',35,175,'F_seu','loc',105)
    block(s,'pm',173,168,205,'Conv 1×1: 192→1\nSigmoid → P_init','loc',60,17);s.link('sem','pm')
    s.label(418,185,'V and P_init feed the three contrast paths below.',17,740)
    # Explicit named inputs keep the contrast lanes free of crossing buses.
    lanes=[(265,'V ⊙ P_init','FG: DW 5×5 → PW 1×1\n32→32; GN → ReLU'),
           (355,'V ⊙ (1−P_init)','BG: DW 5×5 → PW 1×1\n32→32; GN → ReLU'),
           (445,'P_init < 0.1','Masked mean of V\nspatial broadcast; 32 ch')]
    for i,(y,title,detail) in enumerate(lanes):
        block(s,f'm{i}',40,y,185,title,'white',54,17)
        block(s,f'ag{i}',267,y-3,292,detail,'white',60,16);s.link(f'm{i}',f'ag{i}')
    s.plus('bgcat',620,418,'C')
    s.link('ag1','bgcat',via=[(585,382),(585,401)],sb='t')
    s.link('ag2','bgcat',via=[(585,472),(585,435)],sb='b')
    block(s,'bgf',675,390,165,'PW 1×1: 64→32\nGN → ReLU','white',58,16);s.link('bgcat','bgf')
    block(s,'diff',890,275,240,'Absolute difference\nlocal FG − fused BG','white',60)
    s.link('ag0','diff',via=[(805,292),(805,305)])
    s.link('bgf','diff','r','b',via=[(1010,419)])
    block(s,'g',890,478,240,'PW 1×1: 32→16\nGN → ReLU → G','loc',58,17)
    s.link('diff','g','r','r',via=[(1165,305),(1165,507)])
    # Main gate cascade: both gates receive the same G, with separate parameters.
    s.port('sx',40,573,'F_seu','loc',105)
    block(s,'gate1',195,566,210,'Attention gate 1\nX=F_seu; guidance=G','loc',65,16)
    block(s,'gate2',465,566,210,'Attention gate 2\nX=gate 1 output; same G','loc',65,16)
    s.port('out',735,576,'F_icg','loc',115)
    for a,b in [('sx','gate1'),('gate1','gate2'),('gate2','out')]:s.link(a,b)
    s.link('g','gate1','b','t',via=[(1010,550),(300,550)])
    s.route([(570,550),(570,566)],start=('g','b'),end=('gate2','t'));s.dot(570,550)
    s.label(38,663,'Each attention gate (independent weights):',18,850)
    s.port('gx',35,712,'X','loc',62);s.port('gg',35,805,'G','loc',62)
    block(s,'wx',135,705,163,'Conv 1×1\n192→96','white',60)
    block(s,'wg',135,795,163,'Conv 1×1\n16→96','white',60)
    s.plus('add',355,775);s.link('gx','wx');s.link('gg','wg')
    s.link('wx','add',via=[(330,735),(330,758)],sb='t');s.link('wg','add',via=[(330,825),(330,792)],sb='b')
    block(s,'norm',407,745,130,'GN → ReLU','white',60,16)
    block(s,'psi',575,745,145,'Conv 1×1\n96→1','white',60)
    block(s,'sig',758,745,120,'Sigmoid → α','white',60,16)
    s.plus('mul',955,775,'×');s.port('go',1030,752,'α ⊙ X','loc',123)
    for a,b in [('add','norm'),('norm','psi'),('psi','sig'),('sig','mul'),('mul','go')]:s.link(a,b)
    s.link('gx','mul','t','t',via=[(66,693),(955,693)])
    return s

def highres():
    s=Scene('05_AFМ_guided_HighRes'.replace('М','M'),1230,890)
    s.frame(8,8,1214,874,'(e) AFM-guided HighRes head — shared inputs, independent projections','hr')
    entries=[('fe','F_edge','80²×128','edge',93),('fl','F_loc','80²×128','loc',208),('rgb','RGB','320²×3','enc',323)]
    for key,label,shape,col,y in entries:
        s.node(key,35,y,135, sixty:=62,label+'\n'+shape,col,kind='tensor',fs=17)
        conv='Conv 3×3: 3→16' if key=='rgb' else 'Conv 1×1: 128→48'
        block(s,key+'p',212,y,235,conv+'\nGN → ReLU',col,62)
        after='DW 3×3 → PW 1×1\nGN/ReLU after each' if key=='rgb' else 'Bilinear ↑4\n320×320×48'
        block(s,key+'u',490,y,235,after,col,62)
        s.link(key,key+'p');s.link(key+'p',key+'u')
    s.frame(28,439,750,202,'M6 addition: AFM probability and ambiguity','prior')
    s.port('z',45,494,'Z_afm','prior',112)
    block(s,'up',188,488,125,'Bilinear ↑4\n320×320','prior',58,16)
    block(s,'sig',345,488,123,'Sigmoid','prior',58,17);s.port('p',510,495,'p','prior',62)
    block(s,'unc',606,552,135,'4p(1−p)\nambiguity u','fpn',60,17)
    for a,b in [('z','up'),('up','sig'),('sig','p')]:s.link(a,b)
    s.link('p','unc','b','l',via=[(541,582)])
    # Five separate input ports on a concat bar.
    s.node('concat',839,105,48,508,'C\no\nn\nc\na\nt','white',fs=20)
    for key,_,_,_,y in entries:
        s.route([s.point(key+'u','r'),(839,y+31)],start=(key+'u','r'),end=('concat','l'))
        s.label(757,y+3,'16 ch' if key=='rgb' else '48 ch',15,76)
    s.route([s.point('p','r'),(790,518),(790,478),(839,478)],start=('p','r'),end=('concat','l'))
    s.route([s.point('unc','r'),(811,582),(839,582)],start=('unc','r'),end=('concat','l'))
    s.label(798,452,'1',15,35);s.label(802,556,'1',15,35)
    block(s,'fuse',942,319,230,'Conv 1×1: 114→48\nGN → ReLU','hr',78,18)
    s.link('concat','fuse')
    block(s,'ref',942,463,230,'DW 3×3, groups=48\nGN → ReLU\nPW 1×1: 48→48\nGN → ReLU','hr',114,17)
    s.link('fuse','ref','b','t')
    s.node('hr',985,630,145,64,'F_hr\n320²×48','hr',kind='tensor',fs=18);s.link('ref','hr','b','t')
    block(s,'final',660,728,174,'Conv 1×1: 48→1\nFinal logits','prior',64,17)
    block(s,'edge',936,728,230,'Conv 1×1: 48→1\nEdge logits (auxiliary)','edge',64,17)
    s.link('hr','final','b','t',via=[(1057,710),(747,710)])
    s.link('hr','edge','b','t',via=[(1057,710),(1051,710)],dash=True)
    block(s,'pred',310,733,285,'Sigmoid → threshold\n320×320 binary stitch mask','prior',60,17)
    s.link('final','pred','l','r')
    s.label(36,824,'↑: bilinear interpolation, align_corners=False. Dashed output: auxiliary supervision.',16,1170)
    return s

def seu():
    s=Scene('06_SEU_attention',1230,595);s.frame(8,8,1214,579,'(f) Semantic enhancement and channel / spatial attention','loc')
    s.port('x',30,97,'P4','loc',65)
    block(s,'dw',129,91,133,'DW 3×3\nGN → ReLU','white',58,16)
    block(s,'pw',296,91,133,'PW 1×1\nGN → ReLU','white',58,16)
    block(s,'sa',463,91,122,'Spatial attn.\nexpanded below','loc',58,16)
    block(s,'eca',619,91,116,'ECA\nexpanded below','loc',58,16)
    block(s,'dw2',769,91,133,'DW 3×3\nGN → ReLU','white',58,16)
    block(s,'pw2',936,91,120,'PW 1×1 → GN','white',58,15);s.plus('add',1106,120)
    for a,b in [('x','dw'),('dw','pw'),('pw','sa'),('sa','eca'),('eca','dw2'),('dw2','pw2'),('pw2','add')]:s.link(a,b)
    s.link('x','add','b','b',via=[(62,193),(1106,193)])
    block(s,'act',1040,229,136,'ReLU → F_seu','loc',42,16);s.link('add','act','r','r',via=[(1192,120),(1192,250)])
    s.label(36,237,'SEU: all intermediate feature maps remain 20×20×192.',17,900)
    s.label(36,293,'Spatial attention (channel pooling)',18,950)
    s.port('sx',36,344,'X','loc',60)
    block(s,'pool',132,327,210,'Mean / Max over C\nConcat → 2 channels','white', seventy:=70,17)
    block(s,'conv',386,333,166,'Conv 7×7: 2→1','white',58,17)
    block(s,'sig',596,333,134,'Sigmoid','white',58);s.plus('mul',824,362,'×')
    for a,b in [('sx','pool'),('pool','conv'),('conv','sig'),('sig','mul')]:s.link(a,b)
    s.link('sx','mul','b','b',via=[(66,423),(824,423)])
    s.label(883,350,'Spatially weighted X',18,310)
    s.label(36,454,'Efficient channel attention (ECA)',18,950)
    s.port('ex',36,511,'X','loc',60)
    for k,x,w,t in [('gap',133,160,'Global AvgPool'),('reshape',327,178,'Squeeze / transpose'),('c1d',540,175,'Conv1D, k=5'),('esig',750,135,'Sigmoid')]:block(s,k,x,503,w,t,'white',58,16)
    s.plus('em',968,532,'×')
    for a,b in [('ex','gap'),('gap','reshape'),('reshape','c1d'),('c1d','esig'),('esig','em')]:s.link(a,b)
    s.link('ex','em','t','t',via=[(66,490),(968,490)])
    s.label(1010,520,'Channel-weighted X',16,190)
    return s

def encoder():
    s=Scene('07_encoder_primitives',1230,595);s.frame(8,8,1214,579,'(g) Encoder building blocks — MobileNetV4 Hybrid Medium','enc')
    s.label(30,57,'Fused inverted block / Stage 2',18,900)
    for k,x,w,t in [('x',30,80,'32 ch'),('er1',145,245,'Conv 3×3 / s2: 32→128\nBN → ReLU'),('er2',435,237,'Conv 1×1: 128→48\nBN'),('out',715,105,'C2: 48 ch')]:block(s,k,x,91,w,t,'enc',60,17)
    for a,b in [('x','er1'),('er1','er2'),('er2','out')]:s.link(a,b)
    s.label(30,182,'Universal inverted bottleneck (UIB)',18,900)
    ops=[('ux',30,60,'X'),('start',125,147,'DW-start*\nBN'),('exp',308,170,'PW expansion\nBN → ReLU'),('mid',514,158,'DW-middle*\nBN → ReLU'),('proj',708,161,'PW projection\nBN'),('scale',905,134,'LayerScale')]
    for k,x,w,t in ops:block(s,k,x,222,w,t,'white',60,16)
    for a,b in zip([x[0] for x in ops],[x[0] for x in ops][1:]):s.link(a,b)
    s.plus('uadd',1100,252);s.link('scale','uadd');s.link('ux','uadd','b','b',via=[(60,309),(1100,309)],dash=True)
    s.label(32,324,'* Kernel / stride vary by stage; identity when absent. Skip only at matching shape.',15,1160)
    s.label(30,375,'Mobile multi-query attention (MQA; 4 heads)',18,1120)
    block(s,'mx',30,432,65,'X','enc',60);block(s,'bn',125,432,70,'BN','white',60)
    block(s,'qkv',230,419,288,'Q: Conv 1×1 (4 heads)\nK, V: shared projections\nK/V DW downsample in Stage 4','white',86,16)
    block(s,'attn',557,430,200,'Scaled QKᵀ → Softmax\nAttention × V','white',64,16)
    block(s,'mp',798,430,154,'Output projection\nLayerScale','white',64,16)
    s.plus('madd',1020,462)
    for a,b in [('mx','bn'),('bn','qkv'),('qkv','attn'),('attn','mp'),('mp','madd')]:s.link(a,b)
    s.link('mx','madd','b','b',via=[(62,550),(1020,550)])
    s.label(1070,450,'Output',18,120)
    return s

def svg(scene,path):
    out=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{scene.w}" height="{scene.h}" viewBox="0 0 {scene.w} {scene.h}"><title>{escape(scene.name)}</title>',
         '<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="context-stroke"/></marker></defs>',
         '<rect width="100%" height="100%" fill="white"/>']
    # Frames at back, connectors next, operators at front.
    items=sorted(scene.items,key=lambda i:0 if i['kind']=='frame' else 1 if i['kind']=='line' else 2)
    for i in items:
        k=i['kind']; stroke=i['stroke']; dash=' stroke-dasharray="7 5"' if i.get('dash') else ''
        if k=='line':
            pts=' '.join(f'{x},{y}' for x,y in i['points']); marker=' marker-end="url(#arr)"' if i.get('arrow') else ''
            out.append(f'<polyline points="{pts}" fill="none" stroke="{stroke}" stroke-width="1.8" stroke-linejoin="round"{dash}{marker}/>');continue
        x,y,w,h=i['x'],i['y'],i['w'],i['h']
        if k in ('circle','dot'):out.append(f'<ellipse cx="{x+w/2}" cy="{y+h/2}" rx="{w/2}" ry="{h/2}" fill="{i["fill"]}" stroke="{stroke}" stroke-width="1.6"/>')
        elif k=='tensor':out.append(f'<polygon points="{x+9},{y} {x+w},{y} {x+w-9},{y+h} {x},{y+h}" fill="{i["fill"]}" stroke="{stroke}" stroke-width="1.4"/>')
        elif k!='text':out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{2 if k=="frame" else 5}" fill="{i["fill"]}" stroke="{stroke}" stroke-width="1.3"{dash}/>')
        if not i['text']:continue
        lines=i['text'].split('\n');fs=i['fs']; anchor='start' if k=='text' and i.get('align')=='left' else 'middle';xx=x if anchor=='start' else x+w/2
        yy=y+fs if k=='text' else y+h/2-(len(lines)-1)*fs*0.60+fs*0.34
        fill=stroke if k=='text' else COL['ink'];weight='600' if k=='text' else '400'
        for j,t in enumerate(lines):out.append(f'<text x="{xx}" y="{yy+j*fs*1.2}" font-family="Arial, Helvetica, sans-serif" font-size="{fs}" font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">{escape(t)}</text>')
    out.append('</svg>');path.write_text(''.join(out),encoding='utf-8')

def main():
    panels=[overview(),edge(),afm(),icg(),highres(),seu(),encoder()]
    full=Scene('APHNet_M6_detailed_publication',2560,2930)
    full.paste(panels[0],20,20,'a_')
    for panel,x,y,prefix in [(panels[1],20,950,'b_'),(panels[2],1310,950,'c_'),(panels[3],20,1440,'d_'),(panels[4],1310,1440,'e_'),(panels[5],20,2340,'f_'),(panels[6],1310,2340,'g_')]:full.paste(panel,x,y,prefix)
    full.h=2960
    scenes=[full,*panels]
    for sc in scenes:svg(sc,OUT/(sc.name+'.svg'))
    (OUT/'editable_scene.json').write_text(json.dumps([vars(sc) for sc in scenes],ensure_ascii=False,indent=2),encoding='utf-8')
    print(OUT)

if __name__=='__main__':main()

from pathlib import Path
import json
import sys
from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn

sys.path.insert(0, r"H:\C3Netqing\paper_draft_work\_deps")
from latex2mathml.converter import convert as latex_to_mathml
import mathml2omml


ROOT = Path(r"H:\C3Netqing")
WORK = ROOT / "paper_draft_work"
OUT = ROOT / "复杂织物背景下线迹缺陷检测方法_中文论文初稿.docx"
EQ_JSON = WORK / "equations.json"


def font_path(name):
    p = Path(r"C:\Windows\Fonts") / name
    return str(p) if p.exists() else None


def load_font(size, bold=False):
    candidates = ["simhei.ttf" if bold else "simsun.ttc", "msyhbd.ttc" if bold else "msyh.ttc", "arial.ttf"]
    for c in candidates:
        p = font_path(c)
        if p:
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def make_pipeline_diagram(path):
    W, H = 2200, 760
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    title = load_font(52, True)
    head = load_font(42, True)
    body = load_font(31, False)
    small = load_font(27, False)
    d.text((W/2, 45), "复杂织物背景下线迹缺陷检测总体流程", font=title, fill=(25, 42, 70), anchor="ma")
    boxes = [
        (80, 220, 520, 590, (224, 239, 255), "阶段一  MBDNet-P+", ["运动偏置去模糊", "方向尺度路由", "恢复线迹高频结构"]),
        (700, 220, 1140, 590, (225, 246, 236), "阶段二  M6", ["Mobile-FPN 多尺度融合", "边缘—定位双分支", "AFM 引导高分辨率细化"]),
        (1320, 220, 1760, 590, (255, 239, 216), "阶段三  TSR-Defect", ["软轨迹提取与去趋势", "同图像中位数自参照", "跳针/断线/歪斜/浮线判别"]),
        (1900, 275, 2140, 535, (245, 230, 240), "输出", ["缺陷类别", "缺陷位置", "可解释分数"]),
    ]
    for x1,y1,x2,y2,color,h,lines in boxes:
        d.rounded_rectangle((x1,y1,x2,y2), radius=28, fill=color, outline=(70,90,120), width=4)
        d.text(((x1+x2)/2,y1+55), h, font=head if x2-x1>300 else head, fill=(25,45,75), anchor="ma")
        yy=y1+150
        for line in lines:
            d.text(((x1+x2)/2,yy), line, font=body if x2-x1>300 else small, fill=(35,45,55), anchor="ma")
            yy += 62
    for a,b in [(520,700),(1140,1320),(1760,1900)]:
        y=405
        d.line((a+25,y,b-28,y), fill=(44,91,140), width=10)
        d.polygon([(b-28,y-20),(b,y),(b-28,y+20)], fill=(44,91,140))
    d.text((300,650), "模糊 RGB 图像", font=small, fill=(80,80,80), anchor="ma")
    d.text((920,650), "去模糊图像与像素级线迹掩膜", font=small, fill=(80,80,80), anchor="ma")
    d.text((1540,650), "单元级轨迹特征", font=small, fill=(80,80,80), anchor="ma")
    im.save(path, dpi=(180,180))


def make_tsr_diagram(path):
    W, H = 2200, 920
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    title = load_font(52, True)
    head = load_font(38, True)
    body = load_font(29, False)
    d.text((W/2, 40), "TSR-Defect 轨迹自参照推理流程", font=title, fill=(25,42,70), anchor="ma")
    nodes=[
        (65,240,385,620,"M6 概率掩膜",["逐列质量统计","软质心轨迹"]),
        (500,240,820,620,"趋势分解",["全局直线拟合","残差轨迹去趋势"]),
        (935,240,1255,620,"自适应分槽",["连通支撑区间","补充虚拟中心","保留边界截断"]),
        (1370,180,1710,680,"单元特征",["覆盖率与最大缺口","跨度与弧弦比","偏差、下垂与弯曲","局部斜率"]),
        (1825,180,2150,680,"自参照判别",["其余五单元中位数","连续分段映射","阈值仅由清晰校准集确定","输出四类缺陷"]),
    ]
    colors=[(225,239,255),(225,246,236),(255,239,216),(237,232,255),(255,230,235)]
    for idx,(x1,y1,x2,y2,h,lines) in enumerate(nodes):
        d.rounded_rectangle((x1,y1,x2,y2),radius=24,fill=colors[idx],outline=(70,90,120),width=4)
        d.text(((x1+x2)/2,y1+52),h,font=head,fill=(25,45,75),anchor="ma")
        yy=y1+145
        for line in lines:
            d.text(((x1+x2)/2,yy),line,font=body,fill=(35,45,55),anchor="ma")
            yy+=65
    for a,b in [(385,500),(820,935),(1255,1370),(1710,1825)]:
        y=430
        d.line((a+20,y,b-25,y),fill=(44,91,140),width=9)
        d.polygon([(b-25,y-18),(b,y),(b-25,y+18)],fill=(44,91,140))
    d.text((W/2,805),"关键思想：以同一幅图像中的重复线迹单元互为参照，避免依赖额外训练的缺陷分类器。",font=body,fill=(75,75,75),anchor="ma")
    im.save(path,dpi=(180,180))


def set_cell_margins(cell, top=80, start=100, bottom=80, end=100):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcMar = tcPr.first_child_found_in("w:tcMar")
    if tcMar is None:
        tcMar = OxmlElement('w:tcMar')
        tcPr.append(tcMar)
    for m, v in (("top",top),("start",start),("bottom",bottom),("end",end)):
        node = tcMar.find(qn(f'w:{m}'))
        if node is None:
            node = OxmlElement(f'w:{m}')
            tcMar.append(node)
        node.set(qn('w:w'), str(v))
        node.set(qn('w:type'), 'dxa')


def set_repeat_table_header(row):
    trPr = row._tr.get_or_add_trPr()
    tblHeader = OxmlElement('w:tblHeader')
    tblHeader.set(qn('w:val'), 'true')
    trPr.append(tblHeader)


def set_cell_shading(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:fill'), fill)
    tcPr.append(shd)


def set_repeat_keep(paragraph, keep_next=False):
    pPr = paragraph._p.get_or_add_pPr()
    keepLines = OxmlElement('w:keepLines')
    pPr.append(keepLines)
    if keep_next:
        keepNext = OxmlElement('w:keepNext')
        pPr.append(keepNext)


def set_run_font(run, cn="宋体", en="Times New Roman", size=10.5, bold=None, color=None):
    run.font.name = en
    run._element.rPr.rFonts.set(qn('w:eastAsia'), cn)
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color:
        run.font.color.rgb = RGBColor(*color)


def add_field(run, instr):
    fldChar1 = OxmlElement('w:fldChar'); fldChar1.set(qn('w:fldCharType'), 'begin')
    instrText = OxmlElement('w:instrText'); instrText.set(qn('xml:space'), 'preserve'); instrText.text = instr
    fldChar2 = OxmlElement('w:fldChar'); fldChar2.set(qn('w:fldCharType'), 'end')
    run._r.extend([fldChar1, instrText, fldChar2])


class Paper:
    def __init__(self):
        self.doc = Document()
        self.equations = []
        sec = self.doc.sections[0]
        sec.page_width, sec.page_height = Cm(21), Cm(29.7)
        sec.top_margin, sec.bottom_margin = Cm(2.2), Cm(2.0)
        sec.left_margin, sec.right_margin = Cm(2.5), Cm(2.3)
        sec.header_distance, sec.footer_distance = Cm(1.0), Cm(1.0)
        self.setup_styles()
        self.setup_header_footer(sec)

    def setup_styles(self):
        styles = self.doc.styles
        normal = styles['Normal']
        normal.font.name='Times New Roman'; normal._element.rPr.rFonts.set(qn('w:eastAsia'),'宋体'); normal.font.size=Pt(10.5)
        pf=normal.paragraph_format; pf.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY; pf.first_line_indent=Cm(0.74); pf.line_spacing=1.35; pf.space_after=Pt(0); pf.space_before=Pt(0)
        for name, size, cn, color in [('Heading 1',15,'黑体',(30,55,85)),('Heading 2',13,'黑体',(35,65,95)),('Heading 3',11,'黑体',(45,75,105))]:
            s=styles[name]; s.font.name='Times New Roman'; s._element.rPr.rFonts.set(qn('w:eastAsia'),cn); s.font.size=Pt(size); s.font.bold=True; s.font.color.rgb=RGBColor(*color)
            s.paragraph_format.first_line_indent=Cm(0); s.paragraph_format.space_before=Pt(10); s.paragraph_format.space_after=Pt(4); s.paragraph_format.keep_with_next=True
        if 'Caption CN' not in styles:
            s=styles.add_style('Caption CN',WD_STYLE_TYPE.PARAGRAPH)
            s.font.name='Times New Roman'; s._element.rPr.rFonts.set(qn('w:eastAsia'),'宋体'); s.font.size=Pt(9); s.paragraph_format.alignment=WD_ALIGN_PARAGRAPH.CENTER; s.paragraph_format.space_before=Pt(3); s.paragraph_format.space_after=Pt(6)
        if 'Equation Cell' not in styles:
            s=styles.add_style('Equation Cell',WD_STYLE_TYPE.PARAGRAPH)
            s.font.name='Cambria Math'; s._element.rPr.rFonts.set(qn('w:eastAsia'),'Cambria Math'); s.font.size=Pt(10); s.paragraph_format.alignment=WD_ALIGN_PARAGRAPH.CENTER; s.paragraph_format.first_line_indent=Cm(0); s.paragraph_format.space_after=Pt(0); s.paragraph_format.space_before=Pt(0)
        if 'Reference CN' not in styles:
            s=styles.add_style('Reference CN',WD_STYLE_TYPE.PARAGRAPH)
            s.font.name='Times New Roman'; s._element.rPr.rFonts.set(qn('w:eastAsia'),'宋体'); s.font.size=Pt(9); s.paragraph_format.first_line_indent=Cm(-0.65); s.paragraph_format.left_indent=Cm(0.65); s.paragraph_format.line_spacing=1.15; s.paragraph_format.space_after=Pt(2)

    def setup_header_footer(self, sec):
        p=sec.header.paragraphs[0]; p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        r=p.add_run('复杂织物背景下线迹缺陷检测方法（论文初稿）'); set_run_font(r,'宋体','Times New Roman',8.5,color=(100,100,100))
        p=sec.footer.paragraphs[0]; p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        r=p.add_run('— '); set_run_font(r,size=9,color=(90,90,90)); add_field(r,'PAGE'); r2=p.add_run(' —'); set_run_font(r2,size=9,color=(90,90,90))

    def paragraph(self, text='', bold_prefix=None, align=None, first=True, size=10.5, cn='宋体'):
        p=self.doc.add_paragraph()
        if align is not None: p.alignment=align
        if not first: p.paragraph_format.first_line_indent=Cm(0)
        if bold_prefix and text.startswith(bold_prefix):
            r=p.add_run(bold_prefix); set_run_font(r,cn,'Times New Roman',size,True)
            r=p.add_run(text[len(bold_prefix):]); set_run_font(r,cn,'Times New Roman',size)
        else:
            r=p.add_run(text); set_run_font(r,cn,'Times New Roman',size)
        return p

    def heading(self,text,level=1):
        p=self.doc.add_heading(text,level=level); set_repeat_keep(p,True); return p

    def equation(self, formula):
        n=len(self.equations)+1; self.equations.append({'formula':formula,'number':n})
        t=self.doc.add_table(rows=1, cols=3); t.alignment=WD_TABLE_ALIGNMENT.CENTER; t.autofit=False
        widths=[Cm(0.2),Cm(14.7),Cm(1.2)]
        for i,c in enumerate(t.rows[0].cells):
            c.width=widths[i]; c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER; set_cell_margins(c,20,20,20,20)
        t.cell(0,0).text=''
        p=t.cell(0,1).paragraphs[0]; p.style='Equation Cell'; p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        mathml = latex_to_mathml(formula)
        omml = mathml2omml.convert(mathml)
        wrapper = parse_xml('<root xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">' + omml + '</root>')
        p._p.append(wrapper[0])
        p=t.cell(0,2).paragraphs[0]; p.alignment=WD_ALIGN_PARAGRAPH.RIGHT; p.paragraph_format.first_line_indent=Cm(0); r=p.add_run(f'（{n}）'); set_run_font(r,'宋体','Times New Roman',9.5)
        return n

    def figure(self,path,caption,width_cm=15.5):
        p=self.doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.first_line_indent=Cm(0); set_repeat_keep(p,True)
        p.add_run().add_picture(str(path),width=Cm(width_cm))
        cp=self.doc.add_paragraph(caption,style='Caption CN'); set_repeat_keep(cp)

    def table(self, headers, rows, caption, widths=None, font_size=8.7):
        cp=self.doc.add_paragraph(caption,style='Caption CN'); set_repeat_keep(cp,True)
        t=self.doc.add_table(rows=1,cols=len(headers)); t.alignment=WD_TABLE_ALIGNMENT.CENTER; t.autofit=False
        hdr=t.rows[0]; set_repeat_table_header(hdr)
        for j,h in enumerate(headers):
            c=hdr.cells[j]; set_cell_shading(c,'DCE6F1'); c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER; set_cell_margins(c)
            p=c.paragraphs[0]; p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.first_line_indent=Cm(0)
            r=p.add_run(str(h)); set_run_font(r,'黑体','Times New Roman',font_size,True)
            if widths: c.width=Cm(widths[j])
        for row in rows:
            cells=t.add_row().cells
            for j,v in enumerate(row):
                c=cells[j]; c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER; set_cell_margins(c)
                p=c.paragraphs[0]; p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.first_line_indent=Cm(0); p.paragraph_format.line_spacing=1.05
                r=p.add_run(str(v)); set_run_font(r,'宋体','Times New Roman',font_size)
                if widths: c.width=Cm(widths[j])
        return t

    def save(self):
        self.doc.core_properties.title='复杂织物背景下基于运动偏置去模糊与轨迹自参照推理的线迹缺陷检测方法'
        self.doc.core_properties.subject='中文论文初稿'
        self.doc.core_properties.keywords='线迹缺陷检测；运动去模糊；细结构分割；轨迹自参照；复杂织物背景'
        self.doc.save(OUT)
        EQ_JSON.write_text(json.dumps(self.equations,ensure_ascii=False,indent=2),encoding='utf-8')


def build():
    pipeline=WORK/'pipeline_overview.png'; tsr=WORK/'tsr_workflow.png'
    make_pipeline_diagram(pipeline); make_tsr_diagram(tsr)
    p=Paper(); d=p.doc

    # Title page
    for _ in range(3): d.add_paragraph('')
    t=d.add_paragraph(); t.alignment=WD_ALIGN_PARAGRAPH.CENTER; t.paragraph_format.first_line_indent=Cm(0); t.paragraph_format.space_after=Pt(18)
    r=t.add_run('复杂织物背景下基于运动偏置去模糊与\n轨迹自参照推理的线迹缺陷检测方法'); set_run_font(r,'黑体','Times New Roman',20,True,color=(25,45,75))
    e=d.add_paragraph(); e.alignment=WD_ALIGN_PARAGRAPH.CENTER; e.paragraph_format.first_line_indent=Cm(0)
    r=e.add_run('Stitch Defect Detection in Complex Fabric Backgrounds via\nMotion-Biased Deblurring and Trajectory Self-Reference Reasoning'); set_run_font(r,'宋体','Times New Roman',12,False,color=(70,70,70))
    for _ in range(4): d.add_paragraph('')
    for label in ['作者：____________________','单位：____________________','日期：2026年8月']:
        q=d.add_paragraph(); q.alignment=WD_ALIGN_PARAGRAPH.CENTER; q.paragraph_format.first_line_indent=Cm(0); q.paragraph_format.space_after=Pt(12)
        rr=q.add_run(label); set_run_font(rr,'宋体','Times New Roman',12)
    d.add_page_break()

    p.heading('摘  要',1)
    p.paragraph('工业缝制过程中，织物纹理、印花图案、纱线反光以及设备振动共同造成复杂成像干扰。线迹本身具有细长、低占比和重复排列等特点，轻微运动模糊即可削弱其边缘与中心轨迹，使像素级分割误差进一步放大为跳针、断线、整体歪斜及松线/浮线等单元级缺陷的误判。针对这一问题，本文构建由运动偏置去模糊网络 MBDNet-P+、多尺度细线迹分割网络 M6 与轨迹自参照缺陷推理器 TSR-Defect 组成的三级检测方法。MBDNet-P+ 在轻量化编码—解码骨架中引入方向感知的多尺度运动模糊模块，通过通用、短水平和长水平深度卷积分支以及空间自适应路由，重点恢复沿缝制方向退化的高频结构；M6 以 MobileNetV4 为主干，结合 Mobile-FPN、边缘—定位双分支、注意力融合模块与 AFM 引导的高分辨率细化头，在控制参数量的同时保持细线迹的连续性与边界精度；TSR-Defect 则从概率掩膜提取亚像素软轨迹，通过全局趋势分解、自适应线迹分槽和同图像中位数参照，对四类缺陷进行可解释的规则判别。实验表明，MBDNet-P+ 在六域共 500 幅测试图像上取得 29.5201 dB 的平均 PSNR 和 0.9226 的 SSIM，参数量为 13.70 M；M6 在 DB、XJ、SD 与外部 RX 数据上的宏平均 IoU、F1、BF1@1 和 TrajF1@1 分别达到 0.8581、0.9233、0.9796 和 0.9923。在 48 幅源图像的分组四折端到端实验中，直接模糊使缺陷宏平均 F1 从 82.89% 降至 52.66%，经去模糊后回升至 71.01%，恢复了模糊损失的 60.7%。结果说明，面向运动方向的复原、面向细结构的分割与同图像自参照推理之间具有明确的误差抑制协同关系。需要指出的是，当前端到端结论基于小样本和受控合成运动模糊，真实产线泛化能力仍需进一步验证。',first=False)
    p.paragraph('关键词：复杂织物；线迹缺陷检测；运动去模糊；细结构分割；轨迹自参照',bold_prefix='关键词：',first=False)

    p.heading('Abstract',1)
    p.paragraph('Complex fabric textures, printed patterns, yarn reflections, and mechanical vibration jointly degrade stitch images in industrial sewing. Because stitches are thin, sparsely occupied, and periodically repeated, even moderate motion blur can suppress edge and centerline evidence and propagate pixel-level errors to unit-level defect decisions. This paper presents a three-stage method consisting of MBDNet-P+ for motion-biased deblurring, M6 for multi-scale fine-stitch segmentation, and TSR-Defect for trajectory self-reference reasoning. MBDNet-P+ employs generic, short-horizontal, and long-horizontal depthwise branches with spatial routing to restore directionally degraded details. M6 combines a MobileNetV4 backbone, Mobile-FPN, edge-localization dual branches, attention fusion, and an AFM-guided high-resolution refinement head. TSR-Defect extracts a soft trajectory from the probability mask and performs trend decomposition, adaptive unit partitioning, same-image median referencing, and interpretable scoring for skipped stitches, thread breaks, global skew, and loose/floating threads. Across 500 images from six test domains, MBDNet-P+ achieves 29.5201 dB PSNR and 0.9226 SSIM with 13.70 M parameters. M6 obtains macro-averaged IoU, F1, BF1@1, and TrajF1@1 of 0.8581, 0.9233, 0.9796, and 0.9923 over DB, XJ, SD, and external RX datasets. On a grouped four-fold end-to-end evaluation with 48 source images, blur reduces defect macro-F1 from 82.89% to 52.66%, whereas deblurring raises it to 71.01%. These results support the complementarity of direction-aware restoration, topology-sensitive segmentation, and self-referenced trajectory reasoning. The present end-to-end evidence is limited to a small dataset and controlled synthetic blur; real production-line validation remains necessary.',first=False,cn='Times New Roman')
    p.paragraph('Keywords: complex fabric; stitch defect detection; motion deblurring; fine-structure segmentation; trajectory self-reference',bold_prefix='Keywords:',first=False,cn='Times New Roman')

    p.heading('1 引言',1)
    for txt in [
        '缝制线迹是服装、家纺与工业织物质量控制中的关键对象。与普通表面缺陷相比，线迹区域通常只占图像中的少量像素，却需要同时保持边缘、宽度、连续性和周期结构。复杂织物的经纬纹理、花纹边界和纱线高光会产生与线迹相似的局部梯度，导致分割网络容易将背景纹理误判为线迹，或在低对比度位置产生断裂。与此同时，输送机构和相机相对运动造成的模糊往往沿缝制方向延展，使原本有限的横向边界对比度进一步下降。',
        '现有研究常将图像复原、语义分割和缺陷分类作为相互独立的任务处理。然而，在线迹检测中，三个任务存在明显的级联关系：复原阶段决定可供分割利用的高频信息，分割阶段决定轨迹几何是否连续，缺陷推理阶段则将局部像素扰动累积为单元级决策。如果仅优化单一阶段，较高的像素指标未必能够转化为稳定的缺陷识别性能。因此，本文从“退化抑制—结构提取—几何推理”的完整信息链出发设计系统。',
        '本文不额外训练缺陷分类器，而是利用一幅图像中重复出现的多个线迹单元构造内部参照。该策略符合工业场景中“同批次、同工艺、同图像内多数线迹正常”的先验，并能够直接给出覆盖率、缺口、跨度、弧弦比、残差偏差和下垂量等可解释依据。对于样本量有限且缺陷标注成本较高的场景，这种像素学习与几何推理相结合的路线具有现实意义。'
    ]: p.paragraph(txt)
    p.paragraph('本文的主要贡献概括如下：',first=False)
    for i,txt in enumerate([
        '提出面向线迹方向退化的 MBDNet-P+。该网络以深度卷积—逐点卷积块为基础，通过多尺度水平分支、内容门控和尺度门控实现局部自适应去模糊，并以低分辨率长程上下文补充大位移退化建模。',
        '构建兼顾轻量性与精细结构保持的 M6 分割网络。网络以 MobileNetV4 与 Mobile-FPN 建立多尺度语义，在边缘和定位双分支之间引入注意力融合，并将粗分割概率与不确定性显式反馈到原分辨率细化头。',
        '提出无需训练分类器的 TSR-Defect。通过软轨迹、趋势去除、自适应分槽和同图像中位数自参照，将像素掩膜转化为四类线迹缺陷分数，并对边缘截断、天然曲率和全局斜率进行有针对性的解耦。',
        '建立三级端到端评估协议。在源图像级分组四折条件下固定清晰校准集得到的阈值，分别评估清晰输入、直接模糊输入和去模糊输入，从而量化去模糊对最终缺陷判别的实际贡献。'
    ],1):
        q=p.paragraph(f'（{i}）{txt}',first=False); q.paragraph_format.left_indent=Cm(0.74); q.paragraph_format.right_indent=Cm(0.3)

    p.heading('2 相关工作',1)
    p.heading('2.1 单幅图像运动去模糊',2)
    p.paragraph('传统盲去卷积通常显式估计模糊核，再通过正则化反演获得清晰图像；当织物纹理与非均匀运动同时存在时，核估计误差容易产生振铃和伪纹理。深度网络通过端到端学习退化到清晰图像的映射，逐渐形成多阶段复原、编码—解码和频域建模等路线。MPRNet 采用多阶段渐进复原机制[1]，NAFNet 则以简洁的无激活块证明了轻量骨架的有效性[2]。针对本文场景，核心困难并非无限制扩大网络，而是让有限参数优先作用于沿缝制方向展开的中长距离模糊，同时避免在高分辨率层过度平滑细线迹。')
    p.heading('2.2 细结构语义分割',2)
    p.paragraph('U-Net 及其改进结构依靠编码—解码和跳跃连接融合语义与细节[3]。FPN 通过自顶向下路径和横向连接构建多尺度特征[4]，适用于尺度变化明显的目标。细线迹与道路、血管等细结构相似，像素占比低且拓扑连续性重要，单纯提高区域重叠率可能掩盖边界毛刺和局部断裂。Focal Loss 可抑制易分类背景像素的主导作用[5]，Tversky Loss 通过非对称控制假阳性和假阴性，适合类别不平衡任务[6]。本文在此基础上进一步把边缘表征、定位语义和原图细节进行分工，并将粗分割的不确定区域作为高分辨率细化的显式引导。')
    p.heading('2.3 线迹缺陷识别与几何推理',2)
    p.paragraph('已有线迹检测方法多使用纹理描述子、轮廓规则或目标检测网络。深度分类与检测方法能够学习复杂外观，但需要较多缺陷样本，且在织物、线色和照明变化后可能出现域偏移。线迹在局部窗口内具有重复单元，这一结构先验使“同图像内部比较”成为可能。本文将缺陷推理限制在分割轨迹的几何统计上：跳针主要体现为局部覆盖下降和缺口增大，断线表现为跨度和弧长不足，整体歪斜体现为全局线性趋势，松线/浮线则表现为相对同图像正常单元的弯曲、下垂和弧长过剩。这种显式分解能够降低外观域差异对最终判别的影响。')

    p.heading('3 问题定义与总体框架',1)
    p.heading('3.1 成像退化与任务定义',2)
    p.paragraph('设清晰织物图像为二维空间上的彩色函数，运动模糊由空间相关或近似空间不变的核作用后叠加噪声形成。为便于网络设计与受控实验，本文采用下式描述观测过程。')
    p.equation(r'B=K\otimes I+n')
    p.paragraph('式（1）中，B 为模糊观测，I 为潜在清晰图像，K 为运动模糊核，⊗ 表示卷积，n 表示成像噪声。三级系统首先估计清晰图像，再预测线迹概率掩膜，最后把概率掩膜映射为四类缺陷标签与相应位置。')
    p.equation(r'\hat I=D_{\theta}(B),\quad P=S_{\phi}(\hat I),\quad \hat y=R(P)')
    p.paragraph('式（2）中，D、S 和 R 分别对应 MBDNet-P+、M6 与 TSR-Defect；θ 和 φ 为可学习参数，而 R 为基于校准阈值的确定性推理器。该划分使复原和分割可以使用不同规模的数据训练，同时保持缺陷判别的可解释性。')
    p.figure(pipeline,'图1  本文三级线迹缺陷检测总体框架',16.0)
    p.heading('3.2 三级信息接口',2)
    p.paragraph('第一级输出三通道去模糊图像，采用全局残差形式保留低频色彩和织物基底；第二级接收归一化图像，输出与输入同尺寸的线迹 logits、边缘 logits 以及预留的中心线 logits，其中当前 M6 配置仅监督线迹主输出、边缘输出和低分辨率定位输出；第三级仅使用主输出经 Sigmoid 得到的概率图进行几何分析，RGB 图像只用于可视化，不参与缺陷分数计算。因而，端到端性能的关键接口是“概率掩膜是否连续、是否保留真实曲率，以及虚假背景响应是否足够低”，而非单纯追求复原图像的主观锐化。')
    p.heading('3.3 设计原则',2)
    p.paragraph('系统设计遵循三个原则。第一，退化方向与网络感受野匹配：对近水平线迹的主要模糊方向配置短、长水平深度卷积分支，同时保留通用二维分支以适应局部偏转。第二，区域语义与边界细节分离学习：定位分支负责抑制背景并提供注意力，边缘分支负责保存细线迹轮廓，原分辨率细化头再对二者进行联合校正。第三，局部形变与全局趋势解耦：TSR-Defect 先拟合整体斜率，再基于残差轨迹判断局部松弛或断裂，避免把整幅图像的正常倾斜误判为单元异常。')

    p.heading('4 方法',1)
    p.heading('4.1 运动偏置去模糊网络 MBDNet-P+',2)
    p.heading('4.1.1 网络总体结构',3)
    p.paragraph('MBDNet-P+ 采用三尺度 U 形编码—解码结构。输入首先经 3×3 卷积映射到 64 通道；全分辨率编码器含 3 个 DWPW 块，二分之一分辨率编码器含 4 个 DWPW 块和 1 个 MBD 模块，四分之一分辨率编码器依次包含 12 个 DWPW 块、1 个 LRC 模块、1 个 MBD 模块、2 个 DWPW 块和另 1 个 MBD 模块。解码端在二分之一分辨率使用 4 个 DWPW 块和 1 个 MBD 模块，在全分辨率使用 3 个 DWPW 块。三个尺度的通道数分别为 64、128 和 256。全网共含 28 个 DWPW 块和 4 个 MBD 模块，参数量为 13.70 M。')
    p.paragraph('下采样采用步长为 2 的深度卷积与 1×1 逐点卷积组合，使空间压缩和通道变换解耦；上采样先通过 1×1 卷积扩展通道，再使用 PixelShuffle 重排到高分辨率。编码器与解码器之间使用同尺度跳跃连接，减少高频细节在瓶颈层中的损失。MBD 模块仅放置于二分之一和四分之一尺度：一方面，低分辨率上的单位卷积核覆盖更大的原图范围，适合处理中长位移模糊；另一方面，可避免在全分辨率层反复施加方向滤波造成线迹过锐化或背景纹理放大。')
    p.figure(WORK/'deblurring'/'media'/'image1.png','图2  MBDNet-P+ 总体网络结构',15.7)
    p.heading('4.1.2 DWPW 基础复原块',3)
    p.paragraph('DWPW 块负责局部纹理恢复和通道交互。其第一子层先进行二维 LayerNorm 和通道扩展，再用 3×3 深度卷积提取每个通道的空间模式；随后 SimpleGate 将通道均分为两组并逐元素相乘，实现无显式激活函数的非线性调制，最后通过 1×1 卷积投影回输入通道。第二子层对应轻量前馈网络，再次进行归一化、逐点扩展、SimpleGate 与逐点投影。两个子层均采用残差连接。')
    p.equation(r'\begin{array}{l} X_1=X+\beta W_{p2}(\mathrm{SG}(Q)), \\ Q=\mathrm{DW}_{3\times3}(W_{p1}(\mathrm{LN}(X))) \end{array}')
    p.equation(r'Y=X_1+\gamma W_{p4}(\mathrm{SG}(W_{p3}(\mathrm{LN}(X_1))))')
    p.paragraph('式（3）和式（4）中，SG 表示 SimpleGate，DW 表示逐通道空间卷积，W 表示 1×1 逐点投影；β 和 γ 以零初始化。零初始化使网络开始训练时接近恒等映射，从而避免深层残差堆叠在初期破坏输入结构。与常规卷积相比，深度卷积承担空间建模，逐点卷积承担通道混合，在保持较大特征宽度的同时降低参数和计算量。')
    p.heading('4.1.3 多尺度运动偏置模块 MBD',3)
    p.paragraph('MBD 的目标是使不同位置根据自身模糊程度选择合适的方向感受野。输入特征首先进入三个并行深度卷积分支：通用分支使用 3×3 卷积，能够响应斜向边缘、织物交叉纹理和非水平运动；短水平分支使用 1×5 卷积，聚焦轻度水平拖影；长水平分支使用膨胀率为 3 的 1×5 卷积，其水平方向有效感受野为 13 个特征像素，用于覆盖更长的运动轨迹。')
    p.equation(r'\begin{array}{l} F_g=\mathrm{DW}_{3\times3}(X),\quad F_s=\mathrm{DW}_{1\times5}(X), \\ F_l=\mathrm{DW}_{1\times5,d=3}(X) \end{array}')
    p.paragraph('路由器先用 3×3 深度卷积汇聚邻域上下文，再分别产生尺度门控与内容门控。尺度门控在短、长水平分支间采用 Softmax，使二者形成竞争关系；内容门控采用 Sigmoid，在通用二维响应与水平运动响应之间进行逐位置选择。')
    p.equation(r'(\alpha_s,\alpha_l)=\mathrm{Softmax}\!\left(W_s(\mathrm{DW}_{3\times3}(X))\right)')
    p.equation(r'G=\sigma\!\left(W_g(\mathrm{DW}_{3\times3}(X))\right)')
    p.equation(r'F_h=\alpha_s\odot F_s+\alpha_l\odot F_l')
    p.equation(r'F_m=(1-G)\odot F_g+G\odot F_h')
    p.equation(r'Y=X+\eta W_o(F_m)')
    p.paragraph('式（8）中的尺度权重使轻度和重度模糊可共享同一模块；式（9）中的内容门控避免所有位置都被强制施加水平滤波。例如，织物背景的斜向经纬纹理可更多依赖通用分支，而线迹拖影区域可提高水平分支权重。式（10）的残差尺度 η 同样以零初始化，使新增方向模块能够在已有轻量骨架上渐进学习。MBD 的本质不是固定使用水平卷积，而是把水平运动作为候选先验，并由局部内容决定其作用强度。')
    p.figure(WORK/'deblurring'/'media'/'image2.png','图3  DWPW 块与多尺度运动偏置模块',15.7)
    p.heading('4.1.4 低分辨率长程上下文模块 LRC',3)
    p.paragraph('仅依赖方向卷积可能难以区分长拖影与真实的长纹理。为此，网络在四分之一分辨率瓶颈处配置 LRC。该模块将普通 3×3 深度卷积与膨胀率为 2 的 3×3 深度卷积并联，前者保留局部细节，后者扩大二维上下文；二者拼接后经 1×1 卷积融合，并以零初始化残差形式写回主干。')
    p.equation(r'\begin{array}{l} Y=X+\lambda W_c([A,B]), \\ A=\mathrm{DW}_{3\times3}(X),\quad B=\mathrm{DW}_{3\times3,d=2}(X) \end{array}')
    p.paragraph('LRC 只在最低空间尺度出现一次，可用较低计算成本补充非局部背景判断。它与 MBD 的分工为：MBD 强调运动方向及长度选择，LRC 强调二维上下文一致性；二者组合有助于抑制把织物长纹理误恢复为线迹的风险。')
    p.heading('4.1.5 输出重建与优化目标',3)
    p.paragraph('解码器输出经 3×3 卷积恢复到 RGB 通道，并与输入图像形成全局残差。该设计让网络主要预测被模糊抹除的高频修正量，而无需重新生成低频颜色。')
    p.equation(r'\hat I=B+W_{out}(F_{dec})')
    p.paragraph('训练以对数均方误差形式的重建项为主，使优化方向与 PSNR 一致；针对线迹主要沿水平方向延伸的特点，增加水平梯度 Charbonnier 约束，强调垂直边缘变化；FT2 阶段再加入 SSIM 项以改善局部结构。')
    p.equation(r'L_{rec}=10\log_{10}\!\left(\mathrm{MSE}(\hat I,I)+\varepsilon\right)')
    p.equation(r'L_{grad}=\mathrm{Mean}\!\left(\sqrt{(\nabla_x\hat I-\nabla_x I)^2+\varepsilon^2}\right)')
    p.equation(r'\begin{array}{l} L_{deb}=L_{rec}+0.02L_{grad} \\ \quad +0.03\left(1-\mathrm{SSIM}(\hat I,I)\right) \end{array}')
    p.paragraph('基础训练 S30 使用 30 个 epoch 和初始学习率 1×10⁻³；随后 FT1 继续训练 10 个 epoch，学习率为 2×10⁻⁵；FT2 再训练 5 个 epoch，学习率为 5×10⁻⁶。训练裁块为 320×320，批量大小为 2，梯度累积 2 次，对应有效批量 4，并使用混合精度、余弦学习率调度和随机种子 42。')

    p.heading('4.2 多尺度细线迹分割网络 M6',2)
    p.heading('4.2.1 MobileNetV4 主干与 Mobile-FPN',3)
    p.paragraph('M6 使用预训练 MobileNetV4 Hybrid Medium 作为编码主干[7]。对 320×320 输入，主干依次输出 C2、C3、C4 和 C5，空间尺寸分别为 80×80、40×40、20×20 和 10×10，通道数分别为 48、80、160 和 960。高层特征具有较强背景判别能力，但空间分辨率不足；低层特征保留线迹边缘，却容易受到织物纹理干扰。因此，Mobile-FPN 先用 1×1 卷积将各层统一投影到 192 通道，再通过自顶向下融合逐级传播语义。')
    p.equation(r'P_l=\psi_l\!\left(L_l+\mathrm{Up}(P_{l+1})\right)')
    p.paragraph('式（16）中，L 表示横向投影特征，Up 表示双线性上采样，ψ 表示融合后的卷积整合。与直接拼接全部尺度相比，逐级相加能够控制通道规模，并使高层的“线迹位置”先验逐步约束低层纹理响应。')
    p.figure(WORK/'segmentation'/'media'/'image1.png','图4  M6 线迹分割网络总体结构',16.0)
    p.heading('4.2.2 边缘—定位双分支',3)
    p.paragraph('FPN 输出进入边缘分支和定位分支。边缘分支侧重较高分辨率特征，通过局部卷积保持线迹两侧轮廓与细小弯曲；定位分支聚合更强的多尺度语义，用于判断哪些细长响应属于真实线迹而非织物纹理。两个分支的中间通道均设置为 128。分支解耦的动机在于：若用单一特征同时承担边界锐度和全局定位，强语义可能造成过度平滑，而强细节又可能放大背景。')
    p.figure(WORK/'segmentation'/'media'/'image2.png','图5  Mobile-FPN 与边缘—定位双分支',16.0)
    p.heading('4.2.3 注意力融合模块 AFM',3)
    p.paragraph('AFM 将边缘特征和定位特征分别通过 1×1 Conv-GN-ReLU 适配到相同通道。定位特征再经 3×3 Conv-GN-ReLU 与 1×1 卷积生成单通道空间注意图。该注意图对边缘特征进行逐像素门控，随后与未经门控的定位特征拼接并卷积融合，产生低分辨率粗分割 logits。')
    p.equation(r'E=\phi_e(F_e),\quad L=\phi_l(F_l)')
    p.equation(r'A=\sigma(g(L))')
    p.equation(r'F_f=\psi([E\odot A,L]),\quad Z_c=h(F_f)')
    p.paragraph('式（18）中的注意图由定位分支产生，因此它表达“当前位置像线迹的可信度”；乘法门控保留可信区域的边缘细节并抑制无关纹理。与此同时，融合时仍保留完整定位特征，避免注意图过早饱和导致信息不可逆丢失。AFM 的粗 logits 不仅接受低分辨率监督，还被送入高分辨率细化头，因而承担语义锚点和不确定性估计的双重作用。')
    p.heading('4.2.4 AFM 引导的高分辨率细化头',3)
    p.paragraph('为避免直接在 80×80 或更低分辨率上输出造成一至数个像素的轨迹偏移，M6 将边缘和定位特征分别投影到 48 通道，并双线性上采样至 320×320。原始 RGB 图像经 3×3 Conv-GN-ReLU 与深度可分离卷积提取 16 通道浅层细节，向细化头提供未经主干降采样的局部边界。')
    p.paragraph('粗 logits 上采样后经 Sigmoid 得到概率图，并计算对称不确定性。概率接近 0 或 1 时不确定性低，接近 0.5 时达到最大值 1，因此网络能够把更多修正能力集中在边界、断点和低对比度区域。')
    p.equation(r'P_c=\sigma(\mathrm{Up}(Z_c))')
    p.equation(r'U_c=4P_c(1-P_c)')
    p.equation(r'\begin{array}{l} F_h=\rho([\mathrm{Up}(E_h),\mathrm{Up}(L_h), \\ \quad F_{rgb},P_c,U_c]) \end{array}')
    p.equation(r'\begin{array}{l} Z_f=W_f(F_h),\quad Z_e=W_e(F_h), \\ Z_{ctr}=W_{ctr}(F_h) \end{array}')
    p.paragraph('式（22）将上采样边缘特征、上采样定位特征、原图细节、粗概率和不确定性共同拼接，经 1×1 融合与深度可分离细化块获得高分辨率特征。细化头同时预留最终掩膜、边缘与中心线三个输出。需要强调的是，当前 M6 配置启用了最终掩膜、边缘和低分辨率定位监督，中心线、拓扑与额外边界损失权重均为零，因此本文不将预留分支描述为已参与训练的贡献。')
    p.figure(WORK/'segmentation'/'media'/'image3.png','图6  AFM 引导的高分辨率细化头',16.0)
    p.heading('4.2.5 分割损失函数',3)
    p.paragraph('针对前景稀疏问题，各有效输出采用 Focal Loss 与 Tversky Loss 的组合。Focal Loss 降低大量易分类背景像素的梯度权重，Tversky Loss 则用不同系数调节漏检与误检。')
    p.equation(r'L_{focal}=-\alpha(1-p_t)^{\gamma}\log(p_t)')
    p.equation(r'\begin{array}{l} \mathrm{TI}=\frac{\mathrm{TP}}{\mathrm{TP}+\alpha_t\mathrm{FP}+\beta_t\mathrm{FN}+\varepsilon}, \\ L_{tv}=1-\mathrm{TI} \end{array}')
    p.equation(r'\begin{array}{l} L_{seg}=1.0L_{edge}+1.15L_{loc} \\ \quad +1.2\left(L_{final}+0.2L_{dice-edge}\right) \end{array}')
    p.paragraph('实验配置中 Focal Loss 的 α 为 0.25、γ 为 3；定位分支 Tversky 系数为 0.6 和 0.4，更强调抑制误检；最终分支系数为 0.4 和 0.6，更强调减少线迹漏检。最终分支另加入权重 0.2 的边缘 Dice 项，以缓解预测轮廓膨胀。主干采用较小学习率，具体为基础学习率的 0.2 倍，使预训练特征缓慢适应织物域，而新建解码和细化模块以完整学习率优化。')

    p.heading('4.3 轨迹自参照缺陷推理器 TSR-Defect',2)
    p.heading('4.3.1 软轨迹提取与有效列筛选',3)
    p.paragraph('TSR-Defect 不对二值掩膜直接骨架化，而是保留概率信息计算每一列的质量与纵向软质心。这样可减少阈值切换引起的轨迹跳变，并让低置信区域自然降低贡献。')
    p.equation(r'm(x)=\frac{1}{H}\sum_{y=1}^{H}P(y,x)')
    p.equation(r't(x)=\frac{\sum_{y=1}^{H}yP(y,x)}{\sum_{y=1}^{H}P(y,x)+\varepsilon}')
    p.paragraph('当列质量不低于 0.005 时，该列被视为有效。质量阈值的作用是过滤由织物纹理产生的极弱概率响应，同时保留较细或局部模糊的线迹。与逐列取最大概率位置相比，软质心利用整条线迹宽度的信息，对边缘噪声更稳定。')
    p.heading('4.3.2 全局趋势分解',3)
    p.paragraph('图像整体倾斜与单个线迹单元弯曲具有不同的缺陷语义。推理器先在全部有效列上拟合线性趋势，再以残差轨迹描述局部形变。')
    p.equation(r'[a,b]=\mathrm{argmin}_{a,b}\sum_{x\in\Omega}(t(x)-ax-b)^2')
    p.equation(r'r(x)=t(x)-(ax+b)')
    p.paragraph('斜率 a 用于整体歪斜判别，残差 r 用于跳针、断线和松线/浮线判别。该分解可避免一条整体倾斜但内部规则的线迹因纵向偏差较大而被误认为松线。当前配置启用全局斜率判别而关闭局部斜率缺陷判别。')
    p.heading('4.3.3 自适应线迹分槽',3)
    p.paragraph('每幅图像预期包含 6 个水平重复单元。推理器首先在有效列上寻找连续支撑区间，并删除长度小于 3 像素或小于支撑区间中位长度 15% 的微小片段。若候选区间多于 6 个，则优先保留质量和长度占优的区间；若少于 6 个，则在最大中心间隙中插入虚拟中心，直至形成 6 个槽位。相邻中心的中点作为槽位边界。若支撑结构过于含混，则回退到等宽分槽。')
    p.paragraph('与固定等宽切分相比，自适应分槽能够容纳线迹周期的轻微变化，也能保留首尾天然截断单元。对于边缘截断的第一或第六槽位，跳针和断线分数被置零，因为图像裁切造成的短跨度和低覆盖不应被视为真实缺陷。')
    p.heading('4.3.4 单元特征与同图像参照',3)
    p.paragraph('对每个槽位提取覆盖率、最大缺口比、有效跨度比、弧弦比、残差偏差的 90% 分位数、下垂量、弯曲能量和局部斜率。覆盖率描述有效轨迹列所占比例；最大缺口比描述最长连续缺失区间；跨度比反映轨迹在槽内是否完整；弧弦比衡量曲线长度相对端点直线距离的增加；下垂量比较中间三分之一区域与两侧区域的残差高度。')
    p.equation(r'\begin{array}{l} q_i^{ref}=\mathrm{Median}(\mathcal Q_i), \\ \mathcal Q_i=\{q_j\mid j\ne i,\ j=1,\ldots,6\} \end{array}')
    p.paragraph('式（31）对每项特征分别使用其余五个单元的中位数作为参照。中位数对单个异常单元具有鲁棒性，并能随图像内线迹宽度、张力和曲率共同变化。该假设要求一幅图像中异常单元不占多数；这也是方法需要在后续真实工况中重点验证的适用边界。')
    p.heading('4.3.5 连续缺陷分数',3)
    p.paragraph('为避免硬阈值在边界附近产生突变，所有原始异常量先通过分段线性上升函数映射到 0 至 1。')
    p.equation(r'R(v;s,f)=\mathrm{clip}\!\left(\frac{v-s}{f-s},0,1\right)')
    p.paragraph('跳针分数由相对参照的最大缺口增量和覆盖率下降组成，二者权重分别为 0.60 和 0.40。')
    p.equation(r'\begin{array}{l} S_{skip}=0.60R(\Delta gap;0.05,0.28) \\ \quad +0.40R(\Delta cov;0.08,0.45) \end{array}')
    p.paragraph('断线判别先利用覆盖率与跨度构造“存在部分轨迹”的门控，以排除几乎完全空白的槽位；随后综合跨度短缺、缺口增量和弧长不足，并用弧长过剩项抑制松线形变被误判为断线。')
    p.equation(r'\begin{array}{l} G_{part}=R(cov;0.06,0.30) \\ \quad \cdot R(span;0.05,0.30) \end{array}')
    p.equation(r'\begin{array}{l} R_s=R(\Delta span;0.08,0.45), \\ R_g=R(\Delta gap;0.06,0.30), \\ R_a^-=R(\Delta arc^-;0.08,0.35), \\ R_a^+=R(\Delta arc^+;0.35,1.20), \\ T_b=0.50R_s+0.25R_g+0.25R_a^-, \\ S_{break}=G_{part}T_b(1-0.85R_a^+) \end{array}')
    p.paragraph('整体歪斜使用归一化斜率，宽高比校正使不同尺寸图像具有一致尺度；当前输入均为 320×320。')
    p.equation(r'S_{skew}=\mathrm{clip}\!\left(\frac{|a|W/H}{0.40},0,1\right)')
    p.paragraph('松线/浮线要求轨迹具有足够覆盖和跨度，再综合残差偏差、下垂量与弧长过剩。连续门控可避免断裂或大面积缺失时不稳定的曲率统计产生高分。')
    p.equation(r'\begin{array}{l} G_{cont}=R(cov;0.45,0.70) \\ \quad \cdot R(span;0.45,0.75) \end{array}')
    p.equation(r'\begin{array}{l} T_l=0.55R(\Delta dev;0.18,0.95) \\ \quad +0.25R(\Delta sag;0.20,1.20) \\ \quad +0.20R(\Delta arc^{+};0.04,0.25), \\ S_{loose}=G_{cont}T_l \end{array}')
    p.paragraph('每类分数与校准阈值比较后输出标签。阈值仅由各折的清晰校准子集确定，随后在清晰、直接模糊和去模糊三种条件下保持不变，从而避免利用测试条件重新调参。')
    p.figure(tsr,'图7  TSR-Defect 轨迹自参照推理流程',16.0)

    p.heading('5 实验设计',1)
    p.heading('5.1 数据集与划分',2)
    p.paragraph('去模糊训练使用 MultiSource SynBlur-v2。DB、XJ、SD 与 RX 四域的训练集共 4320 对图像，验证集共 160 对；测试集由 DB、XJ、SD、RX 各 100 幅以及外部 RXHF、TJML 各 50 幅构成，共 500 幅。RXHF 与 TJML 不参与训练，用于观察跨域复原能力。训练和评估裁块尺寸均与下游系统保持为 320×320。')
    p.paragraph('分割网络在 DB、XJ 和 SD 三域上训练，训练集 1080 幅、验证集 120 幅，RX 的 500 幅图像作为外部测试集。端到端缺陷数据含 48 幅独立源图像，每幅约包含 6 个水平线迹单元。水平翻转可形成 96 幅图像，但翻转样本只作为数据增强，不作为独立测试样本统计。')
    p.table(['任务','训练/校准数据','验证数据','测试数据','用途'],[
        ['去模糊','DB/XJ/SD/RX，共4320对','四域共160对','六域共500幅','训练与跨域评估'],
        ['线迹分割','DB/XJ/SD，共1080幅','三域共120幅','DB/XJ/SD各100幅；RX 500幅','内部与外部评估'],
        ['缺陷推理','每折36幅清晰图像仅校准阈值','—','每折12幅源图像，四折覆盖48幅','源图像级分组四折']
    ],'表1  各阶段数据划分与用途',[2.1,4.0,3.0,4.2,2.7])
    p.heading('5.2 合成模糊与端到端协议',2)
    p.paragraph('端到端实验为每幅清晰图像生成一个运动模糊对应样本。模糊核长度从 3、5、7、9、11、13 和 15 像素中采样，方向限制在水平轴附近 ±15°，随机种子为 42。48 幅源图像按源图像标识分成 4 折，每折使用 36 幅清晰图像进行阈值校准，12 幅用于测试。所有翻转或退化版本随源图像归入同一折，防止同源泄漏。')
    p.paragraph('每一折先在清晰校准集上确定四类缺陷阈值，然后冻结阈值，依次测试清晰图像经 M6、模糊图像直接经 M6，以及模糊图像经 MBDNet-P+ 后再经 M6 三种路径。该设计不以去模糊图像的 PSNR 代替下游效益，而是直接观察分割与缺陷指标的恢复幅度。')
    p.heading('5.3 实现细节',2)
    p.paragraph('M6 使用 AdamW 优化器，基础学习率为 1×10⁻⁴，主干学习率倍率为 0.2，权重衰减为 0.01；学习率采用余弦退火至 1×10⁻⁶。训练 150 个 epoch，批量大小为 2，启用混合精度，随机种子为 42，并以验证集二值 IoU 选择最佳模型。模型参数量为 10,267,319。')
    p.heading('5.4 评价指标',2)
    p.paragraph('去模糊采用 PSNR、SSIM、MAE 和单幅推理时间；分割采用精确率、召回率、F1、IoU、1 像素容差边界 F1（BF1@1）与 1 像素容差轨迹 F1（TrajF1@1）。缺陷推理在跳针、断线和松线/浮线三个单元级任务上统计 288 个单元，在整体歪斜任务上按 48 幅图像统计，并报告四类宏平均 F1、平均类别准确率、局部完全正确率和图像级完全正确率。')
    p.equation(r'\begin{array}{l} \mathrm{Precision}=\frac{TP}{TP+FP}, \\ \mathrm{Recall}=\frac{TP}{TP+FN} \end{array}')
    p.equation(r'\begin{array}{l} F_1=\frac{2\cdot\mathrm{Precision}\cdot\mathrm{Recall}}{\mathrm{Precision}+\mathrm{Recall}}, \\ \mathrm{IoU}=\frac{TP}{TP+FP+FN} \end{array}')

    p.heading('6 结果与分析',1)
    p.heading('6.1 去模糊定量结果',2)
    p.table(['方法','参数量/M','PSNR/dB','SSIM','中频PSNR/dB','时间/ms'],[
        ['NAFNet S30','17.11','28.3495','0.8982','27.7612','24.70'],
        ['MPRNet S30','20.13','29.2220','0.9201','28.4952','67.19'],
        ['MBDNet-P+ S30','13.70','29.4954','0.9222','28.8346','31.82'],
        ['MBDNet-P+ S30+FT15','13.70','29.5201','0.9226','28.8561','32.98']
    ],'表2  同一协议下不同去模糊网络的总体比较',[3.6,2.1,2.3,2.0,3.0,2.2])
    p.paragraph('在相同训练和测试协议下，MBDNet-P+ S30 的 PSNR 比 NAFNet 高 1.1459 dB，比 MPRNet 高 0.2734 dB，同时参数量分别减少约 19.9% 和 31.9%。FT1 与 FT2 继续微调后，PSNR 和 SSIM 进一步提升到 29.5201 dB 和 0.9226。该结果说明，将额外容量用于与运动方向相匹配的动态尺度路由，比单纯增加通道或阶段数更适合当前退化分布。')
    p.figure(WORK/'deblurring'/'media'/'image3.png','图8  去模糊方法精度、参数量与速度比较',15.0)
    p.table(['测试域','图像数','PSNR/dB','SSIM'],[
        ['DB','100','30.7782','0.9302'],['XJ','100','34.9884','0.9535'],['SD','100','32.0121','0.9514'],['RX','100','26.8644','0.9219'],['RXHF（外部）','50','28.3843','0.9170'],['TJML（外部）','50','24.0933','0.8615']
    ],'表3  MBDNet-P+ 最终模型在不同测试域上的结果',[3.7,2.4,3.2,3.2])
    p.paragraph('XJ 域取得最高 PSNR，而 TJML 外部域最低，说明跨织物和成像条件差异仍显著。尽管总体指标优于对比网络，外部域性能下降提示合成核分布和真实域外退化之间仍存在差距。后续可通过真实模糊—清晰成对采集、核分布自适应和无监督域适配进一步验证。')

    p.heading('6.2 分割消融与跨域结果',2)
    p.table(['模型','主要增量','IoU','F1','BF1@1','TrajF1@1'],[
        ['M0','MobileNetV4+Mobile-FPN 双分支','0.8485','0.9177','0.9770','0.9912'],
        ['M4','M0+高分辨率细化','0.8566','0.9224','0.9791','0.9921'],
        ['M6','M4+AFM 概率/不确定性引导','0.8581','0.9233','0.9796','0.9923']
    ],'表4  DB、XJ、SD 与 RX 上的宏平均消融结果',[2.0,5.6,2.0,2.0,2.2,2.5])
    p.paragraph('从 M0 到 M4，IoU 提升 0.0081，说明恢复到原分辨率并重新引入图像细节对细线迹区域重叠有明显作用；从 M4 到 M6，IoU 继续提升约 0.0015，同时边界和轨迹指标小幅上升。M6 的增益较小但方向一致，表明粗概率与不确定性主要发挥精修作用，而不是替代已有高分辨率特征。由于 TrajF1@1 已接近饱和，IoU 和 BF1 的提升更能反映对宽度与边界的进一步修正。')
    p.table(['数据域','IoU','F1','BF1@1','TrajF1@1'],[
        ['DB','0.890638','0.942092','0.989986','0.996821'],['XJ','0.829237','0.906471','0.966953','0.985925'],['SD','0.859807','0.924405','0.982282','0.992946'],['RX（外部）','0.852745','0.920094','0.979375','0.993572']
    ],'表5  M6 在各数据域上的分割结果',[3.4,2.8,2.8,3.0,3.2])
    p.paragraph('M6 在 DB 域上取得最高 IoU 和 F1，在外部 RX 域仍保持 0.852745 IoU 和 0.993572 TrajF1@1，说明主干语义、双分支与高分辨率细化对背景变化具有一定鲁棒性。XJ 的 IoU 相对较低，但轨迹 F1 仍达到 0.985925，说明主要误差更可能集中在线迹宽度与边界，而非中心轨迹完全丢失。')

    p.heading('6.3 清晰图像上的缺陷识别结果',2)
    p.table(['缺陷类型','评估粒度','样本数/阳性数','Accuracy/%','Precision/%','Recall/%','F1/%'],[
        ['跳针','单元级','288/12','98.61','78.57','91.67','84.62'],
        ['断线','单元级','288/12','98.61','100.00','66.67','80.00'],
        ['整体歪斜','图像级','48/12','93.75','90.91','83.33','86.96'],
        ['松线/浮线','单元级','288/12','98.61','100.00','66.67','80.00']
    ],'表6  清晰图像条件下 TSR-Defect 四类缺陷结果',[2.5,2.1,2.4,2.5,2.5,2.3,2.0])
    p.paragraph('清晰条件下四类宏平均 F1 为 82.89%。跳针召回率较高但精确率较低，说明局部低覆盖和分割缺口可能造成少量误报；断线与松线/浮线精确率均为 100%，但召回率为 66.67%，表明当前阈值更偏保守。整体歪斜在图像级取得 86.96% F1，验证了把全局斜率与局部残差分开的必要性。由于每类阳性仅 12 个，单个样本就会造成较大比例波动，因此这些结果更适合作为方法可行性证据，而非稳定泛化性能的最终估计。')

    p.heading('6.4 去模糊对端到端检测的贡献',2)
    p.table(['输入条件','分割P','分割R','分割F1','IoU','TrajF1@1','BF1@1'],[
        ['清晰','0.9302','0.8589','0.8912','0.8045','0.9804','0.9512'],
        ['直接模糊','0.8201','0.8426','0.8254','0.7076','0.9151','0.8604'],
        ['去模糊后','0.9287','0.8429','0.8812','0.7889','0.9734','0.9395']
    ],'表7  三种输入条件下的端到端分割结果',[3.1,2.0,2.0,2.3,2.1,2.8,2.5])
    p.paragraph('直接模糊使 IoU 从 0.8045 降至 0.7076，TrajF1@1 从 0.9804 降至 0.9151，说明运动拖影不仅扩大或缩小区域，也会破坏轨迹连续性。去模糊后 IoU 回升到 0.7889，TrajF1@1 回升到 0.9734，均接近清晰上限。精确率几乎恢复到清晰条件，而召回率仍略低，提示复原网络更擅长抑制背景伪响应，但对少数被严重抹除的线迹像素无法完全重建。')
    p.table(['输入条件','缺陷宏F1/%','平均类别准确率/%','局部完全正确率/%','图像完全正确率/%'],[
        ['清晰','82.89','97.40','96.53','75.00'],['直接模糊','52.66','91.67','77.43','35.42'],['去模糊后','71.01','96.35','93.06','58.33']
    ],'表8  三种输入条件下的最终缺陷识别结果',[3.1,3.0,3.6,3.6,3.5])
    p.paragraph('去模糊使缺陷宏平均 F1 相比直接模糊提高 18.35 个百分点，恢复了由模糊造成性能损失的 60.7%；局部完全正确率由 77.43% 提升至 93.06%，图像完全正确率由 35.42% 提升至 58.33%。其中断线 F1 由 32.00% 回升至 66.67%，松线/浮线 F1 由 25.40% 回升至 54.55%。这两类缺陷高度依赖轨迹跨度和曲率，因而受益于去模糊后的结构恢复。')
    p.figure(WORK/'deblurring'/'media'/'image4.png','图9  清晰、直接模糊与去模糊条件下的三级指标对比',15.5)
    if (ROOT/'tmp_img048_triplet.png').exists():
        p.figure(ROOT/'tmp_img048_triplet.png','图10  端到端样例：模糊输入、去模糊结果与清晰参考',14.0)

    p.heading('6.5 效率分析',2)
    p.paragraph('在 320×320 端到端样例上，MBDNet-P+ 复原质量为 28.6485 dB PSNR、0.8856 SSIM 和 0.0262 MAE。完整三级流水线单幅平均耗时约 55.22 ms；排除首次运行的初始化开销后，稳态平均耗时约 50.69 ms。该速度表明方法具备接近实时处理的潜力，但不同硬件、批量大小、数据传输和图像读写都会影响测量结果，因此正式投稿时应补充 GPU 型号、软件版本、预热次数与计时边界。')

    p.heading('7 讨论',1)
    p.heading('7.1 模块间协同机理',2)
    p.paragraph('三级系统的收益并非三个独立指标的简单叠加。MBDNet-P+ 通过方向尺度路由恢复线迹两侧梯度，使 M6 的边缘分支获得更可靠的局部证据；M6 的定位分支和 AFM 抑制织物纹理，将去模糊可能产生的高频伪影限制在线迹候选区域之外；高分辨率细化进一步修补轨迹断点，为 TSR-Defect 的覆盖率、最大缺口和曲率统计提供稳定输入。最终，同图像参照将残余的整体宽度、亮度和张力差异抵消，从而把判断集中于单元之间的相对异常。')
    p.heading('7.2 可解释性与工程使用',2)
    p.paragraph('TSR-Defect 的每次判决可追溯到槽位边界、软轨迹、参照中位数和分项得分。工程部署时可同时展示原图、概率掩膜、拟合趋势、异常单元和触发特征，使质检人员判断是覆盖不足、跨度缩短、整体斜率还是局部下垂导致告警。若某一产线的线迹单元数或方向发生改变，可优先调整预期槽位数、方向归一化与校准阈值，而无需重新训练完整缺陷分类器。')
    p.heading('7.3 局限性',2)
    p.paragraph('第一，端到端缺陷实验仅包含 48 幅独立源图像，每类阳性数量有限，置信区间较宽；水平翻转虽然扩大了训练变化，但不能视为新增独立样本。第二，模糊由受控线性运动核合成，尚未覆盖真实设备中的空间变化模糊、曝光饱和、滚动快门和多方向振动，因此当前结论不能外推为真实产线上的普遍性能。第三，自参照方法假设同一图像中多数单元正常，当异常连续出现或占据多数槽位时，中位数参照可能被污染。第四，当前基线主要包括 NAFNet 与 MPRNet，正式投稿前还需加入更新的去模糊和细结构分割方法，并在统一硬件与训练预算下复现。第五，M6 预留的中心线、拓扑和附加边界损失尚未启用，不能据此声称拓扑监督收益。')
    p.heading('7.4 后续工作',2)
    p.paragraph('后续将采集具有同步清晰参考或高帧率参考的真实运动模糊线迹数据，扩大织物、线色、针距、照明和速度范围；采用多随机种子报告均值、标准差与置信区间；研究空间变化模糊建模与盲核估计；对连续多缺陷场景引入稳健参照选择或跨帧参考；并进一步量化各阶段的置信度传播，使上游不确定性能够显式影响下游告警阈值。')

    p.heading('8 结论',1)
    p.paragraph('本文面向复杂织物背景、细线迹低占比和运动模糊引起的级联误差，提出由 MBDNet-P+、M6 与 TSR-Defect 组成的三级线迹缺陷检测方法。MBDNet-P+ 通过通用、短水平和长水平分支及空间路由，针对性恢复沿缝制方向退化的结构；M6 通过 Mobile-FPN、边缘—定位解耦、AFM 与高分辨率不确定性引导细化，兼顾轻量性、区域精度和轨迹连续性；TSR-Defect 通过软轨迹、趋势分解、自适应分槽和同图像中位数参照，以可解释方式识别跳针、断线、整体歪斜及松线/浮线。实验结果验证了各阶段及其协同作用：去模糊不仅提高 PSNR 和 SSIM，也显著恢复了最终缺陷宏平均 F1。本文工作为小样本条件下的线迹质量检测提供了一条融合学习型视觉前端与规则型几何后端的技术路线。')

    p.heading('参考文献',1)
    refs=[
        '[1] ZAMIR S W, ARORA A, KHAN S, et al. Multi-stage progressive image restoration[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021: 14821-14831.',
        '[2] CHEN L, CHU X, ZHANG X, et al. Simple baselines for image restoration[C]//European Conference on Computer Vision. Cham: Springer, 2022: 17-33.',
        '[3] RONNEBERGER O, FISCHER P, BROX T. U-Net: Convolutional networks for biomedical image segmentation[C]//Medical Image Computing and Computer-Assisted Intervention. Cham: Springer, 2015: 234-241.',
        '[4] LIN T Y, DOLLÁR P, GIRSHICK R, et al. Feature pyramid networks for object detection[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017: 2117-2125.',
        '[5] LIN T Y, GOYAL P, GIRSHICK R, et al. Focal loss for dense object detection[C]//Proceedings of the IEEE International Conference on Computer Vision. 2017: 2980-2988.',
        '[6] SALEHI S S M, ERDOGMUS D, GHOLIPOUR A. Tversky loss function for image segmentation using 3D fully convolutional deep networks[J]. arXiv preprint arXiv:1706.05721, 2017.',
        '[7] QIN D, LEICHNER C, DELAKIS M, et al. MobileNetV4: Universal models for the mobile ecosystem[J]. arXiv preprint arXiv:2404.10518, 2024.',
        '[8] WANG Z, BOVIK A C, SHEIKH H R, et al. Image quality assessment: From error visibility to structural similarity[J]. IEEE Transactions on Image Processing, 2004, 13(4): 600-612.',
        '[9] KINGMA D P, BA J. Adam: A method for stochastic optimization[C]//International Conference on Learning Representations. 2015.',
        '[10] LOSHCHILOV I, HUTTER F. Decoupled weight decay regularization[C]//International Conference on Learning Representations. 2019.',
        '[11] SHI W, CABALLERO J, HUSZÁR F, et al. Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016: 1874-1883.',
        '[12] IOFFE S, SZEGEDY C. Batch normalization: Accelerating deep network training by reducing internal covariate shift[C]//International Conference on Machine Learning. 2015: 448-456.',
        '[13] WU Y, HE K. Group normalization[C]//European Conference on Computer Vision. Cham: Springer, 2018: 3-19.',
        '[14] MAO X, LI Q, XIE H, et al. Frequency selection for image restoration[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2023, 37(2): 1905-1913.',
        '[15] TSAI F J, PENG Y T, LIN Y Y, et al. Stripformer: Strip transformer for fast image deblurring[C]//European Conference on Computer Vision. Cham: Springer, 2022: 146-162.'
    ]
    for ref in refs:
        q=d.add_paragraph(ref,style='Reference CN'); set_repeat_keep(q)

    d.add_page_break()
    p.heading('附录A 关键配置与复现实验要点',1)
    p.paragraph('为便于后续复现与正式投稿核查，本文初稿中的网络结构依据工程代码和实际配置文件整理。MBDNet-P+ 对应 MBDNetP_arch.py；M6 对应 m6_afm_guided_highres_full_150_seed42.yaml 及其继承配置，AFM 与高分辨率细化分别对应 fusion_head.py 和 c3net.py；TSR-Defect 对应 trajectory_self_reference_detector.py。正式发布代码时建议固定依赖版本、公开数据清单哈希、保存每折源图像标识与阈值，并将训练、验证和测试日志一并归档。')
    p.table(['阶段','关键配置','当前值'],[
        ['MBDNet-P+','尺度通道','64/128/256'],['MBDNet-P+','DWPW/MBD数量','28/4'],['MBDNet-P+','训练阶段','S30+FT10+FT5'],['M6','输入尺寸/参数量','320×320 / 10,267,319'],['M6','优化器/epoch/种子','AdamW / 150 / 42'],['TSR-Defect','线迹单元数/有效列质量阈值','6 / 0.005'],['端到端','分组方式','48源图像、4折、每折36校准+12测试']
    ],'表A1  关键复现配置摘要',[3.2,6.2,6.0])

    p.save()
    print(str(OUT))
    print(f'equations={len(p.equations)}')


if __name__=='__main__':
    build()

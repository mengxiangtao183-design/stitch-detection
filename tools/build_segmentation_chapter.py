from pathlib import Path
from math import ceil

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "segmentation_chapter"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "复杂织物背景下线迹分割网络_中文论文初稿.docx"
FONT_PATH = Path(r"C:\Windows\Fonts\msyh.ttc")


NAVY = (28, 55, 90)
BLUE = (45, 105, 170)
TEAL = (32, 133, 125)
ORANGE = (224, 126, 62)
GOLD = (204, 145, 49)
PURPLE = (120, 91, 160)
GRAY = (92, 102, 112)
LIGHT = (238, 244, 249)
LIGHT_ORANGE = (253, 242, 233)
LIGHT_TEAL = (232, 246, 244)
WHITE = (255, 255, 255)
DARK = (38, 45, 55)


def font(size, bold=False):
    if FONT_PATH.exists():
        return ImageFont.truetype(str(FONT_PATH), size=size, index=0)
    return ImageFont.load_default()


def text_size(draw, text, f):
    box = draw.textbbox((0, 0), text, font=f)
    return box[2] - box[0], box[3] - box[1]


def center_text(draw, box, text, f, fill=DARK, spacing=8):
    x1, y1, x2, y2 = box
    lines = text.split("\n")
    heights = [text_size(draw, line, f)[1] for line in lines]
    total = sum(heights) + spacing * (len(lines) - 1)
    y = y1 + (y2 - y1 - total) / 2
    for line, h in zip(lines, heights):
        w, _ = text_size(draw, line, f)
        draw.text((x1 + (x2 - x1 - w) / 2, y), line, font=f, fill=fill)
        y += h + spacing


def rounded_box(draw, xy, title, subtitle="", fill=LIGHT, outline=BLUE, title_fill=NAVY):
    x1, y1, x2, y2 = xy
    draw.rounded_rectangle(xy, radius=22, fill=fill, outline=outline, width=4)
    h = 48 if subtitle else 0
    if subtitle:
        draw.rounded_rectangle((x1, y1, x2, y1 + h), radius=22, fill=title_fill)
        draw.rectangle((x1, y1 + 24, x2, y1 + h), fill=title_fill)
        center_text(draw, (x1 + 8, y1 + 4, x2 - 8, y1 + h - 4), title, font(25, True), WHITE)
        center_text(draw, (x1 + 12, y1 + h + 5, x2 - 12, y2 - 7), subtitle, font(21), DARK)
    else:
        center_text(draw, (x1 + 10, y1 + 10, x2 - 10, y2 - 10), title, font(24, True), DARK)


def arrow(draw, start, end, color=GRAY, width=5, head=16):
    x1, y1 = start
    x2, y2 = end
    draw.line((x1, y1, x2, y2), fill=color, width=width)
    if abs(x2 - x1) >= abs(y2 - y1):
        s = 1 if x2 >= x1 else -1
        pts = [(x2, y2), (x2 - s * head, y2 - head * .65), (x2 - s * head, y2 + head * .65)]
    else:
        s = 1 if y2 >= y1 else -1
        pts = [(x2, y2), (x2 - head * .65, y2 - s * head), (x2 + head * .65, y2 - s * head)]
    draw.polygon(pts, fill=color)


def label(draw, xy, text, fill=GRAY):
    center_text(draw, xy, text, font(19), fill)


def save_overall():
    img = Image.new("RGB", (2400, 1260), WHITE)
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, 2400, 92), fill=NAVY)
    center_text(d, (0, 10, 2400, 82), "M6：AFM 引导高分辨率线迹分割总体流程", font(38, True), WHITE)

    # Main route
    boxes = [
        ((75, 205, 345, 405), "输入图像", "RGB\n320×320", (229, 239, 249), BLUE),
        ((455, 150, 825, 465), "MobileNetV4", "Hybrid Medium\nC2: 80×80×48\nC3: 40×40×80\nC4: 20×20×160\nC5: 10×10×960", (232, 246, 244), TEAL),
        ((965, 150, 1335, 465), "Mobile-FPN", "1×1 横向投影至 192 通道\n自顶向下跨尺度融合\n输出 P2、P3、P4、P5", (238, 244, 249), BLUE),
        ((1475, 135, 1785, 345), "双分支解码", "边缘分支：P3/P2 细节\n定位分支：P4/P5 语义\nICG 定位增强", (253, 242, 233), ORANGE),
        ((1905, 135, 2245, 345), "AFM 融合", "定位注意力引导边缘特征\n输出粗分割 logits\n及置信度/不确定性先验", (244, 239, 250), PURPLE),
    ]
    for xy, t, st, fill, col in boxes:
        rounded_box(d, xy, t, st, fill, col, col)
    for a, b in [((345, 305), (455, 305)), ((825, 305), (965, 305)), ((1335, 305), (1475, 240)), ((1785, 240), (1905, 240))]:
        arrow(d, a, b)

    rounded_box(d, (835, 710, 1180, 990), "高分辨率细化头", "上采样边缘与定位特征\n融合低维 RGB 细节\n引入 AFM 概率与不确定性\nDW 可分离卷积细化", (232, 246, 244), TEAL, TEAL)
    rounded_box(d, (1410, 730, 1700, 955), "多任务输出", "最终掩码\n边缘预测\n中心线预测（预留）", (238, 244, 249), BLUE, BLUE)
    rounded_box(d, (1900, 730, 2245, 955), "线迹分割结果", "320×320 二值掩码\n供后续检测/测量使用", (253, 242, 233), ORANGE, ORANGE)
    arrow(d, (2075, 345), (2075, 610), PURPLE)
    arrow(d, (2075, 610), (1180, 785), PURPLE)
    label(d, (1800, 485, 2230, 565), "AFM 先验：概率 + 不确定性", PURPLE)
    arrow(d, (1180, 845), (1410, 845), TEAL)
    arrow(d, (1700, 845), (1900, 845), BLUE)

    # Supervision strip
    d.rounded_rectangle((150, 1070, 2250, 1185), radius=18, fill=(246, 248, 251), outline=(170, 184, 198), width=2)
    center_text(d, (180, 1080, 2220, 1175), "训练约束：L = 1.0L_edge + 1.15L_loc + 1.2L_final + 0.2L_final-edge-dice。当前中心线、拓扑与边界辅助损失权重均为 0，作为后续可验证的扩展项保留。", font(23), DARK)
    p = OUT_DIR / "fig1_overall.png"
    img.save(p)
    return p


def save_fpn():
    img = Image.new("RGB", (2400, 1450), WHITE)
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, 2400, 90), fill=NAVY)
    center_text(d, (0, 10, 2400, 80), "Mobile-FPN 跨尺度特征重建与双分支解码", font(38, True), WHITE)
    levels = [
        ("C5", "10×10×960", 1520, 220, 1760, 315),
        ("C4", "20×20×160", 1260, 445, 1550, 560),
        ("C3", "40×40×80", 940, 705, 1290, 845),
        ("C2", "80×80×48", 560, 1010, 970, 1170),
    ]
    for n, s, x1, y1, x2, y2 in levels:
        rounded_box(d, (x1, y1, x2, y2), n, s + "\n1×1 lateral", (232, 246, 244), TEAL, TEAL)
    # FPN circles / state
    pnodes = [
        ("P5", "10×10×192", 1900, 210, 2205, 340),
        ("P4", "20×20×192", 1770, 470, 2110, 610),
        ("P3", "40×40×192", 1540, 745, 1920, 900),
        ("P2", "80×80×192", 1170, 1050, 1580, 1210),
    ]
    for n, s, x1, y1, x2, y2 in pnodes:
        rounded_box(d, (x1, y1, x2, y2), n, s + "\n上采样 + 相加 + 细化", (238, 244, 249), BLUE, BLUE)
    # Links lateral to P
    for src, dst in [((1760, 270),(1900,270)), ((1550,500),(1770,540)), ((1290,775),(1540,820)), ((970,1090),(1170,1130))]:
        arrow(d, src, dst, TEAL)
    # topdown P links
    for src,dst in [((2050,340),(1940,470)),((1940,610),(1740,745)),((1740,900),(1400,1050))]:
        arrow(d,src,dst,BLUE)
    label(d,(1980,360,2330,440),"×2 双线性上采样",BLUE)
    label(d,(1780,630,2140,710),"×2 双线性上采样",BLUE)
    label(d,(1440,920,1810,1010),"×2 双线性上采样",BLUE)
    # decoder branches
    rounded_box(d,(150,560,500,740),"边缘分支", "P3 与 P2 融合\n提取高分辨率边界证据\n输出 128 通道边缘特征",(253,242,233),ORANGE,ORANGE)
    rounded_box(d,(150,855,500,1040),"定位分支", "P4/P5 语义聚合 + ICG\n逐级传递至 P3/P2\n输出 128 通道定位特征",(244,239,250),PURPLE,PURPLE)
    arrow(d,(1170,1130),(500,650),ORANGE)
    arrow(d,(1770,535),(500,945),PURPLE)
    d.rounded_rectangle((150,1240,2220,1340),radius=15,fill=(246,248,251),outline=(170,184,198),width=2)
    center_text(d,(180,1250,2190,1330),"设计要点：深层 P4/P5 提供线迹存在性与上下文定位，浅层 P2/P3 保留窄线迹纹理与边缘。FPN 以统一的 192 通道接口降低多尺度融合的语义鸿沟。",font(23),DARK)
    p=OUT_DIR/"fig2_mobile_fpn.png"; img.save(p); return p


def save_afm_highres():
    img = Image.new("RGB", (2400, 1350), WHITE)
    d = ImageDraw.Draw(img)
    d.rectangle((0,0,2400,90),fill=NAVY)
    center_text(d,(0,10,2400,80),"AFM 引导的高分辨率细化机制",font(38,True),WHITE)
    rounded_box(d,(100,210,475,450),"定位特征 F_loc", "128 通道\n线迹区域语义与上下文",(244,239,250),PURPLE,PURPLE)
    rounded_box(d,(100,650,475,890),"边缘特征 F_edge", "128 通道\n细粒度纹理与边界证据",(253,242,233),ORANGE,ORANGE)
    rounded_box(d,(660,255,1030,475),"AFM", "由定位特征预测空间注意力\n注意力调制边缘特征\n融合得到粗分割 logits",(238,244,249),BLUE,BLUE)
    arrow(d,(475,330),(660,330),PURPLE)
    arrow(d,(475,770),(660,400),ORANGE)
    rounded_box(d,(1190,220,1510,430),"AFM 先验转换", "p = sigmoid(logits)\nu = 4p(1-p)\n概率 p 与不确定性 u",(232,246,244),TEAL,TEAL)
    arrow(d,(1030,365),(1190,325),BLUE)
    rounded_box(d,(660,700,1030,945),"RGB 细节编码", "原图 → 16 通道低维细节\n3×3 卷积 + DW 细化\n仅提供局部外观补偿",(253,242,233),ORANGE,ORANGE)
    rounded_box(d,(1680,255,2240,860),"HighResStitchRefineHead", "① 边缘/定位特征经 1×1 投影至 48 通道\n② 双线性上采样至 320×320\n③ 拼接 RGB 细节、AFM 概率与不确定性\n④ 1×1 融合 + 深度可分离细化\n⑤ 输出最终掩码、边缘与中心线 logits",(232,246,244),TEAL,TEAL)
    arrow(d,(1030,825),(1680,705),ORANGE)
    arrow(d,(475,770),(1680,600),ORANGE)
    arrow(d,(475,330),(1680,475),PURPLE)
    arrow(d,(1510,325),(1680,380),TEAL)
    rounded_box(d,(1900,1030,2240,1185),"最终掩码", "高分辨率\n320×320",(238,244,249),BLUE,BLUE)
    arrow(d,(1960,860),(2070,1030),TEAL)
    d.rounded_rectangle((130,1080,1650,1210),radius=18,fill=(246,248,251),outline=(170,184,198),width=2)
    center_text(d,(165,1090,1615,1200),"协同逻辑：AFM 提供“哪里可能是线迹”的全局定位先验；高分辨率头负责“边界是否贴合、宽度是否稳定、弱线是否连续”的局部重建。概率与不确定性共同避免将不可靠粗预测直接硬性放大。",font(23),DARK)
    p=OUT_DIR/"fig3_afm_highres.png"; img.save(p); return p


def save_training_flow():
    img=Image.new("RGB",(2400,1120),WHITE); d=ImageDraw.Draw(img)
    d.rectangle((0,0,2400,90),fill=NAVY)
    center_text(d,(0,10,2400,80),"训练、验证与部署流程",font(38,True),WHITE)
    stages=[
        ((100,300,430,585),"数据准备","织物图像与二值掩码\n训练集：DB/XJ/SD\n外部测试：RX\n（具体数量待补充）",TEAL),
        ((570,300,900,585),"训练前向","输入图像 → M6 网络\n输出最终分割、边缘预测\n以及预留的中心线预测",BLUE),
        ((1040,220,1390,465),"损失优化","L_edge：边缘约束\nL_loc：定位分支约束\nL_final：焦点/Tversky\nL_edge-dice：边缘 Dice",ORANGE),
        ((1040,650,1390,895),"模型选择","验证集监控 Binary IoU\n保存 model_best.pth\n早停策略待统一说明",PURPLE),
        ((1530,300,1860,585),"统一评估","Binary IoU / F1\nPrecision / Recall\nBoundary F1@1px\nTrajectory F1@1px",TEAL),
        ((2000,300,2330,585),"工业部署","batch=1 单图推理\n输出二值掩码\n供后续检测与测量",BLUE),
    ]
    for xy,t,st,c in stages:
        fill={TEAL:(232,246,244),BLUE:(238,244,249),ORANGE:(253,242,233),PURPLE:(244,239,250)}[c]
        rounded_box(d,xy,t,st,fill,c,c)
    arrow(d,(430,445),(570,445),TEAL); arrow(d,(900,445),(1040,345),BLUE); arrow(d,(1215,465),(1215,650),GRAY); arrow(d,(1390,345),(1530,445),ORANGE); arrow(d,(1860,445),(2000,445),TEAL)
    d.rounded_rectangle((180,960,2220,1040),radius=14,fill=(246,248,251),outline=(170,184,198),width=2)
    center_text(d,(220,970,2180,1030),"说明：RX 仅用于外部泛化测试，不参与训练、阈值扫描或模型选择；所有比较实验应使用相同数据划分、预训练来源、输入尺寸与单图推理设置。",font(22),DARK)
    p=OUT_DIR/"fig4_training_flow.png"; img.save(p); return p


def set_font(run, name="宋体", size=10.5, bold=None, color=None, italic=None):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    run._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    run._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color is not None:
        run.font.color.rgb = RGBColor(*color)
    if italic is not None:
        run.italic = italic


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=80, start=120, bottom=80, end=120):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcMar = tcPr.first_child_found_in("w:tcMar")
    if tcMar is None:
        tcMar = OxmlElement('w:tcMar')
        tcPr.append(tcMar)
    for m, v in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tcMar.find(qn(f"w:{m}"))
        if node is None:
            node = OxmlElement(f"w:{m}")
            tcMar.append(node)
        node.set(qn("w:w"), str(v)); node.set(qn("w:type"), "dxa")


def set_repeat_table_header(row):
    trPr = row._tr.get_or_add_trPr()
    tblHeader = OxmlElement('w:tblHeader')
    tblHeader.set(qn('w:val'), "true")
    trPr.append(tblHeader)


def set_table_geometry(table, widths):
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl = table._tbl
    tblPr = tbl.tblPr
    tblW = tblPr.first_child_found_in("w:tblW")
    tblW.set(qn("w:w"), str(sum(widths))); tblW.set(qn("w:type"), "dxa")
    tblLayout = tblPr.first_child_found_in("w:tblLayout")
    if tblLayout is None:
        tblLayout = OxmlElement("w:tblLayout"); tblPr.append(tblLayout)
    tblLayout.set(qn("w:type"), "fixed")
    ind = OxmlElement("w:tblInd"); ind.set(qn("w:w"), "120"); ind.set(qn("w:type"), "dxa"); tblPr.append(ind)
    grid = tbl.tblGrid
    for gc, width in zip(grid.gridCol_lst, widths):
        gc.set(qn("w:w"), str(width))
    for row in table.rows:
        for cell, width in zip(row.cells, widths):
            tcPr = cell._tc.get_or_add_tcPr()
            tcW = tcPr.find(qn("w:tcW"))
            if tcW is None:
                tcW = OxmlElement("w:tcW"); tcPr.append(tcW)
            tcW.set(qn("w:w"), str(width)); tcW.set(qn("w:type"), "dxa")
            set_cell_margins(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def set_border(paragraph, color="AAB7C4", size="8"):
    p = paragraph._p
    pPr = p.get_or_add_pPr()
    pbdr = OxmlElement('w:pBdr')
    bottom = OxmlElement('w:bottom')
    bottom.set(qn('w:val'), 'single'); bottom.set(qn('w:sz'), size)
    bottom.set(qn('w:space'), '5'); bottom.set(qn('w:color'), color)
    pbdr.append(bottom); pPr.append(pbdr)


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run("第 ")
    set_font(run, size=9)
    fld = OxmlElement('w:fldSimple'); fld.set(qn('w:instr'), 'PAGE')
    paragraph._p.append(fld)
    run = paragraph.add_run(" 页")
    set_font(run, size=9)


def add_para(doc, text="", style=None, before=0, after=6, first=2, align=WD_ALIGN_PARAGRAPH.JUSTIFY, bold_prefix=None):
    p = doc.add_paragraph(style=style)
    p.alignment = align
    pf = p.paragraph_format
    pf.space_before = Pt(before); pf.space_after = Pt(after)
    pf.line_spacing = 1.5
    if first:
        pf.first_line_indent = Cm(0.74 * first)
    if bold_prefix and text.startswith(bold_prefix):
        r = p.add_run(bold_prefix); set_font(r, size=10.5, bold=True)
        r = p.add_run(text[len(bold_prefix):]); set_font(r, size=10.5)
    else:
        r = p.add_run(text); set_font(r, size=10.5)
    return p


def add_heading(doc, text, level=1):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt(13 if level == 1 else 9)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.2
    r = p.add_run(text)
    if level == 1:
        set_font(r, name="黑体", size=15, bold=True, color=NAVY)
    elif level == 2:
        set_font(r, name="黑体", size=12.5, bold=True, color=(31,77,120))
    else:
        set_font(r, name="楷体", size=11, bold=True, color=DARK)
    return p


def add_caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(3); p.paragraph_format.space_after = Pt(8)
    r = p.add_run(text); set_font(r, size=9.5)
    return p


def add_figure(doc, path, caption, width=16.0):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6); p.paragraph_format.space_after = Pt(0)
    p.add_run().add_picture(str(path), width=Cm(width))
    add_caption(doc, caption)


def add_callout(doc, title, text):
    t = doc.add_table(rows=1, cols=1)
    set_table_geometry(t, [9360])
    cell = t.cell(0,0); set_cell_shading(cell, "F4F6F9")
    p = cell.paragraphs[0]; p.paragraph_format.space_before = Pt(2); p.paragraph_format.space_after = Pt(2); p.paragraph_format.line_spacing = 1.35
    r=p.add_run(title); set_font(r,name="黑体",size=10.5,bold=True,color=NAVY)
    r=p.add_run(text); set_font(r,size=10.5)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def add_table(doc, headers, rows, widths, caption):
    table = doc.add_table(rows=1, cols=len(headers))
    set_table_geometry(table, widths)
    row=table.rows[0]; set_repeat_table_header(row)
    for cell, text in zip(row.cells,headers):
        set_cell_shading(cell,"E8EEF5")
        p=cell.paragraphs[0]; p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after=Pt(0); p.paragraph_format.line_spacing=1.15
        r=p.add_run(text); set_font(r,name="黑体",size=9.5,bold=True,color=NAVY)
    for vals in rows:
        cells=table.add_row().cells
        for i,(cell,text) in enumerate(zip(cells,vals)):
            p=cell.paragraphs[0]; p.alignment=WD_ALIGN_PARAGRAPH.CENTER if i != 1 else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_after=Pt(0); p.paragraph_format.line_spacing=1.15
            r=p.add_run(text); set_font(r,size=9.2)
    add_caption(doc,caption)
    return table


def build_doc():
    f1, f2, f3, f4 = save_overall(), save_fpn(), save_afm_highres(), save_training_flow()
    doc=Document()
    sec=doc.sections[0]
    sec.page_width=Cm(21); sec.page_height=Cm(29.7)
    sec.top_margin=Cm(2.5); sec.bottom_margin=Cm(2.5); sec.left_margin=Cm(2.8); sec.right_margin=Cm(2.5)
    sec.header_distance=Cm(1.25); sec.footer_distance=Cm(1.25)
    # Normal style
    normal=doc.styles['Normal']
    normal.font.name='宋体'; normal._element.rPr.rFonts.set(qn('w:eastAsia'),'宋体')
    normal.font.size=Pt(10.5)
    normal.paragraph_format.space_after=Pt(6); normal.paragraph_format.line_spacing=1.5
    # Header/footer
    hp=sec.header.paragraphs[0]; hp.alignment=WD_ALIGN_PARAGRAPH.CENTER
    hr=hp.add_run('复杂织物背景下的线迹分割网络'); set_font(hr,name='宋体',size=9,color=GRAY)
    set_border(hp, color="AAB7C4", size="6")
    fp=sec.footer.paragraphs[0]; add_page_number(fp)

    # Cover / title
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_before=Pt(42); p.paragraph_format.space_after=Pt(14)
    r=p.add_run('复杂织物背景下的线迹分割网络'); set_font(r,name='黑体',size=22,bold=True,color=NAVY)
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_after=Pt(30)
    r=p.add_run('——中文论文分割章节初稿（含待补充实验位置）'); set_font(r,name='楷体',size=13,color=GRAY)
    add_callout(doc,'使用说明：','本文档依据当前 M6 分割网络的实际实现编写，采用“问题定义—网络设计—训练目标—实验方案—结果讨论”的论文式结构。对比实验、消融实验及可视化结果尚未完成的部分以【待补充】标示，后续可在不改动方法论主体的情况下直接填入。')
    add_heading(doc,'摘要',1)
    add_para(doc,'针对织物表面线迹在低对比度、复杂印花、周期织纹和局部反光条件下易发生断裂、边界漂移及背景误检的问题，本文构建了一种面向工业视觉检测的轻量级线迹分割网络。该网络以 MobileNetV4 Hybrid Medium 为编码骨干，通过 Mobile-FPN 完成多尺度语义—细节重建；随后并行建立边缘分支与定位分支，以分别建模线迹的局部轮廓证据和全局存在性语义；在输出端设计高分辨率线迹细化头，并利用注意力融合模块（AFM）产生的粗定位概率与不确定性作为显式先验，实现对细窄线迹宽度、连续性及边界的协同恢复。网络在保持轻量化推理的同时，重点抑制复杂背景纹理对细线迹的干扰。实验部分将从区域重叠、像素分类、边界贴合和轨迹连续性四个维度验证所提方法的有效性。')
    p=doc.add_paragraph(); p.paragraph_format.space_after=Pt(10); p.paragraph_format.line_spacing=1.5
    r=p.add_run('关键词：'); set_font(r,name='黑体',size=10.5,bold=True)
    r=p.add_run('线迹分割；复杂织物背景；MobileNetV4；特征金字塔；高分辨率细化；注意力先验'); set_font(r,size=10.5)

    add_heading(doc,'1 研究背景与问题定义',1)
    add_para(doc,'缝制线迹通常呈现为宽度窄、对比度弱、走向连续但局部弯曲的细长结构。在实际织物图像中，线迹同时受到经纬纹理、刺绣图案、金属反光、褶皱阴影和运动模糊等因素影响，使得“局部边缘强度高”并不等价于“该位置属于线迹”。尤其当线迹与面料颜色相近时，依赖单一尺度纹理或单次上采样的分割网络容易出现两类典型误差：其一是弱线迹被背景吞没，造成漏检和轨迹断裂；其二是印花轮廓或织纹被误识别为线迹，造成假阳性扩张。')
    add_para(doc,'设输入图像为 I∈R^(3×H×W)，目标为预测二值掩码 Ŷ∈[0,1]^(H×W)，其中 Y 表示对应的像素级标注。本文的目标不仅是提高掩码区域重叠度，还要求结果在细线宽度、局部边界与长程连续性上满足后续检测和测量任务的几何需求。因此，网络设计遵循“深层语义负责确认线迹存在性、浅层特征负责恢复边缘细节、输出端负责将二者重建为高分辨率掩码”的总体原则。')
    add_callout(doc,'方法定位：','本文方法是独立的像素级线迹分割模块。其输出为 320×320 的二值概率图/掩码，可直接用于后续线迹检测、缺陷分析或针距测量；去模糊、几何测量等其他模块不参与本节网络的主体定义。')

    add_heading(doc,'2 网络总体结构',1)
    add_para(doc,'所提出的网络以当前性能最优的 M6 配置为基础，整体由轻量编码器、Mobile-FPN 跨尺度重建、边缘—定位双分支解码、AFM 融合以及 AFM 引导的高分辨率细化头组成。输入为 3 通道 320×320 的织物图像。MobileNetV4 Hybrid Medium 输出 C2 至 C5 四级特征，空间分辨率依次为 80×80、40×40、20×20 和 10×10。随后，各尺度特征通过 1×1 横向投影统一至 192 通道，并在自顶向下路径中形成 P2 至 P5 多尺度金字塔。')
    add_figure(doc,f1,'图 1  所提出 M6 线迹分割网络的总体流程图',16.0)
    add_para(doc,'在解码阶段，P2/P3 中较高空间分辨率的特征被送入边缘分支，以保留细窄线迹的纹理与轮廓证据；P4/P5 中较强的语义特征经定位路径聚合，用于判断线迹区域的整体分布。AFM 以定位特征为注意力来源，对边缘特征进行语义约束并输出粗分割结果。与仅将粗分割作为最终输出不同，本文将其概率图及其不确定性图送入高分辨率细化头，以实现定位语义对局部细节恢复的显式引导。')

    add_heading(doc,'2.1 轻量编码与 Mobile-FPN 跨尺度特征重建',2)
    add_para(doc,'原始 C3Net 的表征路线难以同时兼顾工业部署对低延迟的要求与细线迹对高分辨率细节的需求。为此，本文选用 MobileNetV4 Hybrid Medium 作为编码骨干。该骨干以较低计算代价提取分层语义，同时保留对局部纹理变化的响应。设第 l 级编码特征为 C_l，横向投影记为 L_l=φ_l(C_l)，其中 φ_l 为 1×1 卷积、归一化和非线性激活的组合。各级 L_l 通道统一为 192，使跨尺度融合具有固定接口。')
    add_figure(doc,f2,'图 2  Mobile-FPN 跨尺度重建及边缘—定位双分支解码示意图',16.0)
    add_para(doc,'自顶向下融合采用逐级上采样、逐元素相加和轻量细化的方式构建 P_l。其形式可表述为：P_l=ψ_l[L_l+Up(P_(l+1))]，其中 Up(·) 表示 2 倍双线性上采样，ψ_l 为轻量卷积细化算子。该过程将深层的线迹语义逐级传递到 P2/P3，同时避免直接把低分辨率特征硬性放大所引入的空间错位。对于细线迹任务，P4/P5 的主要作用是排除背景干扰并给出粗定位，P2/P3 的主要作用是保留边界形状和局部灰度变化，二者缺一不可。')
    add_table(doc,['特征层','来源','尺寸（输入为 320×320）','通道数','主要作用'],[
        ['C2 / P2','浅层编码 / FPN','80×80','48 / 192','细粒度纹理、边缘与局部宽度'],
        ['C3 / P3','中浅层编码 / FPN','40×40','80 / 192','线迹走向与中尺度轮廓'],
        ['C4 / P4','中深层编码 / FPN','20×20','160 / 192','区域级语义与定位上下文'],
        ['C5 / P5','深层编码 / FPN','10×10','960 / 192','全局语义与背景抑制'],
    ],[1200,1850,2450,1450,2410],'表 1  主要多尺度特征及其功能说明')

    add_heading(doc,'2.2 边缘—定位双分支解码与 AFM 融合',2)
    add_para(doc,'线迹分割的难点在于边缘证据和区域语义存在天然差异。若仅依赖边缘特征，印花边界、织纹和高亮颗粒会造成大量假阳性；若仅依赖深层语义，极细线迹又容易因下采样而被削弱。因此，本文设置并行的边缘分支和定位分支。边缘分支以 P3/P2 为主，输出 128 通道边缘特征 F_edge；定位分支以 P4/P5 语义为起点，经上下文聚合和逐级传递后输出 128 通道定位特征 F_loc。定位路径中的 ICG 保持原始定位增强链路，用于提升区域级前景—背景辨别能力。')
    add_para(doc,'AFM 在两类特征之间建立单向语义引导。首先由 F_loc 生成空间注意力 A，随后以 A 对 F_edge 进行重加权，并通过融合卷积得到粗分割 logits Z_afm。该设计使局部边缘仅在定位语义支持的区域被强化，减少背景纹理在最终掩码中的直接扩张。需要说明的是，AFM 本身是基础网络的融合路线；本文的改进重点不在于重新定义 AFM，而在于将其输出进一步转化为可被高分辨率重建利用的概率—不确定性先验。')

    add_heading(doc,'2.3 AFM 引导的高分辨率线迹细化头',2)
    add_para(doc,'常规融合头在低分辨率特征图上完成预测，再通过上采样恢复到原图尺寸，容易导致细线迹出现宽度不稳、边界锯齿或局部中断。为此，本文在 AFM 输出之后设置 HighResStitchRefineHead。该模块不重新执行重型语义编码，而是将边缘特征与定位特征分别投影至 48 通道，上采样至输出分辨率后，与由 RGB 图像提取的 16 通道低维细节共同融合。这样，细化头能够在保留全局语义约束的前提下恢复局部纹理。')
    add_figure(doc,f3,'图 3  AFM 引导的高分辨率细化头结构图',16.0)
    add_para(doc,'为避免将粗预测中的错误直接放大，AFM 粗分割 logits Z_afm 被转换为概率先验 p=σ(Z_afm) 与不确定性先验 u=4p(1-p)。当 p 接近 0 或 1 时，u 较小，表示粗定位较为确定；当 p 接近 0.5 时，u 较大，表示该区域需要更依赖边缘和 RGB 细节进行校正。最终，细化头将上采样后的 F_edge、F_loc、图像细节、p 和 u 进行拼接，经 1×1 融合与深度可分离卷积细化后，预测最终分割、边缘和中心线 logits。当前正式配置只将最终分割和边缘预测用于优化，中心线输出作为后续连续性监督的结构预留。')

    add_heading(doc,'3 训练目标与优化策略',1)
    add_heading(doc,'3.1 多分支损失函数',2)
    add_para(doc,'训练过程中，网络同时优化边缘分支、定位分支和最终分割输出。总损失定义为：L=λ_eL_edge+λ_lL_loc+λ_fL_final+λ_dL_dice-edge。其中，当前正式配置取 λ_e=1.0、λ_l=1.15、λ_f=1.2 和 λ_d=0.2。最终分割损失以焦点项与偏向召回的 Tversky 项为主体，并叠加边缘区域 Dice 约束，以缓解极细线迹前景像素稀少导致的类别不平衡。')
    add_para(doc,'边缘区域 Dice 仅作用于真值掩码的边界邻域，用于提高边界位置的重叠质量。该项与全局区域损失互补：前者关注线迹轮廓是否贴合，后者关注线迹整体是否被完整分出。对于具有较大纹理干扰的织物背景，该组合可降低模型仅依赖面积统计而忽略边界漂移的风险。')
    add_callout(doc,'当前损失状态：','代码中已实现中心线辅助 Focal+Dice、soft-clDice 拓扑损失与 Boundary-Dice，但 M6 当前正式配置中 centerline_weight=0、topology_weight=0、boundary_weight=0。因此，本稿不将中心线连续性监督表述为已启用创新；如后续开展该方向实验，应单列消融并报告其对 IoU、Recall 与 Trajectory F1@1px 的影响。')
    add_heading(doc,'3.2 数据划分、训练与模型选择',2)
    add_figure(doc,f4,'图 4  网络训练、验证与单图部署流程',16.0)
    add_para(doc,'训练数据由 DB、XJ 和 SD 三个内部数据集组成，RX 保持为纯外部泛化测试集。为确保对比公平性，RX 不参与训练、验证集模型选择、阈值扫描或超参数调节。所有实验使用相同输入尺寸、数据增强方式、预训练来源和测试阈值；模型根据验证集指标保存最佳权重。具体训练轮数、学习率、批大小、优化器及增强参数将在实验完成后按表 2 补充。')
    add_table(doc,['项目','当前设定 / 待补充内容'],[
        ['输入分辨率','320×320'],
        ['训练数据','DB、XJ、SD（具体样本数及划分比例【待补充】）'],
        ['外部测试数据','RX，仅用于外部泛化评估'],
        ['优化器与学习率','【待补充：优化器类型、初始学习率、调度器】'],
        ['批大小与训练轮数','【待补充：batch size、最大 epoch、早停策略】'],
        ['模型选择依据','【待补充：统一为验证集 Binary IoU 或其他主指标，并说明理由】'],
        ['推理设置','batch=1；统计单张图像平均推理时间'],
    ],[2600,6760],'表 2  训练与评估协议（待补充项）')

    add_heading(doc,'4 实验设计与结果呈现（待补充）',1)
    add_heading(doc,'4.1 评价指标',2)
    add_para(doc,'为全面反映模型在细线迹分割任务中的表现，建议以 Binary IoU 作为主指标，并联合报告 Binary F1、Binary Precision、Binary Recall、Boundary F1@1px 和单张推理时间。Binary IoU 衡量预测掩码与真值掩码的区域重叠，是本文的首要评价指标；Binary F1 综合反映查准率与查全率；Precision 用于衡量背景误检程度，Recall 用于衡量弱线迹漏检程度；Boundary F1@1px 在 1 像素容忍范围内评价边界贴合；推理时间则体现模型的工业在线应用潜力。Trajectory F1@1px 可作为补充连续性指标，在主表篇幅受限时可置于附录或专题分析中。')
    add_para(doc,'设 TP、FP、FN 分别表示真阳性、假阳性和假阴性像素，则 Precision=TP/(TP+FP)，Recall=TP/(TP+FN)，F1=2·Precision·Recall/(Precision+Recall)，IoU=TP/(TP+FP+FN)。对于不同数据集，应先分别计算各指标，再给出宏平均值，以避免样本量较大的数据集主导总体结论。')
    add_heading(doc,'4.2 对比实验设置与结果表',2)
    add_para(doc,'对比实验建议选取具有代表性的轻量语义分割网络、医学细结构分割网络和工业表面缺陷分割网络作为参照，并统一使用相同的输入尺寸、训练数据、数据增强和评价脚本。若部分公开方法无法完全复现，应在表注中说明其训练策略、预训练来源或公开权重差异，避免不公平比较。')
    add_table(doc,['方法','骨干 / 关键结构','Binary IoU (%)','Binary F1 (%)','Boundary F1@1px (%)','时间 (ms/张)'],[
        ['【待补充：方法 A】','【待补充】','—','—','—','—'],
        ['【待补充：方法 B】','【待补充】','—','—','—','—'],
        ['【待补充：方法 C】','【待补充】','—','—','—','—'],
        ['本文方法 M6','MobileNetV4 + Mobile-FPN + AFM-guided HighRes','【待填】','【待填】','【待填】','【待填】'],
    ],[1500,3000,1350,1350,1500,660],'表 3  与现有方法的定量对比结果（四数据集宏平均，待补充）')
    add_para(doc,'【结果分析待补充】建议围绕以下逻辑撰写：首先说明本文方法在 IoU 与 Boundary F1 上的相对优势，以证明区域分割与边界贴合的协同提升；其次结合 Precision 与 Recall 分析误检和漏检的变化；最后以单图推理时间讨论工业部署的可行性。若某类方法在特定指标上更优，应从其网络特点、参数规模或任务侧重点出发进行客观解释，避免只报告有利结果。')
    add_heading(doc,'4.3 消融实验设计',2)
    add_para(doc,'为验证各阶段设计的有效性，建议采用逐级累加式消融。以原始 C3Net 为基线，依次引入轻量编码与 Mobile-FPN、高分辨率线迹细化以及 AFM 引导的高分辨率细化。每组实验应保持训练集、随机种子、最大训练轮数、预训练来源和模型选择策略一致，以保证模块贡献具有可归因性。')
    add_table(doc,['编号','模型配置','验证目的','Binary IoU (%)','Binary F1 (%)','Boundary F1@1px (%)','时间 (ms/张)'],[
        ['A0','原始 C3Net','基线性能与效率','【待填】','【待填】','【待填】','【待填】'],
        ['A1','A0 + MobileNetV4 + Mobile-FPN（M0）','验证轻量跨尺度表征重建','【待填】','【待填】','【待填】','【待填】'],
        ['A2','A1 + HighResStitchRefineHead（M4）','验证高分辨率线迹细化','【待填】','【待填】','【待填】','【待填】'],
        ['A3','A2 + AFM-guided HighRes（M6）','验证定位先验引导细化','【待填】','【待填】','【待填】','【待填】'],
    ],[500,2600,2200,900,950,1300,910],'表 4  逐级消融实验结果（待补充）')
    add_para(doc,'【消融分析待补充】A1 用于验证轻量骨干与跨尺度重建能否在降低时延的同时维持或提高基础分割能力；A2 用于说明高分辨率细化对窄线迹宽度、断裂和边界锯齿的修正作用；A3 用于检验 AFM 概率—不确定性先验是否可使细化头获得更可靠的定位约束。应重点报告 A2 与 A3 之间的 Precision/Recall 变化，以论证先验引导不是简单提高阈值，而是改善误检—漏检平衡。')
    add_heading(doc,'4.4 定性可视化与误差分析',2)
    add_para(doc,'建议在该小节插入 4—6 组具有代表性的图像，按照“原图—真值—基线—本文方法—局部放大图”的顺序排布。样本应覆盖同色线迹、复杂印花、周期织纹、反光颗粒、低照度和局部遮挡等情形。局部放大区域可重点展示：①弱线迹是否被补全；②印花/织纹是否被抑制；③线迹边界是否更平滑；④转折或交叠区域是否保持正确宽度。')
    add_table(doc,['图号','建议内容','说明'],[
        ['图 5','典型场景可视化对比','【待插入：原图、真值、基线、M6 预测及局部放大】'],
        ['图 6','复杂背景误差案例','【待插入：织纹、反光或印花导致的误检/漏检分析】'],
        ['图 7','外部数据集 RX 泛化结果','【待插入：至少两组外部样本及文字讨论】'],
    ],[800,2500,4650],'表 5  定性结果图的预留位置')

    add_heading(doc,'5 创新点归纳与小结',1)
    add_para(doc,'本文分割模块可归纳为三个递进式设计。第一，构建轻量跨尺度表征重建网络：以 MobileNetV4 和 Mobile-FPN 替换较重的表征路线，在统一通道接口下实现语义与细节的高效传递。第二，设计高分辨率线迹细化头：针对细线迹在低分辨率预测中易发生断裂、宽度不稳和边界锯齿的问题，将边缘特征、定位特征与低维图像细节在原图尺度进行重建。第三，提出 AFM 引导的高分辨率细化机制：将已有 AFM 的粗分割输出显式转化为概率与不确定性先验，用以约束高分辨率细化过程，形成“先定位、再校正”的协同路径。')
    add_para(doc,'需要强调的是，AFM 本身属于基础融合路线，第三项创新应表述为“利用 AFM 概率—不确定性先验引导高分辨率重建”，而非重复声明 AFM 模块。当前正式训练仅启用最终分割、定位与边缘相关损失，中心线连续性、软拓扑和独立边界损失仍是可开展但尚未验证的扩展方向。后续实验应严格依据统一评估协议补充定量比较、逐级消融和外部泛化分析，从而形成完整、可信的论文证据链。')

    doc.save(OUT_FILE)
    print(OUT_FILE)


if __name__ == '__main__':
    build_doc()

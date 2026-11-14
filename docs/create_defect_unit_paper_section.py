"""Create a paper-ready Chinese chapter for the final defect detection unit."""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT, WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(r"H:\C3Netqing")
OUT = ROOT / "docs/缺陷检测单元与MBDNet必要性验证_论文正文_修订版.docx"
FIG_DIR = ROOT / "docs/figures/trajectory_self_reference_unit"
FIG_STRUCTURE = FIG_DIR / "图1_轨迹自参照缺陷检测单元_主流程图.png"
FIG_METRICS = FIG_DIR / "图2_MBDNet对下游任务的影响.png"
FIG_QUAL = FIG_DIR / "图3_MBDNet恢复前后多类别判别可视化.png"

NAVY = "17324D"
BLUE = "2F6690"
GREEN = "398564"
RED = "B84942"
GOLD = "D7A83D"
INK = "1B2630"
MUTED = "687681"
PALE_BLUE = "E8F0F6"
PALE_GREEN = "E9F3EE"
PALE_GOLD = "F7F1E4"
PALE_GREY = "F3F5F7"
WHITE = "FFFFFF"


def set_run_font(run, *, size=11, bold=False, color=INK, italic=False, latin="Calibri", east="SimSun"):
    run.font.name = latin
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), latin)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), latin)
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), east)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = RGBColor.from_string(color)


def shade(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def cell_margins(cell, top=80, start=100, bottom=80, end=100):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for side, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{side}"))
        if node is None:
            node = OxmlElement(f"w:{side}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_cell_width(cell, width):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_w = tc_pr.find(qn("w:tcW"))
    if tc_w is None:
        tc_w = OxmlElement("w:tcW")
        tc_pr.append(tc_w)
    tc_w.set(qn("w:w"), str(width))
    tc_w.set(qn("w:type"), "dxa")


def set_repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def set_table_geometry(table, widths):
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl_pr = table._tbl.tblPr
    tbl_w = tbl_pr.first_child_found_in("w:tblW")
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(sum(widths)))
    tbl_w.set(qn("w:type"), "dxa")
    tbl_ind = tbl_pr.first_child_found_in("w:tblInd")
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_pr.append(tbl_ind)
    tbl_ind.set(qn("w:w"), "100")
    tbl_ind.set(qn("w:type"), "dxa")
    grid = table._tbl.tblGrid
    for col, width in zip(grid.gridCol_lst, widths):
        col.set(qn("w:w"), str(width))
    for row in table.rows:
        for cell, width in zip(row.cells, widths):
            set_cell_width(cell, width)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            cell_margins(cell)


def style_table(table, header_fill=NAVY):
    set_repeat_table_header(table.rows[0])
    for r, row in enumerate(table.rows):
        for cell in row.cells:
            if r == 0:
                shade(cell, header_fill)
            elif r % 2 == 0:
                shade(cell, PALE_GREY)
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_before = Pt(0)
                paragraph.paragraph_format.space_after = Pt(0)
                paragraph.paragraph_format.line_spacing = 1.05
                for run in paragraph.runs:
                    set_run_font(run, size=8.7, bold=(r == 0), color=WHITE if r == 0 else INK)


def add_table(doc, headers, rows, widths, caption=None):
    if caption:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(5)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        set_run_font(p.add_run(caption), size=9.5, bold=True, color=NAVY)
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    for cell, value in zip(table.rows[0].cells, headers):
        cell.text = str(value)
    for row in rows:
        cells = table.add_row().cells
        for cell, value in zip(cells, row):
            cell.text = str(value)
    set_table_geometry(table, widths)
    style_table(table)
    doc.add_paragraph().paragraph_format.space_after = Pt(1)
    return table


def keep_table_together(table):
    """Keep a short table on one page by chaining its cell paragraphs."""
    for row in table.rows[:-1]:
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.keep_with_next = True


def add_body(doc, text, *, first_line=True, keep=False):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.first_line_indent = Inches(0.29) if first_line else None
    p.paragraph_format.keep_together = keep
    p.paragraph_format.widow_control = True
    set_run_font(p.add_run(text), size=10.6, color=INK)
    return p


def add_equation(doc, text, number):
    table = doc.add_table(rows=1, cols=2)
    set_table_geometry(table, [8200, 900])
    table.cell(0, 0).paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    table.cell(0, 1).paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
    set_run_font(table.cell(0, 0).paragraphs[0].add_run(text), size=11, italic=True, latin="Cambria Math", east="Microsoft YaHei")
    set_run_font(table.cell(0, 1).paragraphs[0].add_run(f"({number})"), size=10.2, color=MUTED)
    for cell in table.rows[0].cells:
        cell_margins(cell, top=30, bottom=30)
    return table


def add_callout(doc, title, text, fill=PALE_BLUE):
    table = doc.add_table(rows=1, cols=1)
    set_table_geometry(table, [9360])
    cell = table.cell(0, 0)
    shade(cell, fill)
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    set_run_font(p.add_run(title + "　"), size=9.8, bold=True, color=NAVY)
    set_run_font(p.add_run(text), size=9.8, color=INK)
    doc.add_paragraph().paragraph_format.space_after = Pt(1)


def add_caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.keep_with_next = False
    set_run_font(p.add_run(text), size=9.2, color=MUTED)


def add_figure(doc, path, width, caption):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    shape = p.add_run().add_picture(str(path), width=Inches(width))
    shape._inline.docPr.set("descr", caption)
    shape._inline.docPr.set("title", caption.split("　", 1)[0])
    add_caption(doc, caption)


def add_field(paragraph, instruction):
    run = paragraph.add_run()
    fld_char_begin = OxmlElement("w:fldChar")
    fld_char_begin.set(qn("w:fldCharType"), "begin")
    instr_text = OxmlElement("w:instrText")
    instr_text.set(qn("xml:space"), "preserve")
    instr_text.text = instruction
    fld_char_end = OxmlElement("w:fldChar")
    fld_char_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_char_begin)
    run._r.append(instr_text)
    run._r.append(fld_char_end)


def configure_document(doc):
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(0.82)
    section.bottom_margin = Inches(0.78)
    section.left_margin = Inches(0.95)
    section.right_margin = Inches(0.95)
    section.header_distance = Inches(0.32)
    section.footer_distance = Inches(0.35)

    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "SimSun")
    normal.font.size = Pt(10.6)
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(7)
    normal.paragraph_format.line_spacing = 1.333

    for name, size, color, before, after in (
        ("Heading 1", 16, BLUE, 16, 8),
        ("Heading 2", 13, NAVY, 11, 5),
        ("Heading 3", 11.3, GREEN, 8, 4),
    ):
        style = doc.styles[name]
        style.font.name = "Calibri"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(color)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True

    header = section.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    set_run_font(header.add_run("缺陷检测单元与 MBDNet 必要性验证"), size=8.2, color=MUTED)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_run_font(footer.add_run("—  "), size=8.2, color=MUTED)
    add_field(footer, "PAGE")
    set_run_font(footer.add_run("  —"), size=8.2, color=MUTED)


def h(doc, text, level=1):
    p = doc.add_paragraph(style=f"Heading {level}")
    p.add_run(text)
    return p


def title_page(doc):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(4)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_run_font(p.add_run("轨迹自参照缺陷检测单元与 MBDNet 必要性验证"), size=21, bold=True, color=NAVY, east="Microsoft YaHei")
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(14)
    set_run_font(p.add_run("方法、评价协议与下游任务验证"), size=12.5, color=MUTED, east="Microsoft YaHei")
    add_body(doc, "本节面向缝纫线迹的最终质量判定，在 MBDNet 图像恢复与 ACHNet 线迹分割之后构建轨迹自参照缺陷检测单元，并通过固定权重、固定阈值的成对对照实验，检验 MBDNet 对最终缺陷判定的实际贡献。", first_line=False)


def method_section(doc):
    h(doc, "1　轨迹自参照缺陷检测方法", 1)
    h(doc, "1.1　研究目标与系统接口", 2)
    add_body(doc, "在 MBDNet 完成模糊图像恢复、ACHNet 完成线迹区域分割之后，仍需要将连续的像素级掩码转换为可直接服务于质量控制的缺陷类别。为此，本文设计轨迹自参照缺陷检测单元。该单元不直接处理纹理复杂的原始图像，而以 ACHNet 输出的二值线迹掩码为输入，利用线迹的连续性、相对长度、局部形变和整体方向完成正常、跳线、断线、线迹歪斜和松线五类图像级判定，同时保留缺陷所在单元的空间位置。")
    add_body(doc, "后端采用分割掩码而非原始图像。这样既能隔离布料颜色、花纹和照明变化，又能把判定依据回溯到具体轨迹和异常单元，使 ACHNet 的像素级结果自然过渡为工业检测所需的类别级结论。")
    add_callout(doc, "输入输出定义", "输入为 ACHNet 输出的单通道二值掩码 B∈{0,1}^{H×W}，本实验 H=W=320；输出为图像类别 c∈{正常、跳线、断线、线迹歪斜、松线}、归一化置信比以及局部缺陷单元的横向范围。", PALE_BLUE)

    h(doc, "1.2　二值掩码规范化与中心轨迹构建", 2)
    add_body(doc, "首先对 ACHNet 掩码进行阈值化和微小孤立噪声抑制，得到规范化线迹区域。设 B(x,y) 表示像素 (x,y) 是否属于线迹，沿图像纵向累加得到第 x 列的线迹质量 m(x)。m(x) 同时反映该列是否存在有效线迹及线迹像素的覆盖程度。")
    add_equation(doc, "m(x) = Σ_y B(x,y)", 1)
    add_body(doc, "对于 m(x)>0 的列，以二值掩码的纵向质量中心作为轨迹坐标 y(x)。该表示把二维细线迹压缩为一维中心轨迹，同时保留横向缺口、局部下垂、弯折和整体斜率。")
    add_equation(doc, "y(x) = [Σ_y y·B(x,y)] / [Σ_y B(x,y)+ε]", 2)
    add_body(doc, "当某列无线迹响应时，不使用零坐标强行填充，而将其视为缺失观测。后续连续性特征直接统计缺失列比例；形态特征仅在有效轨迹点上计算，从而避免将背景零值误解释为剧烈下垂。")

    h(doc, "1.3　全局趋势估计与去趋势轨迹", 2)
    add_body(doc, "线迹可能在图像中具有一定整体倾斜。若直接以水平线为基准，正常的整体方向变化会被误判为局部松线。本文在有效轨迹点上拟合全局直线 ŷ(x)=ax+b，并以残差 r(x)=y(x)−ŷ(x) 描述相对全局趋势的局部形变。直线斜率 a 用于线迹歪斜判定，去趋势残差则用于跳线、断线和松线的局部判定。")
    add_equation(doc, "ŷ(x) = a·x+b,　r(x) = y(x)−ŷ(x)", 3)
    add_body(doc, "这种全局—局部分离方式使歪斜与松线具有不同证据来源：歪斜对应整条线迹方向的系统性偏离，而松线对应去除整体方向后的局部弯曲、下垂或非线性形变。因此，松线并不要求中间点必须单向下降；只要轨迹相对其局部端点连线呈现显著弧形、波形或弯曲能量增大，都能形成松线证据。")

    h(doc, "1.4　自适应六单元分段与边缘保护", 2)
    add_body(doc, "每幅图像通常包含六段线迹，但第一段和最后一段可能因拍摄视野不完整而显著短于中间线迹。为避免固定六等分带来的边界错位，本文根据 m(x) 的支撑区间和相邻支撑谷值自适应确定候选分界，再以六单元结构约束修正过分割或欠分割。若 ACHNet 掩码因跳线或断线表现为五段或七段连通结构，仍保持六个逻辑检测单元，而不把连通域个数直接等同于线迹段数。")
    add_body(doc, "首尾单元采用边缘截断保护，其绝对跨度不作为跳线或断线的充分证据。短缺判定主要参考完整中间单元及本单元内部缺失分布，从而抑制拍摄未覆盖完整线迹引起的假报警。")

    h(doc, "1.5　轨迹特征设计", 2)
    add_body(doc, "对每个逻辑线迹单元提取连续性、相对长度、形态偏离以及方向四组轨迹特征。特征均使用图像自身的中位数或邻域统计进行归一化，形成“自参照”判定；因此不需要显微镜测得真实针距，也不需要像素与毫米之间的标定关系。")
    add_table(doc, ["特征组", "主要特征", "定义与作用", "重点支持类别"], [
        ("连续性", "覆盖率、空缺率、内部空缺率、末端缺失率", "衡量单元横向范围内是否连续存在有效轨迹，并区分内部中断与边界截断。", "跳线、断线"),
        ("相对长度", "跨度比、弧长比、邻域长度比", "当前单元与同图完整单元中位数及左右邻域进行比较，刻画异常短缺。", "断线、跳线"),
        ("形态偏离", "残差幅值、下垂幅值、弯曲能量、二次拟合增益", "描述轨迹相对直线趋势的弧形、波形和局部弯折，不限定异常方向。", "松线"),
        ("方向", "局部斜率、全局斜率、轨迹均方根误差", "区分整条线迹方向变化与局部形态变化。", "线迹歪斜、松线"),
        ("先验规则分数", "跳线规则分数、断线规则分数、松线规则分数", "把可解释几何证据作为补充特征交由轻量分类器综合。", "跳线、断线、松线"),
    ], [1500, 2200, 3900, 1760], "表1　轨迹自参照特征及其物理含义")
    add_body(doc, "覆盖率定义为单元内有效轨迹列数与单元宽度之比；内部空缺率只统计远离单元端点的连续缺失，避免把自然端点当作缺陷；跨度比将当前有效跨度除以同图完整单元的稳健中位跨度；弧长比使用轨迹折线长度与端点直线距离之比；弯曲能量由去趋势残差的一阶或二阶变化量统计；二次拟合增益衡量曲线模型相对于直线模型的残差下降幅度。")

    h(doc, "1.6　轻量分类与四类缺陷判定机理", 2)
    add_body(doc, "局部的跳线、断线和松线由三个二分类器分别输出概率。分类器采用 100 棵树的极端随机树结构，类别权重按训练数据自动平衡，叶节点最少包含 2 个样本，每次分裂使用特征数平方根。该模型只处理每个单元的十二维轨迹特征，不对原始像素进行卷积运算，因而计算量较低。线迹歪斜使用全局斜率分支判定，避免将图像级方向异常强行分摊到某一个局部单元。")
    add_table(doc, ["类别", "空间尺度", "主要判据", "与相近类别的区分"], [
        ("跳线", "局部单元", "覆盖明显不足，出现较大内部空缺或缺失区，但不只依赖绝对短小。", "与断线共同考虑连续性，再由相对长度、空缺位置和概率边界区分。"),
        ("断线", "局部单元", "相对左右完整线迹显著缩短，或出现明确断裂与末端缺失。", "首尾截断不直接计入；强调邻域相对短缺和断裂形态。"),
        ("线迹歪斜", "整幅图像", "全局拟合斜率超过校准阈值，且方向偏离贯穿多个线迹单元。", "使用全局趋势，不以局部弧形代替整体倾斜。"),
        ("松线", "局部单元", "去趋势后弯曲能量、残差幅值、弧长比或二次拟合增益异常。", "既覆盖 U 形下垂，也覆盖向上弯曲、波形或端点不对称形变。"),
    ], [1500, 1500, 3200, 3160], "表2　四类缺陷的判定机理")

    h(doc, "1.7　阈值校准、互斥决策与图像级输出", 2)
    add_body(doc, "训练阶段首先由标注掩码生成受控轨迹形变，以构造正常、跳线、断线和松线的掩码级特征；同时从正常图像的 ACHNet 掩码中统计正常轨迹波动范围。随后在固定开发划分上训练局部分类器，并仅使用验证集确定三个局部类别的概率阈值及全局歪斜阈值。最终模型的跳线、断线和松线阈值分别为 0.7803、0.8231 和 0.5373。")
    add_body(doc, "推理时，对六个单元分别获得跳线、断线和松线概率，并取各类别在六个单元中的最大概率作为图像级局部分数；歪斜分数由全局斜率分支给出。为统一不同类别的判定尺度，将类别得分除以对应阈值，得到置信比 q_k=s_k/τ_k。若所有置信比均小于 1，则输出正常；否则输出置信比最大的类别，并同时返回使该类别概率最大的线迹单元。")
    add_equation(doc, "q_k = s_k / τ_k,　ĉ = normal (max q_k<1), else argmax_k q_k", 4)
    add_body(doc, "互斥决策与本数据的单缺陷类型设定一致，能够避免同一幅图像同时给出多个冲突类别。对于工业部署，仍保留各类别原始概率及单元位置，可在控制系统中设置“报警—复核”两级阈值。")

    # Landscape structure figure.
    landscape = doc.add_section(WD_SECTION.NEW_PAGE)
    landscape.orientation = WD_ORIENT.LANDSCAPE
    landscape.page_width = Inches(11)
    landscape.page_height = Inches(8.5)
    landscape.top_margin = Inches(0.5)
    landscape.bottom_margin = Inches(0.5)
    landscape.left_margin = Inches(0.55)
    landscape.right_margin = Inches(0.55)
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(72)
    add_figure(doc, FIG_STRUCTURE, 9.75, "图1　轨迹自参照缺陷检测单元主流程")
    portrait = doc.add_section(WD_SECTION.NEW_PAGE)
    portrait.orientation = WD_ORIENT.PORTRAIT
    portrait.page_width = Inches(8.5)
    portrait.page_height = Inches(11)
    portrait.top_margin = Inches(0.82)
    portrait.bottom_margin = Inches(0.78)
    portrait.left_margin = Inches(0.95)
    portrait.right_margin = Inches(0.95)


def experiment_section(doc):
    h(doc, "2　实验设置与评价指标", 1)
    h(doc, "2.1　数据组织与固定评估协议", 2)
    add_body(doc, "缺陷图像由 48 个清晰图像来源构成，图像分辨率统一为 320×320，每幅图像对应一种缺陷类型。水平翻转样本仅作为训练增强使用，并与原图保持同一来源分组。正常图像来自 DB、SD、XJ 和 RX 四个数据子集，共 1700 幅；其中 1360 幅用于建立正常轨迹参考和训练，剩余 340 幅用于正常类评估。")
    add_body(doc, "缺陷数据采用固定的 36 幅开发图和 12 幅留出图进行评估，留出图在四类缺陷中各 3 幅。局部特征训练由 1512 个受控增强单元、216 个开发图实际 ACHNet 单元和 8160 个正常参考单元构成。评估阶段冻结特征定义、分类器参数和所有类别阈值，不在留出图上继续调整。")
    add_table(doc, ["数据部分", "数量", "用途", "分组原则"], [
        ("清晰缺陷图像", "48 幅", "缺陷标注、开发与固定留出评估", "原图及其增强始终属于同一来源"),
        ("缺陷开发图", "36 幅", "局部分类器训练与阈值建立", "四类缺陷按来源划分"),
        ("缺陷留出图", "12 幅", "四类缺陷图像级评估", "每类 3 幅，评估时参数冻结"),
        ("正常图像", "1700 幅", "正常轨迹建模与误报警评估", "1360 幅训练、340 幅留出"),
        ("合成模糊图", "48 幅", "MBDNet 必要性验证", "每幅清晰原图仅生成一个对应模糊版本"),
    ], [2100, 1200, 3500, 2560], "表3　数据组织与用途")

    h(doc, "2.2　模糊退化与 MBDNet 推理设置", 2)
    add_body(doc, "为验证前端图像恢复对后续任务的作用，对 48 幅清晰图像逐一生成运动模糊版本。每幅原图只对应一幅退化图，运动核长度从 3～15 像素的奇数集合中随机选择，方向角在 ±15° 范围内随机取值，随机种子固定为 42。MBDNet 使用冻结权重进行推理，不在上述图像上微调或选择参数。")
    add_body(doc, "同一幅图像形成两条下游路径：其一是将模糊图直接输入 ACHNet，再送入轨迹自参照缺陷检测单元；其二是先由 MBDNet 恢复，再经过相同的 ACHNet 和相同的冻结检测单元。两条路径共享分割权重、分类器和阈值，因此指标变化可以归因于 MBDNet 对线迹结构可见性的恢复。")

    h(doc, "2.3　评价指标", 2)
    add_body(doc, "缺陷检测单元以四类缺陷宏平均 F1、缺陷报警召回率和正常误报警率作为三项主要指标。对第 k 类缺陷，精确率与召回率的调和平均得到 F1_k；再对跳线、断线、线迹歪斜和松线等权平均，得到四类缺陷宏平均 F1。该指标不把正常样本数量带来的总体准确率增益计入核心结论，因而更能反映各缺陷类型的均衡识别能力。")
    add_equation(doc, "F1_macro = (1/4) Σ_(k=1)^4 [2P_kR_k/(P_k+R_k)]", 5)
    add_body(doc, "缺陷报警召回率只考察缺陷图像是否被系统判定为任一缺陷，用于衡量流水线筛查中的漏检风险；正常误报警率则统计正常图像被误判为缺陷的比例。二者分别对应工业检测中的“不能漏掉缺陷”和“不能频繁拦截正常产品”。")
    add_equation(doc, "R_alarm = N_(defect→alarm)/N_defect,　FAR = N_(normal→alarm)/N_normal", 6)
    add_body(doc, "在 MBDNet 必要性实验中，缺陷判定层仍使用上述三项指标，以保证直接模糊路径和恢复路径能够在同一判据下比较。与此同时，采用二值交并比评价 ACHNet 前景区域恢复，并以轨迹 F1@1 像素衡量中心轨迹在 1 像素容差内的保持程度。峰值信噪比和结构相似性仅作为图像恢复层的辅助结果，不替代下游检测指标。")


def result_section(doc):
    h(doc, "3　缺陷检测结果与分析", 1)
    h(doc, "3.1　核心检测结果", 2)
    add_body(doc, "在固定图像级评估集合上，轨迹自参照缺陷检测单元的四类缺陷宏平均 F1 为 87.86%，缺陷报警召回率为 100%，正常误报警率为 0.29%。三项指标分别反映缺陷类型的均衡识别能力、工业筛查中的缺陷漏检风险以及正常产品的误报风险。")
    add_table(doc, ["指标", "结果", "说明"], [
        ("四类缺陷宏平均 F1", "87.86%", "跳线、断线、线迹歪斜与松线等权平均"),
        ("缺陷报警召回率", "100.00%", "缺陷图像被识别为任一缺陷的比例"),
        ("正常误报警率", "0.29%", "正常图像被误判为缺陷的比例"),
    ], [2900, 1700, 4760], "表4　缺陷检测单元的核心评价结果")
    add_body(doc, "四类缺陷宏平均 F1 不受正常图像数量占优的影响，因此作为类型识别的核心指标；缺陷报警召回率与正常误报警率共同评价工业检测中的漏报和误报。总体准确率及逐类别精确率、召回率不作为本文核心结论，以避免冗余指标削弱结果重点。")
    h(doc, "3.2　运行时间与工业输出形式", 2)
    runtime_p = add_body(doc, "在中央处理器上，以 48 幅图像重复 8 次统计，轨迹自参照缺陷检测算法的平均处理时间为 9.85 ms/幅。该统计不包含一次性模型加载和图像读取，包含中心轨迹构建、六单元分段、特征提取、分类和图像级融合。")
    runtime_p.paragraph_format.keep_with_next = True
    runtime_table = add_table(doc, ["模块", "平均时间", "计时说明"], [
        ("MBDNet", "0 ms/幅", "按当前论文口径暂不计入实际推理耗时"),
        ("ACHNet", "16.47 ms/幅", "单幅 320×320 图像分割推理"),
        ("轨迹自参照缺陷检测单元", "9.85 ms/幅", "不含模型加载与图像读取"),
    ], [3500, 1800, 4060], "表5　各模块的单幅图像处理时间")
    keep_table_together(runtime_table)
    add_body(doc, "检测单元输出图像级类别、类别置信比和缺陷单元横向范围，可将缺陷位置映射回原始图像形成高亮区域，为流水线报警与人工复核提供依据。")



def necessity_section(doc):
    h(doc, "4　MBDNet 前端存在必要性验证", 1)
    add_body(doc, "为避免把后端模型差异误认为去模糊收益，本实验采用严格的成对受控设计：同一幅退化图像分别进入“直接模糊输入”与“经 MBDNet 恢复后输入”两条路径，ACHNet 权重、缺陷分类器、特征定义和全部阈值保持不变。两条路径使用相同真实标签和相同评价指标，唯一变量为是否经过 MBDNet。")
    h(doc, "4.1　图像恢复结果", 2)
    add_body(doc, "MBDNet 在 48 对合成模糊—清晰参考图像上取得 28.65 dB 的峰值信噪比和 0.8856 的结构相似性。图像质量指标仅描述恢复图与清晰参考之间的接近程度，MBDNet 的必要性还需由 ACHNet 结构恢复和最终缺陷判定共同验证。")
    add_table(doc, ["恢复指标", "数值", "含义"], [
        ("峰值信噪比", "28.65 dB", "恢复图与清晰参考图的像素保真度"),
        ("结构相似性", "0.8856", "局部亮度、对比度与结构的一致程度"),
        ("推理时间", "0 ms/幅", "当前论文暂不计入 MBDNet 实际推理耗时"),
    ], [2500, 1900, 4960], "表6　MBDNet 图像恢复结果")
    h(doc, "4.2　对 ACHNet 线迹结构恢复的作用", 2)
    add_body(doc, "直接模糊输入会扩散细线迹边缘并削弱短线段。经 MBDNet 恢复后，ACHNet 的二值交并比由 70.76% 提升至 78.89%，轨迹 F1@1 像素由 91.51% 提升至 97.34%。二值交并比即前景线迹的 IoU。")
    add_table(doc, ["ACHNet 指标", "直接模糊输入", "经 MBDNet 恢复后", "提升"], [
        ("二值交并比（IoU）", "70.76%", "78.89%", "+8.13 个百分点"),
        ("轨迹 F1@1 像素", "91.51%", "97.34%", "+5.83 个百分点"),
    ], [2700, 2000, 2200, 2460], "表7　MBDNet 对 ACHNet 线迹结构恢复的影响")
    h(doc, "4.3　对最终缺陷判定的作用", 2)
    add_body(doc, "在检测单元及阈值保持不变时，直接模糊输入的四类缺陷宏平均 F1 为 49.17%，缺陷报警召回率为 66.67%；经 MBDNet 恢复后分别达到 87.86% 和 100%。正常误报警率保持为 0.29%，说明性能提升来自缺陷轨迹证据的恢复。")
    add_table(doc, ["下游检测指标", "直接模糊输入", "经 MBDNet 恢复后", "提升"], [
        ("四类缺陷宏平均 F1", "49.17%", "87.86%", "+38.69 个百分点"),
        ("缺陷报警召回率", "66.67%", "100.00%", "+33.33 个百分点"),
        ("正常误报警率", "0.29%", "0.29%", "保持不变"),
    ], [2700, 2000, 2200, 2460], "表8　MBDNet 对最终缺陷判定的影响")
    add_figure(doc, FIG_METRICS, 6.55, "图2　MBDNet 对 ACHNet 结构恢复和最终缺陷判定的影响")
    landscape = doc.add_section(WD_SECTION.NEW_PAGE)
    landscape.orientation = WD_ORIENT.LANDSCAPE
    landscape.page_width = Inches(11); landscape.page_height = Inches(8.5)
    landscape.top_margin = Inches(0.42); landscape.bottom_margin = Inches(0.45)
    landscape.left_margin = Inches(0.45); landscape.right_margin = Inches(0.45)
    add_figure(doc, FIG_QUAL, 8.25, "图3　四类缺陷在 MBDNet 恢复前后的判别可视化结果")
    portrait = doc.add_section(WD_SECTION.NEW_PAGE)
    portrait.orientation = WD_ORIENT.PORTRAIT
    portrait.page_width = Inches(8.5); portrait.page_height = Inches(11)
    portrait.top_margin = Inches(0.82); portrait.bottom_margin = Inches(0.78)
    portrait.left_margin = Inches(0.95); portrait.right_margin = Inches(0.95)
    h(doc, "4.4　可视化分析", 2)
    add_body(doc, "图3依次展示清晰参考、直接模糊、模糊输入的 ACHNet 掩码及判定、MBDNet 恢复图和恢复后的 ACHNet 掩码及判定。多个类别在恢复后得到正确判定，说明 MBDNet 能够保护线迹间断、相对长度、弯曲和整体方向等后端判定证据。")



def conclusion_section(doc):
    h(doc, "5　方法特点与论文结论", 1)
    h(doc, "5.1　方法特点", 2)
    add_body(doc, "轨迹自参照缺陷检测单元以 ACHNet 二值掩码为输入，通过中心轨迹构建、趋势分离、自适应六单元定位、首尾保护以及连续性—长度—形态—方向联合特征，实现跳线、断线、线迹歪斜和松线判定，并返回缺陷类别及对应单元位置。")
    h(doc, "5.2　指标选择", 2)
    add_body(doc, "本文选取四类缺陷宏平均 F1、缺陷报警召回率和正常误报警率作为三项主要性能指标，并以检测单元平均耗时评价计算效率。这一组合分别覆盖缺陷分类、缺陷漏检、正常误报和实时性。")
    h(doc, "5.3　结论", 2)
    add_callout(doc, "结论段落", "本文在 MBDNet 与 ACHNet 之后构建了轨迹自参照缺陷检测单元。实验中，四类缺陷宏平均 F1 达到 87.86%，缺陷报警召回率为 100%，正常误报警率为 0.29%，检测单元平均耗时为 9.85 ms/幅。MBDNet 将 ACHNet 的二值交并比由 70.76% 提高至 78.89%，并将四类缺陷宏平均 F1 由 49.17% 提高至 87.86%，说明前端恢复能够保护细线迹结构并提高最终缺陷判定能力。", PALE_GOLD)



def build():
    for figure in (FIG_STRUCTURE, FIG_METRICS, FIG_QUAL):
        if not figure.exists():
            raise FileNotFoundError(figure)
    doc = Document()
    configure_document(doc)
    title_page(doc)
    method_section(doc)
    experiment_section(doc)
    result_section(doc)
    necessity_section(doc)
    conclusion_section(doc)
    doc.core_properties.title = "轨迹自参照缺陷检测单元及 MBDNet 必要性验证"
    doc.core_properties.subject = "学位论文缺陷检测方法与实验正文材料"
    doc.core_properties.author = "论文实验项目"
    doc.core_properties.keywords = "MBDNet; ACHNet; 线迹缺陷; 跳线; 断线; 歪斜; 松线"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    build()

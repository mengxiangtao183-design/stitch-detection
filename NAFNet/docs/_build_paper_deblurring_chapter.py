from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_ALIGN_VERTICAL, WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


ROOT = Path(r"H:\C3Netqing")
DOC_DIR = ROOT / "NAFNet" / "docs"
FIG_DIR = DOC_DIR / "paper_deblurring_figures"
OUT = DOC_DIR / "论文去模糊网络章节_MBDNetPPlus.docx"


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=70, start=80, bottom=70, end=80):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for m, v in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{m}"))
        if node is None:
            node = OxmlElement(f"w:{m}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(v))
        node.set(qn("w:type"), "dxa")


def set_table_borders(table, header=True):
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.first_child_found_in("w:tblBorders")
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "bottom"):
        el = borders.find(qn(f"w:{edge}"))
        if el is None:
            el = OxmlElement(f"w:{edge}")
            borders.append(el)
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "12")
        el.set(qn("w:color"), "000000")
    for edge in ("left", "right", "insideV", "insideH"):
        el = borders.find(qn(f"w:{edge}"))
        if el is None:
            el = OxmlElement(f"w:{edge}")
            borders.append(el)
        el.set(qn("w:val"), "nil")
    if header:
        for cell in table.rows[0].cells:
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_borders = tc_pr.first_child_found_in("w:tcBorders")
            if tc_borders is None:
                tc_borders = OxmlElement("w:tcBorders")
                tc_pr.append(tc_borders)
            bottom = OxmlElement("w:bottom")
            bottom.set(qn("w:val"), "single")
            bottom.set(qn("w:sz"), "8")
            bottom.set(qn("w:color"), "000000")
            tc_borders.append(bottom)


def set_repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    repeat = OxmlElement("w:tblHeader")
    repeat.set(qn("w:val"), "true")
    tr_pr.append(repeat)


def set_font(run, east_asia="宋体", latin="Times New Roman", size=10.5, bold=None, italic=None):
    run.font.name = latin
    run._element.rPr.rFonts.set(qn("w:eastAsia"), east_asia)
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def add_text(p, text, bold=False, italic=False, size=None, east_asia="宋体"):
    r = p.add_run(text)
    set_font(r, east_asia=east_asia, size=size or 10.5, bold=bold, italic=italic)
    return r


def add_body(doc, text, first_line=True, keep=False):
    p = doc.add_paragraph(style="正文")
    p.paragraph_format.keep_together = keep
    if not first_line:
        p.paragraph_format.first_line_indent = Cm(0)
    add_text(p, text)
    return p


def add_list_item(doc, label, text):
    p = doc.add_paragraph(style="正文")
    p.paragraph_format.left_indent = Cm(0.74)
    p.paragraph_format.first_line_indent = Cm(-0.74)
    add_text(p, label, bold=True)
    add_text(p, text)
    return p


def add_equation(doc, expr, number):
    p = doc.add_paragraph(style="公式")
    p.paragraph_format.tab_stops.add_tab_stop(Cm(15.2), WD_TAB_ALIGNMENT.RIGHT)
    r = p.add_run(expr)
    set_font(r, east_asia="Cambria Math", latin="Cambria Math", size=11.5)
    p.add_run("\t")
    r = p.add_run(f"（{number}）")
    set_font(r, size=10.5)
    return p


def add_caption(doc, text):
    p = doc.add_paragraph(style="图题")
    add_text(p, text, size=9)
    return p


def add_picture(doc, path, width_cm, caption):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(3)
    shape = p.add_run().add_picture(str(path), width=Cm(width_cm))
    shape._inline.docPr.set("descr", caption)
    shape._inline.docPr.set("title", caption)
    add_caption(doc, caption)


def add_table(doc, headers, rows, widths=None, caption=None, note=None):
    if caption:
        p = doc.add_paragraph(style="表题")
        add_text(p, caption, size=9)
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    set_repeat_table_header(table.rows[0])
    for j, h in enumerate(headers):
        cell = table.rows[0].cells[j]
        set_cell_shading(cell, "E7E6E6")
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_text(p, str(h), bold=True, size=8.5)
        set_cell_margins(cell)
        if widths:
            cell.width = Cm(widths[j])
    for row in rows:
        cells = table.add_row().cells
        for j, value in enumerate(row):
            cells[j].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            p = cells[j].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
            add_text(p, str(value), size=8.5)
            set_cell_margins(cells[j])
            if widths:
                cells[j].width = Cm(widths[j])
    set_table_borders(table)
    if note:
        p = doc.add_paragraph(style="表注")
        add_text(p, note, size=8.5)
    return table


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    fld_char1 = OxmlElement("w:fldChar")
    fld_char1.set(qn("w:fldCharType"), "begin")
    instr_text = OxmlElement("w:instrText")
    instr_text.set(qn("xml:space"), "preserve")
    instr_text.text = "PAGE"
    fld_char2 = OxmlElement("w:fldChar")
    fld_char2.set(qn("w:fldCharType"), "end")
    run._r.extend((fld_char1, instr_text, fld_char2))
    set_font(run, size=9)


def build_document():
    doc = Document()
    sec = doc.sections[0]
    sec.page_height = Cm(29.7)
    sec.page_width = Cm(21.0)
    sec.top_margin = Cm(2.5)
    sec.bottom_margin = Cm(2.5)
    sec.left_margin = Cm(2.6)
    sec.right_margin = Cm(2.4)
    sec.header_distance = Cm(1.3)
    sec.footer_distance = Cm(1.3)
    add_page_number(sec.footer.paragraphs[0])

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    normal.font.size = Pt(10.5)

    body = styles.add_style("正文", WD_STYLE_TYPE.PARAGRAPH)
    body.base_style = normal
    body.font.name = "Times New Roman"
    body._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    body.font.size = Pt(10.5)
    body.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    body.paragraph_format.first_line_indent = Cm(0.74)
    body.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    body.paragraph_format.space_after = Pt(0)
    body.paragraph_format.widow_control = True

    eq = styles.add_style("公式", WD_STYLE_TYPE.PARAGRAPH)
    eq.base_style = normal
    eq.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    eq.paragraph_format.space_before = Pt(3)
    eq.paragraph_format.space_after = Pt(3)
    eq.paragraph_format.keep_together = True

    for name in ("图题", "表题"):
        st = styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
        st.base_style = normal
        st.font.name = "Times New Roman"
        st._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
        st.font.size = Pt(9)
        st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        st.paragraph_format.space_before = Pt(0)
        st.paragraph_format.space_after = Pt(6 if name == "图题" else 3)
        st.paragraph_format.keep_with_next = name == "表题"
    note = styles.add_style("表注", WD_STYLE_TYPE.PARAGRAPH)
    note.base_style = normal
    note.font.size = Pt(8.5)
    note._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), "宋体")
    note.paragraph_format.left_indent = Cm(0.4)
    note.paragraph_format.right_indent = Cm(0.4)
    note.paragraph_format.space_after = Pt(6)

    for idx, size in ((1, 16), (2, 14), (3, 12)):
        st = styles[f"Heading {idx}"]
        st.font.name = "Times New Roman"
        st._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.color.rgb = RGBColor(0, 0, 0)
        st.paragraph_format.space_before = Pt(14 if idx == 1 else 10)
        st.paragraph_format.space_after = Pt(6)
        st.paragraph_format.keep_with_next = True
        st.paragraph_format.keep_together = True

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_before = Pt(36)
    title.paragraph_format.space_after = Pt(12)
    r = title.add_run("论文去模糊网络部分（中文稿）")
    set_font(r, east_asia="黑体", size=20, bold=True)
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = subtitle.add_run("基于局部运动偏置的线迹图像去模糊网络 MBDNet-P+")
    set_font(r, east_asia="黑体", size=14, bold=True)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(18)
    add_text(p, "适配“方法—对比实验—存在必要性实验”的完整章节结构", size=10.5)

    doc.add_heading("内容提要", level=1)
    add_body(doc, "针对工业缝制过程中相机与工件相对运动引起的局部线性拖影，以及织物纹理、印花、亮片等复杂背景与细线迹结构相互耦合的问题，本文构建一种基于局部运动偏置的单幅图像去模糊网络 MBDNet-P+。该网络以紧凑的编码器—解码器为主体，在中、低分辨率特征域中引入运动偏置去模糊模块（Motion-Biased Deblurring，MBD），通过逐像素内容路由与短、长水平支撑尺度路由，自适应融合通用各向同性恢复特征和近水平方向运动恢复特征。网络推理时仅接收 RGB 图像，不依赖数据域标签、线迹掩膜或预存退化参数。")
    add_body(doc, "在统一的 MultiSource SynBlur-v2 六域测试协议下，MBDNet-P+ 的全图宏平均 PSNR 和 SSIM 分别达到 29.5201 dB 和 0.9226；相较 NAFNet 分别提高 1.1706 dB 和 0.0244，相较 MPRNet 分别提高 0.2981 dB 和 0.0025，同时单图推理时间由 MPRNet 的 67.19 ms 降至 32.98 ms。进一步以 48 张一一对应样本构造“图像恢复—线迹分割—缺陷检测”三层证据链：直接模糊使缺陷宏平均 F1 由 82.89% 降至 52.66%，经 MBDNet-P+ 恢复后回升至 71.01%，从下游任务角度证明去模糊环节的必要性。")
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    add_text(p, "关键词：", bold=True)
    add_text(p, "工业线迹；运动模糊；图像恢复；局部路由；深度可分离卷积；缺陷检测")
    doc.add_page_break()

    doc.add_heading("2.3 基于局部运动偏置的线迹图像去模糊网络", level=1)
    doc.add_heading("2.3.1 问题分析与设计动机", level=2)
    add_body(doc, "工业缝制图像中的模糊通常并非均匀失焦，而是由送料机构、缝纫机头、工件和相机之间的相对运动造成。对于本研究数据，退化方向总体以近水平方向为主，但模糊强度、核长度和局部显著性会随位置变化；与此同时，织物经纬纹理、金粉、刺绣和印花边缘又具有任意方向。若在整个特征图上施加固定水平卷积或全局方向注意力，虽然能够增强部分线迹，却容易把背景纹理误判为运动轨迹并产生过锐化。反之，仅依赖各向同性 3×3 卷积，则难以在较低计算量下覆盖较长拖影。")
    add_body(doc, "MPRNet 通过多阶段渐进恢复和跨阶段监督取得良好性能[22]，但多阶段结构带来较高推理开销；NAFNet 以简洁的非线性门控块建立了较强恢复基线[23]，其通用性较好，但没有显式编码本数据中“局部近水平运动拖影与任意方向材质纹理并存”的先验。频域方法能够扩大感受野并增强全局建模[24]，模糊感知注意力也能按退化程度调节特征[25]，但对本项目而言，直接堆叠频率分支或全局模糊权重并不能保证各数据域同步获益。基于上述观察，本文不采用材料分类或硬方向掩膜，而将方向先验限制为一种可被局部内容否决的软偏置。")
    add_body(doc, "围绕上述矛盾，MBDNet-P+ 的核心设计可概括为以下三点：")
    add_list_item(doc, "（1）", "构建局部运动偏置去模糊模块 MBD，以通用分支保护任意方向纹理，以短、长水平深度卷积分支覆盖不同长度拖影，并用两级逐像素路由完成柔性选择；")
    add_list_item(doc, "（2）", "采用尺度受限的模块布局，仅在 1/2 和 1/4 分辨率部署 MBD，在全分辨率层保留通用 DWPW 恢复单元，使方向先验服务于模糊建模而不过度干预细粒度背景；")
    add_list_item(doc, "（3）", "通过全局图像残差、跨尺度跳跃连接和零初始化残差尺度建立稳定优化路径，并在基础重建目标上加入水平梯度与 SSIM 约束，使恢复结果同时兼顾像素保真、线迹连续性和结构相似性。")

    doc.add_heading("2.3.2 网络总体结构", level=2)
    add_body(doc, "MBDNet-P+ 采用两级下采样的对称编码器—解码器，如图 1 所示。给定模糊图像 I_b∈R^(3×H×W)，网络首先使用 3×3 卷积生成 64 通道浅层特征；随后通过全分辨率、半分辨率和四分之一分辨率三个尺度进行编码。各尺度通道数依次为 64、128 和 256，DWPW 单元数量依次为 3、4 和 12。瓶颈在 12 个通用单元之后加入局部—空洞上下文单元、第一 MBD、2 个 DWPW 精炼单元和第二 MBD，从而在较低空间开销下获得较大的有效感受野。")
    add_picture(doc, FIG_DIR / "Fig1_MBDNetPPlus_overall.png", 16.0, "图 1  MBDNet-P+ 总体网络结构")
    add_body(doc, "解码端采用 1×1 通道扩展与 PixelShuffle 完成上采样，并与相同尺度的编码特征逐元素相加。相比特征拼接，相加式跳跃连接不增加后续通道宽度，可直接复用编码端的高频定位信息。最终 3×3 卷积预测残差图 R，清晰图像通过 I_hat=I_b+R 获得。全局残差使网络重点学习模糊分量与清晰图像之间的差值，能够保留低频颜色和大面积背景亮度，并缩短梯度传播路径。输入尺寸不能被 4 整除时，仅在右侧和下侧进行复制填充，推理结束后裁回原尺寸，因此不改变有效视场。")
    add_table(doc,
              ["阶段", "空间尺度", "通道数", "主要单元", "数量/连接"],
              [
                  ["浅层映射", "H×W", "64", "3×3 Conv", "1"],
                  ["全分辨率编码", "H×W", "64", "DWPW", "3"],
                  ["半分辨率编码", "H/2×W/2", "128", "DWPW + MBD", "4 + 1"],
                  ["低分辨率瓶颈", "H/4×W/4", "256", "DWPW + LRC + MBD", "12 + 1 + 1"],
                  ["瓶颈精炼", "H/4×W/4", "256", "DWPW + MBD", "2 + 1"],
                  ["半分辨率解码", "H/2×W/2", "128", "DWPW + MBD", "4 + 1"],
                  ["全分辨率解码", "H×W", "64", "DWPW", "3"],
                  ["图像重建", "H×W", "3", "3×3 Conv + 全局残差", "1"],
              ],
              widths=[2.7, 2.8, 1.8, 4.2, 2.4],
              caption="表 1  MBDNet-P+ 的分层配置",
              note="注：全网络包含 28 个 DWPW 单元和 4 个 MBD 模块；默认参数量为 13.70 M。")

    doc.add_heading("2.3.3 DWPW 通用恢复单元", level=2)
    add_body(doc, "DWPW 单元承担网络中的通用图像恢复任务，其结构如图 2(a) 所示。该单元沿用归一化—空间混合—通道混合的两级残差形式，但将普通卷积拆分为逐点卷积和 3×3 深度卷积，以减少空间混合的参数量。设输入特征为 X，LN(·) 表示二维层归一化，P_i(·) 表示 1×1 逐点卷积，DW_3(·) 表示 3×3 深度卷积，SG(·) 表示 SimpleGate。SimpleGate 将通道均分为 X_1 与 X_2，并通过逐元素乘积 X_1⊙X_2 形成数据依赖的非线性交互。DWPW 的两级运算为：")
    add_equation(doc, "Y = X + β ⊙ P₁[SG(DW₃(P₀(LN(X))))]", 1)
    add_equation(doc, "Z = Y + γ ⊙ P₃[SG(P₂(LN(Y)))]", 2)
    add_body(doc, "其中 β 和 γ 为可学习的逐通道缩放参数，并初始化为 0。训练开始时模块等价于恒等映射，随着优化进行再逐步引入空间残差和通道残差，可降低深层恢复网络在随机初始化阶段出现梯度放大或色彩漂移的风险。第一子层先扩展通道，再以深度卷积完成局部空间混合；第二子层相当于轻量前馈网络，负责跨通道重组。该设计避免复杂激活函数和自注意力的二次空间复杂度，适合 320×320 图像块训练与单幅工业图像快速推理。")

    doc.add_heading("2.3.4 局部运动偏置去模糊模块", level=2)
    add_body(doc, "MBD 是本文面向线迹运动退化设计的核心结构，如图 2(b) 所示。与采用单一方向卷积不同，MBD 同时保留通用分支、短程水平分支和长程水平分支，并用内容路由与尺度路由逐像素控制其贡献。设输入为 F，归一化特征为 F_n，通用分支 G、短程分支 H_s 与长程分支 H_l 定义为：")
    add_equation(doc, "F_n = LN(F),   G = DW₃×₃(F_n)", 3)
    add_equation(doc, "H_s = DW₁×₅(F_n),   H_l = DW₁×₅^(d=3)(F_n)", 4)
    add_body(doc, "短程分支的有效水平支撑为 5，适用于轻度拖影；长程分支使用膨胀率 3 的 1×5 深度卷积，其有效水平支撑扩大至 13，可覆盖更长模糊轨迹。两者均为深度卷积，不在空间混合阶段引入额外的跨通道冗余。路由上下文由 3×3 深度卷积获得，并产生两类权重：")
    add_equation(doc, "C = DW₃×₃(F_n),   [ω_s,ω_l] = Softmax(P_s(C)),   α = Sigmoid(P_c(C))", 5)
    add_body(doc, "其中 ω_s 和 ω_l 在每个空间位置上和为 1，用于选择短程或长程水平支撑；α∈[0,1] 是逐像素内容门控，用于判断当前位置是否应采用运动偏置。当背景包含斜向印花、织物经纬或高亮装饰而缺少水平拖影证据时，α 可趋近 0，使模块主要依赖通用分支；在线迹及其水平拖尾区域，α 可增大并由尺度路由进一步选择合适的模糊长度。最终输出为：")
    add_equation(doc, "H = ω_s ⊙ H_s + ω_l ⊙ H_l", 6)
    add_equation(doc, "M = (1−α) ⊙ G + α ⊙ H,   F_out = F + ρ ⊙ P(M)", 7)
    add_body(doc, "ρ 为零初始化的逐通道残差尺度，P(·) 为 1×1 投影。与硬方向掩膜相比，式（7）的软选择在空间上连续、可微，并允许模型完全否决不合适的方向先验；与单一全局模糊权重相比，它能够处理同一图像中线迹、背景与装饰区域退化程度不同的情形。MBD 被限制在 1/2 和 1/4 尺度，使长程分支以较低成本覆盖更大原图范围，同时避免在全分辨率层直接放大细密织纹。")
    add_picture(doc, FIG_DIR / "Fig2_DWPW_MBD_modules.png", 16.0, "图 2  DWPW 通用恢复单元与 MBD 运动偏置模块")

    doc.add_heading("2.3.5 低分辨率上下文与多尺度重建", level=2)
    add_body(doc, "低分辨率上下文单元（Low-Resolution Context，LRC）仅布置在四分之一尺度。其并行使用局部 3×3 深度卷积和膨胀率为 2 的 3×3 深度卷积，再通过通道拼接与 1×1 卷积融合。局部分支用于维持线迹端点和织纹细节，空洞分支用于聚合更大范围内的模糊连续性；该单元同样采用零初始化残差尺度，以避免训练初期扩大伪影。LRC 不承担方向判别，而是向后续 MBD 路由提供更稳定的上下文。")
    add_body(doc, "下采样由步长为 2 的 3×3 深度卷积和 1×1 通道投影组成，先进行空间抗混叠式聚合再增加通道；上采样先以 1×1 卷积扩展通道，再通过 PixelShuffle 恢复空间分辨率。编码器与解码器在 H/2、H 两个尺度上采用逐元素相加连接。该设计使浅层边缘定位与深层模糊上下文协同工作，同时避免转置卷积常见的棋盘伪影。")

    doc.add_heading("2.3.6 损失函数与训练策略", level=2)
    add_body(doc, "基础训练使用对数均方误差形式的 PSNR 损失。设批量大小为 B，预测与真值之间的均方误差为 MSE_i，则重建项为：")
    add_equation(doc, "L_rec = (10/ln10) · (1/B) Σ_i ln(MSE_i + ε)", 8)
    add_body(doc, "最小化式（8）等价于最大化批内样本的 PSNR，并比直接优化 L1 更强调高质量区间的细微误差。为强化线迹的水平连续性，在最终微调阶段引入水平方向梯度 Charbonnier 损失：")
    add_equation(doc, "L_grad = mean{[(∇_x I_hat − ∇_x I_gt)² + ε_g²]^(1/2)}", 9)
    add_body(doc, "其中 ∇_x 表示相邻列差分。该约束并非强制图像生成水平边缘，而是约束恢复图与真值在水平方向的局部变化一致，从而抑制沿线迹方向的断裂和振铃。结构相似性项定义为 L_ssim=1−SSIM(I_hat,I_gt)，总损失为：")
    add_equation(doc, "L_total = L_rec + 0.02L_grad + 0.03L_ssim", 10)
    add_body(doc, "训练分为三个连续阶段：首先随机初始化训练 30 轮，初始学习率为 1×10^−3；随后从最佳权重以 2×10^−5 的学习率继续进行 10 轮 PSNR 微调；最后以 5×10^−6 的学习率进行 5 轮方向梯度与 SSIM 联合微调。训练裁剪尺寸为 320×320，物理批量为 2，梯度累积 2 次，等效批量为 4，采用自动混合精度和余弦学习率衰减，随机种子固定为 42。三个阶段均只使用训练/验证清单，不读取测试图像的退化参数。最终联合微调相较纯 PSNR 微调的宏平均 PSNR 仅增加 0.0016 dB，因此本文将其解释为结构一致性的收尾优化，而不将该微小差异夸大为独立结构贡献。")

    doc.add_heading("5.3 去模糊实验与结果分析", level=1)
    doc.add_heading("5.3.1 数据协议与评价指标", level=2)
    add_body(doc, "为验证跨材质与跨场景恢复能力，本文采用 MultiSource SynBlur-v2 配对协议。DB、XJ、SD 和 RX 四类数据参与训练与验证；RXHF 与 TJML 不参与参数更新，仅作为外部分布测试域。每个训练样本由清晰图像及其受控线性运动退化图像构成。训练集共 4320 对，验证集 160 对；测试集包含 DB、XJ、SD、RX 各 100 对，以及 RXHF、TJML 各 50 对。所有方法使用相同测试清单，逐图计算指标后再聚合。")
    add_table(doc,
              ["数据域", "训练对", "验证对", "测试对", "角色"],
              [
                  ["DB", "1080", "40", "100", "训练/验证/测试"],
                  ["XJ", "1080", "40", "100", "训练/验证/测试"],
                  ["SD", "1080", "40", "100", "训练/验证/测试"],
                  ["RX", "1080", "40", "100", "训练/验证/测试"],
                  ["RXHF", "0", "0", "50", "纯外部测试"],
                  ["TJML", "0", "0", "50", "纯外部测试"],
                  ["合计", "4320", "160", "500", "—"],
              ],
              widths=[2.4, 2.2, 2.2, 2.2, 4.2],
              caption="表 2  MultiSource SynBlur-v2 数据分布",
              note="注：RX 已纳入训练与验证，不属于纯外部测试；纯外部测试域为 RXHF 与 TJML。")
    add_body(doc, "峰值信噪比 PSNR 根据预测图像与真值图像的均方误差计算；结构相似性 SSIM 采用 11×11 高斯窗口、标准差 1.5，并对 RGB 三通道求均值。除全图指标外，本文同时报告图像中间水平三分之一区域的 PSNR/SSIM，用于观察主要线迹区域的恢复情况。宏平均对六个数据域赋予相同权重，能够避免样本数量较多的数据域主导结论；样本加权平均作为补充，用于描述全部 500 张测试图像的总体表现。")
    add_equation(doc, "PSNR = 10log₁₀(MAX²/MSE)", 11)
    add_equation(doc, "Macro(M) = (1/6) Σ_(d=1)^6 M_d", 12)

    doc.add_heading("5.3.2 与典型去模糊网络的比较", level=2)
    add_body(doc, "当前对比包含多阶段渐进恢复网络 MPRNet[22] 和通用图像恢复基线 NAFNet[23]。三种模型均使用同一 MultiSource SynBlur-v2 训练/验证划分、同一六域测试清单、RGB [0,1] 范围、FP32 推理和逐图宏平均方式评估。为区分严格训练预算与最终部署模型，表 3 同时给出三种模型统一训练 30 轮的结果，以及 MBDNet-P+ 在此基础上继续低学习率微调 15 轮的最终结果。严格结构比较应采用 S30 行；最终行用于说明当前交付权重的性能。后续新增模型时应保持清单、输入范围、裁剪方式、总训练轮次、推理精度和计时预热策略一致，而不应直接引用其他论文在 GoPro 等数据集上的数值。")
    add_table(doc,
              ["模型", "训练阶段", "参数量/M", "全图 PSNR/dB", "全图 SSIM", "中带 PSNR/dB", "时间/ms"],
              [
                  ["NAFNet[23]", "S30", "17.11", "28.3495", "0.8982", "27.7612", "24.70"],
                  ["MPRNet[22]", "S30", "20.13", "29.2220", "0.9201", "28.4952", "67.19"],
                  ["MBDNet-P+（本文）", "S30", "13.70", "29.4954", "0.9222", "28.8346", "31.82"],
                  ["MBDNet-P+（最终）", "S30+FT15", "13.70", "29.5201", "0.9226", "28.8561", "32.98"],
              ],
              widths=[3.2, 1.9, 1.7, 2.4, 1.9, 2.4, 1.8],
              caption="表 3  同一协议下不同去模糊模型的六域宏平均结果",
              note="注：时间为单张图像纯网络推理平均耗时；不同于第 5.4 节 48 张端到端验证中的完整处理时间。加粗可在论文定稿时按列标注最优值。")
    add_picture(doc, FIG_DIR / "Fig3_model_comparison.png", 15.8, "图 3  统一 30 轮训练预算下三种模型的宏平均 PSNR 与 SSIM")
    add_body(doc, "由表 3 可见，在统一 30 轮训练预算下，MBDNet-P+ 的宏平均 PSNR 相较 NAFNet 提高 1.1459 dB，SSIM 提高 0.0241；参数量减少约 19.94%，代价是平均推理时间增加 7.12 ms。相较 MPRNet，MBDNet-P+ 的 PSNR 和 SSIM 分别提高 0.2734 dB 和 0.0021，参数量减少约 31.93%，推理时间缩短 35.37 ms（约 52.6%）。最终微调模型在此基础上进一步达到 29.5201 dB 和 0.9226。结果说明在本数据协议下，局部运动偏置与尺度受限布局能够在精度和速度之间取得较优平衡。需要强调的是，当前只完成两个公开基线，结论应限定为“优于已完成对比模型”；在论文终稿中仍需补充近年开源网络的同协议复现实验。")
    add_table(doc,
              ["数据域", "图像数", "全图 PSNR/dB", "全图 SSIM", "中带 PSNR/dB", "中带 SSIM", "时间/ms"],
              [
                  ["DB", "100", "30.7782", "0.9302", "29.8204", "0.9224", "37.04"],
                  ["XJ", "100", "34.9884", "0.9535", "33.6976", "0.9476", "31.45"],
                  ["SD", "100", "32.0121", "0.9514", "31.2381", "0.9399", "31.53"],
                  ["RX", "100", "26.8644", "0.9219", "26.4578", "0.9147", "31.83"],
                  ["RXHF（外部）", "50", "28.3843", "0.9170", "28.1824", "0.9090", "33.29"],
                  ["TJML（外部）", "50", "24.0933", "0.8615", "23.7403", "0.8452", "32.75"],
                  ["六域宏平均", "—", "29.5201", "0.9226", "28.8561", "0.9131", "32.98"],
              ],
              widths=[2.6, 1.4, 2.2, 1.8, 2.2, 1.8, 1.7],
              caption="表 4  最终 MBDNet-P+ 在各数据域上的结果",
              note="注：RXHF 与 TJML 为纯外部测试域；六域宏平均对每个数据域赋予相同权重。")
    add_body(doc, "从分域结果看，最终模型在 RXHF 与 TJML 两个纯外部测试域分别取得 28.3843 dB 和 24.0933 dB，说明局部路由并未依赖显式域标签。对照原始测试记录可见，最终模型在 DB、XJ、SD、RXHF 和 TJML 上的 PSNR 高于 MPRNet，而 RX 域比 MPRNet 低 0.0924 dB，提示近水平方向先验并非对所有材质均为最优。保留通用分支与逐像素否决机制能够缓解方向偏置的不利影响，但后续仍可在不接触测试标签的前提下，通过更丰富的训练退化核分布改善该域。")

    doc.add_heading("5.3.3 训练阶段的递进结果与消融边界", level=2)
    add_table(doc,
              ["训练阶段", "训练设置", "宏 PSNR/dB", "宏 SSIM", "中带 PSNR/dB", "中带 SSIM"],
              [
                  ["基础训练", "随机初始化，30 轮，PSNR 损失", "29.4954", "0.9222", "28.8346", "0.9128"],
                  ["FT1", "+10 轮低学习率 PSNR 微调", "29.5185", "0.9226", "28.8554", "0.9131"],
                  ["FT2（最终）", "+5 轮梯度与 SSIM 联合微调", "29.5201", "0.9226", "28.8561", "0.9131"],
              ],
              widths=[2.8, 5.1, 2.2, 2.0, 2.3, 2.0],
              caption="表 5  MBDNet-P+ 的训练阶段递进结果",
              note="注：该表反映同一结构的优化过程，不等同于模块消融。严格模块消融需在相同训练轮次与初始化条件下分别移除 MBD、LRC 或路由分支。")
    add_body(doc, "表 5 表明基础模型已贡献主要性能，后续 15 轮微调的累计增益约为 0.0247 dB。为避免不公平归因，论文不应把 FT1 和 FT2 写成两个结构创新。建议正式消融至少设置四组：通用 DWPW 编码器—解码器；加入 MBD 但去除尺度路由（固定均匀融合短、长支撑）；完整 MBD；完整 MBD 加 LRC。四组应使用相同数据、随机种子、总轮次和学习率计划，并同时报告六域宏平均与两个外部域结果。待这些实验完成后，可将表 5 保留为训练策略分析，并另增严格结构消融表。")

    doc.add_heading("5.4 去模糊模块存在必要性实验", level=1)
    doc.add_heading("5.4.1 实验设计与因果控制", level=2)
    add_body(doc, "仅报告 PSNR/SSIM 只能说明恢复图像接近清晰参考，不能直接证明去模糊对最终线迹缺陷识别有用。为此，本文建立图像恢复、线迹分割和缺陷检测三层证据链。实验从同一批 48 张清晰源图出发，为每张图像生成一个一一对应的运动模糊样本，并设置三条输入路径：清晰图像直接进入下游网络；模糊图像不经恢复直接进入下游网络；模糊图像先经冻结的 MBDNet-P+ 恢复，再进入同一冻结下游网络。所有分割器、缺陷检测器、决策阈值和标注均保持不变，MBDNet-P+ 不在这 48 张样本上微调或选参，因此“直接模糊—去模糊后”的差异可以归因于恢复环节。")
    add_picture(doc, FIG_DIR / "Fig4_necessity_causal_chain.png", 16.0, "图 4  去模糊模块存在必要性的三层因果证据")

    doc.add_heading("5.4.2 图像恢复层结果", level=2)
    add_body(doc, "在 48 张样本上，冻结的 MBDNet-P+ 相对清晰参考图取得 28.6485 dB PSNR、0.8856 SSIM 和 0.0262 MAE，平均完整推理时间为 55.22 ms/图。该时间统计来自必要性实验的独立端到端流程，包含其自身的数据读入与前后处理条件，不应与表 3 中统一测试器测得的 32.98 ms 直接横向比较。恢复层结果证明模型能够将受控运动模糊映射回接近参考图的图像空间，但是否真正有利于任务仍需后两层验证。")
    add_table(doc,
              ["样本数", "PSNR/dB", "SSIM", "MAE", "完整推理时间/ms"],
              [["48", "28.6485", "0.8856", "0.0262", "55.22"]],
              widths=[2.4, 2.6, 2.4, 2.4, 3.4],
              caption="表 6  必要性实验的图像恢复层结果")

    doc.add_heading("5.4.3 线迹分割层结果", level=2)
    add_table(doc,
              ["指标", "清晰参考/%", "直接模糊/%", "去模糊后/%", "较模糊提升/百分点", "损失恢复率/%"],
              [
                  ["二值 F1", "89.12", "82.54", "88.12", "+5.58", "84.7"],
                  ["二值 IoU", "80.45", "70.76", "78.89", "+8.13", "83.9"],
                  ["轨迹 F1（1 px）", "98.04", "91.51", "97.34", "+5.83", "89.2"],
                  ["边界 F1（1 px）", "95.12", "86.04", "93.95", "+7.91", "87.1"],
              ],
              widths=[3.0, 2.2, 2.2, 2.2, 2.8, 2.2],
              caption="表 7  清晰、直接模糊与去模糊后三种路径的分割结果",
              note="注：损失恢复率=(去模糊后−直接模糊)/(清晰参考−直接模糊)×100%。")
    add_body(doc, "运动模糊使二值 IoU 下降 9.69 个百分点、边界 F1 下降 9.08 个百分点，说明拖影主要破坏线迹与背景之间的边界定位。加入去模糊后，二值 IoU 和边界 F1 分别回升 8.13 与 7.91 个百分点，对应恢复了 83.9% 和 87.1% 的清晰—模糊性能损失。轨迹 F1 恢复率达到 89.2%，表明 MBDNet-P+ 对线迹中心轨迹的连续性恢复尤为有效。去模糊结果尚未完全达到清晰上限，说明恢复网络仍可能产生少量边界偏移或弱纹理损失，但其净收益稳定地体现在所有四项分割指标上。")

    doc.add_heading("5.4.4 缺陷检测层结果", level=2)
    add_table(doc,
              ["指标", "清晰参考/%", "直接模糊/%", "去模糊后/%", "较模糊提升/百分点", "损失恢复率/%"],
              [
                  ["四类缺陷宏 F1", "82.89", "52.66", "71.01", "+18.35", "60.7"],
                  ["类别平均准确率", "97.40", "91.67", "96.35", "+4.69", "81.8"],
                  ["局部单元完全正确率", "96.53", "77.43", "93.06", "+15.63", "81.8"],
                  ["图像级完全正确率", "75.00", "35.42", "58.33", "+22.92", "57.9"],
              ],
              widths=[3.3, 2.2, 2.2, 2.2, 2.8, 2.2],
              caption="表 8  清晰、直接模糊与去模糊后三种路径的缺陷检测结果",
              note="注：宏 F1 对跳线、断线、歪斜和松线四类缺陷等权平均；局部单元与图像级完全正确率使用固定的清晰标定阈值。")
    add_body(doc, "缺陷检测对上游边界误差更敏感。直接模糊使四类缺陷宏 F1 由 82.89% 降至 52.66%，降幅达 30.23 个百分点；经去模糊后回升至 71.01%，相对直接模糊提高 18.35 个百分点，恢复了 60.7% 的损失。局部单元完全正确率由 77.43% 提升至 93.06%，接近清晰参考的 96.53%；图像级完全正确率由 35.42% 提升至 58.33%。由于图像级完全正确要求一张图像内所有局部判断同时正确，其指标更严格，因此恢复比例低于局部指标是合理的。上述结果构成“模糊导致任务退化—恢复使任务回升”的直接证据，而非仅依赖视觉主观判断。")
    add_picture(doc, ROOT / "results" / "deblurring_ablation" / "quexian_synblur48_end_to_end_tsr_v5" / "qualitative_end_to_end_defect_examples.png", 15.4, "图 5  清晰、直接模糊与去模糊后三种输入下的端到端缺陷检测示例")
    add_body(doc, "图 5 从上至下分别给出跳线、断线、歪斜和松线示例。直接模糊会使黄色线迹轨迹发生扩散或局部断裂，导致固定阈值下出现漏检和类别混淆；去模糊后，线迹边缘和局部间隔得到恢复，跳线、断线、歪斜与松线的判别趋近清晰输入。需要注意，去模糊结果不必在像素层面完全复制清晰图像，只要恢复与缺陷判别相关的线迹边界、连通关系和间距信息，即可对下游任务产生实际价值。")

    doc.add_heading("5.4.5 结果讨论与有效性边界", level=2)
    add_body(doc, "三层实验共同说明，PSNR/SSIM 的提升并非孤立的图像质量变化，而能沿“恢复—分割—缺陷判断”链条向后传递。尤其是缺陷宏 F1 的 18.35 个百分点净回升，比主对比中 0.1～1 dB 量级的 PSNR 差异更能体现工业任务价值。因此，论文应将图像质量指标作为恢复保真度证据，将分割与缺陷指标作为存在必要性的核心因果证据。")
    add_body(doc, "同时，本实验仍有三个边界条件。第一，48 张样本使用一一对应的受控合成运动模糊，后续应增加真实高速采集模糊样本作为外部验证；第二，目前只报告单次固定划分结果，若论文需要显著性检验，应对关键模型使用多个随机种子并报告均值与标准差；第三，现有对比模型仅为 MPRNet 和 NAFNet，应继续加入近年开源去模糊方法，并按同一清单复现，而不能混用其他数据集上公布的指标。上述限制不否定当前因果链，但界定了结论的适用范围。")

    doc.add_heading("本节小结", level=1)
    add_body(doc, "本文针对线迹图像中近水平运动拖影与复杂任意方向背景共存的问题，提出 MBDNet-P+。网络通过 DWPW 通用恢复单元建立高效主干，通过 MBD 的内容路由和尺度路由在局部区域选择通用、短程或长程运动恢复特征，并将方向先验限制在中低分辨率层。统一六域协议表明，MBDNet-P+ 在 13.70 M 参数量下取得 29.5201 dB 宏平均 PSNR 和 0.9226 SSIM，优于当前已完成的 MPRNet 与 NAFNet。进一步的三层必要性实验显示，去模糊后分割 IoU 提高 8.13 个百分点，缺陷宏 F1 提高 18.35 个百分点，从端到端任务角度验证了恢复模块的实际作用。")

    doc.add_heading("参考文献（与原稿编号衔接）", level=1)
    refs = [
        "[22] Zamir S W, Arora A, Khan S, et al. Multi-stage progressive image restoration[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021: 14821–14831.",
        "[23] Chen L, Chu X, Zhang X, et al. Simple baselines for image restoration[C]//European Conference on Computer Vision. Cham: Springer, 2022: 17–33.",
        "[24] Mao X, Liu Y, Shen W, et al. Intriguing findings of frequency selection for image deblurring[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2023, 37(2): 1905–1913.",
        "[25] Tsai F J, Peng Y T, Tsai C C, et al. BANet: A blur-aware attention network for dynamic scene deblurring[J]. IEEE Transactions on Image Processing, 2022, 31: 6789–6799.",
    ]
    for ref in refs:
        p = doc.add_paragraph(style="正文")
        p.paragraph_format.first_line_indent = Cm(-0.74)
        p.paragraph_format.left_indent = Cm(0.74)
        p.paragraph_format.line_spacing = 1.2
        add_text(p, ref, size=9.5)

    props = doc.core_properties
    props.title = "论文去模糊网络部分：MBDNet-P+"
    props.subject = "工业线迹图像去模糊网络、对比实验与存在必要性实验"
    props.author = "C3Netqing 项目"
    props.keywords = "MBDNet-P+, 去模糊, 线迹, MPRNet, NAFNet, 必要性实验"
    props.comments = "根据项目代码、同协议测试结果与三层端到端证据生成。"

    DOC_DIR.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    build_document()

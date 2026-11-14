"""Create the AI-facing technical brief for the three-stage thesis pipeline."""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "论文三阶段整体技术说明_供AI写作.docx"
FIGURE = ROOT / "results" / "deblurring_ablation" / "quexian_synblur48_end_to_end_tsr_v5" / "qualitative_end_to_end_defect_examples.png"

NAVY = "1F4D78"
BLUE = "2E74B5"
LIGHT_BLUE = "E8EEF5"
LIGHT_GREY = "F4F6F9"
BLACK = "202020"
MUTED = "666666"


def set_font(run, name="Microsoft YaHei", size=None, bold=None, color=None, italic=None):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:ascii"), name)
    run._element.rPr.rFonts.set(qn("w:hAnsi"), name)
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color is not None:
        run.font.color.rgb = RGBColor.from_string(color)


def shade(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_cell_width(cell, width_dxa):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_w = tc_pr.find(qn("w:tcW"))
    if tc_w is None:
        tc_w = OxmlElement("w:tcW")
        tc_pr.append(tc_w)
    tc_w.set(qn("w:w"), str(width_dxa))
    tc_w.set(qn("w:type"), "dxa")


def set_table_geometry(table, widths):
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
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
    tbl_ind.set(qn("w:w"), "120")
    tbl_ind.set(qn("w:type"), "dxa")
    grid = table._tbl.tblGrid
    for col, width in zip(grid.gridCol_lst, widths):
        col.set(qn("w:w"), str(width))
    for row in table.rows:
        for cell, width in zip(row.cells, widths):
            set_cell_width(cell, width)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            tc_pr = cell._tc.get_or_add_tcPr()
            margins = tc_pr.first_child_found_in("w:tcMar")
            if margins is None:
                margins = OxmlElement("w:tcMar")
                tc_pr.append(margins)
            for side in ("top", "bottom", "start", "end"):
                tag = qn(f"w:{side}")
                element = margins.find(tag)
                if element is None:
                    element = OxmlElement(f"w:{side}")
                    margins.append(element)
                element.set(qn("w:w"), "80" if side in ("top", "bottom") else "120")
                element.set(qn("w:type"), "dxa")


def style_table(table, header=True):
    for r, row in enumerate(table.rows):
        for cell in row.cells:
            if header and r == 0:
                shade(cell, LIGHT_BLUE)
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_before = Pt(0)
                paragraph.paragraph_format.space_after = Pt(0)
                paragraph.paragraph_format.line_spacing = 1.1
                for run in paragraph.runs:
                    set_font(run, size=9.2, bold=(header and r == 0), color=BLACK)


def add_table(doc, headers, rows, widths):
    table = doc.add_table(rows=1, cols=len(headers))
    for cell, text in zip(table.rows[0].cells, headers):
        cell.text = str(text)
    for values in rows:
        cells = table.add_row().cells
        for cell, value in zip(cells, values):
            cell.text = str(value)
    set_table_geometry(table, widths)
    style_table(table)
    return table


def add_heading(doc, text, level=1):
    paragraph = doc.add_paragraph(style=f"Heading {level}")
    paragraph.paragraph_format.keep_with_next = True
    paragraph.add_run(text)
    return paragraph


def add_body(doc, text, lead=None):
    paragraph = doc.add_paragraph()
    if lead:
        run = paragraph.add_run(lead)
        set_font(run, size=10.5, bold=True, color=BLACK)
    run = paragraph.add_run(text)
    set_font(run, size=10.5, color=BLACK)
    return paragraph


def add_note(doc, title, text):
    table = doc.add_table(rows=1, cols=1)
    set_table_geometry(table, [9360])
    cell = table.cell(0, 0)
    shade(cell, LIGHT_GREY)
    paragraph = cell.paragraphs[0]
    paragraph.paragraph_format.space_before = Pt(3)
    paragraph.paragraph_format.space_after = Pt(3)
    run = paragraph.add_run(title + "：")
    set_font(run, size=10, bold=True, color=NAVY)
    run = paragraph.add_run(text)
    set_font(run, size=10, color=BLACK)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def add_path(doc, label, path):
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.left_indent = Inches(0.18)
    run = paragraph.add_run(label + "：")
    set_font(run, size=9.3, bold=True, color=BLACK)
    run = paragraph.add_run(path)
    set_font(run, name="Consolas", size=8.5, color=MUTED)


def configure(doc):
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(0.82)
    section.bottom_margin = Inches(0.75)
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(1.0)
    section.header_distance = Inches(0.35)
    section.footer_distance = Inches(0.35)
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Microsoft YaHei"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    normal.font.size = Pt(10.5)
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.25
    for name, size, color, before, after in (("Heading 1", 16, BLUE, 18, 10), ("Heading 2", 13, BLUE, 12, 6), ("Heading 3", 11.5, NAVY, 8, 4)):
        style = styles[name]
        style.font.name = "Microsoft YaHei"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor.from_string(color)
        style.font.bold = True
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True
    header = section.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    header_run = header.add_run("三阶段线迹缺陷检测系统技术说明")
    set_font(header_run, size=8.5, color=MUTED)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer_run = footer.add_run("供论文写作 AI 使用 - 请以实验协议与限制条件为准")
    set_font(footer_run, size=8.2, color=MUTED)


def build():
    doc = Document()
    configure(doc)
    # Opening / intended use
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(50)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("基于去模糊、分割与轨迹自参照的\n线迹缺陷检测系统")
    set_font(run, size=23, bold=True, color=NAVY)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("论文整体技术说明（供 AI 辅助论文写作）")
    set_font(run, size=13.5, color=MUTED)
    doc.add_paragraph().paragraph_format.space_after = Pt(8)
    add_table(doc, ["项目", "已确认内容"], [
        ("整体系统", "MBDNet-P+ FT2 去模糊网络 → M6 线迹分割网络 → TSR-Defect 缺陷检测单元"),
        ("任务", "针对含六段线迹的缝纫图像，识别跳线、断线、整体歪斜与松线/浮线。"),
        ("图像尺寸", "所有实验输入统一为 320 × 320 px。"),
        ("数据规模", "48 个原始图像来源；另有其水平翻转，共 96 张。翻转图仅用于增强，不作为独立测试样本。"),
        ("写作目的", "向论文写作 AI 提供可复现的方法、实验数据、指标定义、结论边界与文件位置。"),
    ], [1800, 7560])
    add_note(doc, "使用要求", "论文正文应把本文件中的“已验证结果”与“限制条件”同时写入。不得把合成模糊验证表述为真实生产现场的完全泛化结论。")

    add_heading(doc, "1. 研究问题、总体流程与创新定位")
    add_body(doc, "研究目标是建立面向缝纫线迹质量检测的三级视觉系统：先恢复运动模糊造成的细线迹可见性，再分割出连续线迹轨迹，最后根据轨迹的局部中断、长度异常和连续形变识别缺陷类型。", "研究目标")
    add_body(doc, "清晰图或模糊图首先经过 MBDNet-P+ FT2 得到复原图；M6 输出二值线迹掩码；TSR-Defect 将掩码投影为一维软轨迹，并逐段输出跳线、断线、歪斜、松线的分数与判定。", "推理链路")
    add_body(doc, "后端 TSR-Defect 与原参考文献中依赖连通域、质心、针距换算和像素-毫米标定的测量模块不同。它不以针距计算为核心，而是使用同图其他线迹单元作为鲁棒参照，分析轨迹的缺口、覆盖率、跨度与曲率。", "与参考文档的区别")
    add_table(doc, ["模块", "输入", "输出", "在整体中的作用"], [
        ("MBDNet-P+ FT2", "合成运动模糊线迹图", "去模糊 RGB 图像", "降低模糊对细线迹结构的破坏"),
        ("M6", "清晰图或去模糊图", "320 × 320 二值线迹掩码", "提供可解释的结构化线迹表示"),
        ("TSR-Defect", "M6 二值掩码", "六段单元及整图的四类缺陷判定", "完成最终缺陷类型识别"),
    ], [1800, 2500, 2200, 2860])

    add_heading(doc, "2. 数据集、标注与划分原则")
    add_body(doc, "原始数据包含 48 张清晰宏观缝纫线迹图。每张图中约有 6 段横向线迹。原始 JPEG 图像已转换为 PNG 并统一缩放至 320 × 320 px；对应的二值线迹 GT 掩码也同步处理。", "基础数据")
    add_body(doc, "每张原图按照六个线迹单元标注跳线、断线、松线；整体歪斜按图像级标注并在六段标签中保持一致。图像首段和末段可能因拍摄视野而天然较短，不能据此标记为断线。", "缺陷标签")
    add_body(doc, "48 张水平翻转图用于扩充训练或增强，但翻转图与原图属于同一来源。正式交叉验证必须以原始来源图像分组，确保一对原图/翻转图不会被分到训练侧和测试侧，避免数据泄漏。", "划分原则")
    add_table(doc, ["项目", "具体设置"], [
        ("原图与掩码", "48 个原始来源；统一为 320 × 320 px。"),
        ("增强图", "每张原图水平翻转一次，得到 96 张图像；不增加独立样本数。"),
        ("缺陷类别", "跳线 skip、断线 break、歪斜 skew、松线/浮线 loose。"),
        ("验证策略", "4 折、按 48 个来源图分组；每一折 36 张用于阈值校准、12 张用于测试。"),
        ("模糊退化", "每张清晰原图仅生成 1 张对应的合成运动模糊图，核长度随机取 3、5、7、9、11、13、15 px，角度在 ±15° 内；随机种子为 42。"),
    ], [1900, 7460])

    add_heading(doc, "3. 第一部分：MBDNet-P+ FT2 去模糊网络")
    add_body(doc, "本研究将用户确认的 MBDNet-P+ FT2 作为前端去模糊网络。在本组消融实验中使用固定权重进行推理，不针对 48 张评价图进行微调、再训练或参数选择。", "定位")
    add_body(doc, "输入是一张合成运动模糊图，输出为同尺寸的复原 RGB 图像。去模糊的价值不只用 PSNR 衡量，更重要的是观察复原图是否能使后续 M6 重新获得连续、准确的线迹掩码，并进一步提高缺陷检测结果。", "输入输出与评价逻辑")
    add_table(doc, ["图像复原指标（48 对）", "结果", "解释"], [
        ("PSNR", "28.65 dB", "复原图与清晰参考图之间的像素保真度"),
        ("SSIM", "0.8856", "结构相似性"),
        ("MAE", "0.0262", "平均绝对误差（归一化像素尺度）"),
        ("平均推理时间", "55.22 ms/图", "包含首次 CUDA 预热的均值；除首张外稳态均值约 50.69 ms/图"),
    ], [2500, 1800, 5060])
    add_note(doc, "写作限制", "MBDNet-P+ FT2 的网络内部结构、训练语料与训练损失应以该模型的实际配置和原始训练记录为准；本文件只确认当前部署权重、冻结推理协议与本实验结果。")

    add_heading(doc, "4. 第二部分：M6 线迹分割网络")
    add_body(doc, "M6 是 AFM-guided HighRes MobileNetV4-FPN 线迹分割变体。它把图像映射为 320 × 320 的线迹概率/二值掩码，为后续轨迹分析提供结构化输入。", "网络定位")
    add_body(doc, "当前 M6 配置启用了最终输出边缘区域 Dice 约束（final_edge_dice_weight = 0.2），但中心线连续性损失、拓扑 soft-clDice 和独立 Boundary-Dice 在该 M6 配置中均未启用。因此 Trajectory F1@1px 与 Boundary F1@1px 是推理评估指标，不是当前训练中直接反向传播的独立损失。", "训练约束的准确表述")
    add_table(doc, ["条件", "Binary IoU", "Binary F1", "Precision", "Recall", "Trajectory F1@1px", "Boundary F1@1px"], [
        ("清晰图 → M6", "80.45%", "89.12%", "93.02%", "85.89%", "98.04%", "95.12%"),
        ("直接模糊图 → M6", "70.76%", "82.54%", "82.01%", "84.26%", "91.51%", "86.04%"),
        ("模糊图 → MBDNet-P+ FT2 → M6", "78.89%", "88.12%", "92.87%", "84.29%", "97.34%", "93.95%"),
    ], [1450, 1050, 1050, 1050, 1050, 1800, 1900])
    add_body(doc, "直接模糊使分割 IoU 相比清晰条件下降 9.69 个百分点，去模糊后回升 8.13 个百分点；轨迹 F1@1px 从 91.51% 回升至 97.34%。这证明去模糊能恢复 M6 所依赖的细线迹结构。", "分割层结论")

    add_heading(doc, "5. 第三部分：TSR-Defect 缺陷检测单元")
    add_body(doc, "TSR-Defect（Trajectory Self-Reference Defect Reasoner）是一个可解释的后端规则检测单元，而不是在当前小样本上额外训练的深度分类器。它的直接输入为 M6 的二值线迹掩码；原 RGB 图仅用于可视化审阅，不参与最终规则决策。", "输入与性质")
    add_body(doc, "系统首先按列计算线迹前景的加权纵坐标，形成一维软轨迹；随后估计并去除整图线性趋势，以避免全局斜率干扰局部形变判断。六个线迹单元优先由轨迹支持区域中的谷值自适应定位，而非强制六等分。这样能保留由于拍摄裁切造成的首段、末段较短现象。", "轨迹与自适应分段")
    add_table(doc, ["缺陷类型", "核心证据", "判定逻辑"], [
        ("跳线 skip", "内部缺口、覆盖率下降，但后续存在轨迹恢复", "强调局部支持中断与恢复；相机裁切的首末段不作为缺陷。"),
        ("断线 break", "部分轨迹的跨度/覆盖率明显小于同图其他单元", "要求保留部分直线残余；对高曲率残余进行抑制，降低与跳线或松线混淆。"),
        ("松线 loose", "轨迹连续但相对同图参考出现较大弯曲、偏移、弧长或 U/S 形变", "连续性门控后，综合残差偏移、弯曲能量、下垂量与弧长超额。"),
        ("歪斜 skew", "整图线迹的全局斜率", "作为图像级标签和判定；适用于已排除拍摄姿态影响的采集条件。"),
    ], [1500, 3300, 4560])
    add_body(doc, "阈值并非对每个条件单独调节。每个外层折仅使用 36 张清晰校准图选择四类缺陷的 F1 最优阈值，再把该阈值冻结后用于该折的清晰、直接模糊和去模糊条件。这样，三种条件间的差异能够归因于输入质量与去模糊，而不是阈值变化。", "公平比较协议")
    add_table(doc, ["清晰 M6 掩码上的 TSR-Defect 结果", "Accuracy", "Precision", "Recall", "F1"], [
        ("跳线", "98.61%", "78.57%", "91.67%", "84.62%"),
        ("断线", "98.61%", "100.00%", "66.67%", "80.00%"),
        ("歪斜（图像级）", "93.75%", "90.91%", "83.33%", "86.96%"),
        ("松线", "98.61%", "100.00%", "66.67%", "80.00%"),
        ("宏平均 F1", "-", "-", "-", "82.89%"),
    ], [3000, 1500, 1500, 1500, 1860])
    add_body(doc, "F1 是 Precision 与 Recall 的调和平均：F1 = 2PR/(P+R)。在正常线迹明显多于缺陷线迹的情况下，仅使用 Accuracy 会被大量正确的正常判断抬高；因此论文应以各类 Precision、Recall、F1 和宏平均 F1 为主要缺陷识别指标，同时报告 Accuracy 作为辅助指标。", "指标解释")

    add_heading(doc, "6. 三阶段端到端消融：去模糊存在必要性的证据")
    add_body(doc, "必要性不是通过“复原图更接近清晰图”这一单一事实来证明，而是采用三级证据链：图像复原质量 → 线迹分割质量 → 最终缺陷识别质量。关键比较是直接模糊条件与去模糊后条件，清晰条件是理论参考上界。", "论证逻辑")
    add_table(doc, ["端到端条件", "缺陷宏平均 F1", "平均类别 Accuracy", "单元完整判定 Accuracy", "整图完整判定 Accuracy"], [
        ("清晰图 → M6 → TSR", "82.89%", "97.40%", "96.53%", "75.00%"),
        ("模糊图 → M6 → TSR", "52.66%", "91.67%", "77.43%", "35.42%"),
        ("模糊图 → MBDNet-P+ FT2 → M6 → TSR", "71.01%", "96.35%", "93.06%", "58.33%"),
    ], [2600, 1600, 1700, 1800, 1660])
    add_body(doc, "直接模糊使最终缺陷宏平均 F1 从 82.89% 降至 52.66%，下降 30.23 个百分点；去模糊后 F1 回升至 71.01%，恢复 18.35 个百分点，即恢复了约 60.7% 的模糊损失。断线 F1 由 32.00% 回升至 66.67%，松线 F1 由 25.40% 回升至 54.55%。", "端到端结论")
    if FIGURE.exists():
        doc.add_picture(str(FIGURE), width=Inches(6.42))
        caption = doc.add_paragraph()
        caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = caption.add_run("图 1  清晰、直接模糊与去模糊后三种条件下的端到端缺陷检测定性结果。黄色为 M6 线迹轨迹，粉色框为标注缺陷区域或整图歪斜范围。")
        set_font(r, size=8.8, color=MUTED)

    add_heading(doc, "7. 可直接交给论文写作 AI 的写作约束")
    add_note(doc, "论文贡献表述建议", "（1）构建 MBDNet-P+ FT2、M6 与 TSR-Defect 串联的三级线迹缺陷检测流程；（2）提出以同图自参照轨迹特征为核心的可解释缺陷判定单元；（3）通过固定阈值的三条件端到端消融，说明去模糊对分割和最终缺陷识别的贡献。")
    add_body(doc, "写作时应把跳线、断线、松线作为单元级分类任务，把歪斜作为图像级任务；不要把歪斜六次重复标签误写为六个独立样本。", "标签粒度")
    add_body(doc, "不要宣称 TSR-Defect 是端到端训练的深度分类网络。它是后分割的、基于轨迹自参照特征的规则推理单元，优势是解释性和对小样本条件的适配性。", "方法性质")
    add_body(doc, "不要以 Accuracy 单独证明缺陷检测性能。缺陷类别不均衡，必须同时报告 Precision、Recall、F1、宏平均 F1；整图完整判定 Accuracy 可补充说明实际部署时的严格正确率。", "指标使用")
    add_body(doc, "去模糊实验采用的是由清晰图生成的一对一合成运动模糊。论文应写作“在受控合成运动模糊条件下验证鲁棒性”，不应直接外推为真实工业模糊场景的全面泛化。真实模糊图的独立测试集是后续工作。", "外推边界")

    add_heading(doc, "8. 可复现文件与结果位置")
    add_path(doc, "去模糊权重", r"H:\C3Netqing\NAFNet\checkpoints\MultiSourceV2_MBDNetPPlus_FTDirGradSSIM5_S42\model_best.pth")
    add_path(doc, "去模糊推理与指标", r"H:\C3Netqing\results\deblurring_ablation\quexian_synblur48_mbdnetpplus_restored\restoration_summary.json")
    add_path(doc, "M6 权重", r"H:\C3Netqing\results\training\runs\m6-afmguided-highres-mobilenetv4-fpn-full-150-seed42_20260726_003419\checkpoints\model_best.pth")
    add_path(doc, "M6 配置", r"H:\C3Netqing\configs\ablations\m6_afm_guided_highres_clear100_seed42.yaml")
    add_path(doc, "TSR 主体", r"H:\C3Netqing\models\trajectory_self_reference_detector.py")
    add_path(doc, "TSR 配置", r"H:\C3Netqing\configs\defect_head\tsr_defect_rules.yaml")
    add_path(doc, "端到端汇总", r"H:\C3Netqing\results\deblurring_ablation\quexian_synblur48_end_to_end_tsr_v5\deblur_necessity_evidence.json")
    add_path(doc, "分割层 CSV", r"H:\C3Netqing\results\deblurring_ablation\quexian_synblur48_end_to_end_tsr_v5\deblur_necessity_segmentation.csv")
    add_path(doc, "缺陷层 CSV", r"H:\C3Netqing\results\deblurring_ablation\quexian_synblur48_end_to_end_tsr_v5\deblur_necessity_defect.csv")
    add_path(doc, "可视化图", r"H:\C3Netqing\results\deblurring_ablation\quexian_synblur48_end_to_end_tsr_v5\qualitative_end_to_end_defect_examples.png")
    add_path(doc, "缺陷标签", r"H:\DATASET\quexian\dataset_96_320\defect_review_v2\defect_labels_template.csv")
    add_path(doc, "合成模糊记录", r"H:\DATASET\quexian\dataset_96_320\synblur_original48_k3to15\degradation_protocol.json")
    add_heading(doc, "9. 供写作 AI 使用的一段核心摘要")
    add_note(doc, "可直接使用", "本文构建了由 MBDNet-P+ FT2 去模糊网络、M6 线迹分割网络和 TSR-Defect 轨迹自参照缺陷检测单元组成的三级检测系统。在受控合成运动模糊条件下，直接模糊会显著损害线迹分割与缺陷识别；经去模糊后，M6 的分割 IoU 由 70.76% 回升至 78.89%，最终缺陷宏平均 F1 由 52.66% 回升至 71.01%。TSR-Defect 不依赖大规模缺陷分类训练集，而是通过同图自适应分段、轨迹趋势去除及局部结构自参照，对跳线、断线、整体歪斜和松线进行可解释判定。")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    build()

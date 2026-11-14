from pathlib import Path
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.section import WD_SECTION
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_BREAK
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'ablation_experiments'
OUT.mkdir(parents=True, exist_ok=True)
FIGDIR = OUT / 'figures'
FIGDIR.mkdir(exist_ok=True)
DOCX = OUT / 'APH-StitchNet_分割网络消融实验_论文小节.docx'

NAVY = '173B63'; BLUE = '2D70B3'; TEAL = '168C83'; ORANGE = 'E47D38'; PURPLE = '7D58A6'; GOLD = 'C79A3B'
LIGHT = 'F5F8FB'; GRID = 'D6DEE8'; TEXT = '1E2934'

RESULTS = [
    ('C3Net', '原始 C3Net', 84.201, 91.393, 93.231, 89.844, 97.576, 99.086, 53.55),
    ('M0', '+ LMFR', 84.853, 91.771, 92.894, 90.895, 97.701, 99.124, 15.19),
    ('M4', '+ HSR', 85.663, 92.244, 92.783, 91.929, 97.909, 99.208, 15.65),
    ('M6', '+ APG-HR', 85.811, 92.327, 92.018, 92.879, 97.965, 99.232, 16.47),
]


def font_prop(run, name='Microsoft YaHei', size=10.5, bold=None, color=None):
    run.font.name = name
    run._element.rPr.rFonts.set(qn('w:eastAsia'), name)
    run._element.rPr.rFonts.set(qn('w:ascii'), name)
    run._element.rPr.rFonts.set(qn('w:hAnsi'), name)
    run.font.size = Pt(size)
    if bold is not None: run.bold = bold
    if color: run.font.color.rgb = RGBColor.from_string(color)


def set_cell_shading(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = tcPr.find(qn('w:shd'))
    if shd is None:
        shd = OxmlElement('w:shd'); tcPr.append(shd)
    shd.set(qn('w:fill'), fill)


def set_cell_border(cell, color=GRID, size='6'):
    tcPr = cell._tc.get_or_add_tcPr()
    borders = tcPr.first_child_found_in('w:tcBorders')
    if borders is None:
        borders = OxmlElement('w:tcBorders'); tcPr.append(borders)
    for edge in ('top','left','bottom','right'):
        tag = 'w:' + edge
        el = borders.find(qn(tag))
        if el is None: el = OxmlElement(tag); borders.append(el)
        el.set(qn('w:val'), 'single'); el.set(qn('w:sz'), size); el.set(qn('w:color'), color)


def set_cell_width(cell, width_cm):
    tcPr = cell._tc.get_or_add_tcPr()
    tcW = tcPr.find(qn('w:tcW'))
    if tcW is None:
        tcW = OxmlElement('w:tcW'); tcPr.append(tcW)
    tcW.set(qn('w:w'), str(int(width_cm / 2.54 * 1440)))
    tcW.set(qn('w:type'), 'dxa')


def finalize_table_geometry(table, widths_cm, indent_dxa=120):
    """Set explicit Word table/grid/cell geometry for deterministic layout."""
    table.autofit = False
    tblPr = table._tbl.tblPr
    tblW = tblPr.find(qn('w:tblW'))
    if tblW is None:
        tblW = OxmlElement('w:tblW'); tblPr.append(tblW)
    widths = [int(w / 2.54 * 1440) for w in widths_cm]
    tblW.set(qn('w:w'), str(sum(widths))); tblW.set(qn('w:type'), 'dxa')
    ind = tblPr.find(qn('w:tblInd'))
    if ind is None:
        ind = OxmlElement('w:tblInd'); tblPr.append(ind)
    ind.set(qn('w:w'), str(indent_dxa)); ind.set(qn('w:type'), 'dxa')
    grid = table._tbl.tblGrid
    for grid_col, width in zip(grid.gridCol_lst, widths):
        grid_col.set(qn('w:w'), str(width))
    for row in table.rows:
        for cell, width_cm in zip(row.cells, widths_cm):
            set_cell_width(cell, width_cm)


def cell_text(cell, text, bold=False, color=TEXT, size=9.2, align=WD_ALIGN_PARAGRAPH.CENTER):
    cell.text = ''
    p = cell.paragraphs[0]; p.alignment = align
    p.paragraph_format.space_after = Pt(0); p.paragraph_format.space_before = Pt(0)
    r = p.add_run(str(text)); font_prop(r, size=size, bold=bold, color=color)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def set_repeat_table_header(row):
    trPr = row._tr.get_or_add_trPr(); tblHeader = OxmlElement('w:tblHeader'); tblHeader.set(qn('w:val'), 'true'); trPr.append(tblHeader)


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run('第 '); font_prop(run, size=8.5, color='6B7280')
    fld = OxmlElement('w:fldSimple'); fld.set(qn('w:instr'), 'PAGE')
    run._r.addnext(fld)
    run = paragraph.add_run(' 页'); font_prop(run, size=8.5, color='6B7280')


def make_figures():
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    # progressive ablation structure
    fig, ax = plt.subplots(figsize=(12, 3.6), dpi=180)
    ax.set_xlim(0, 13.8); ax.set_ylim(0, 4.1); ax.axis('off')
    boxes = [
        (0.3, 1.3, 2.55, 1.35, 'Baseline\n原始 C3Net', '#EAF3FC', '#2D70B3'),
        (3.65, 1.3, 2.55, 1.35, 'M0\n+ LMFR', '#E9F7F4', '#168C83'),
        (7.0, 1.3, 2.55, 1.35, 'M4\n+ HSR', '#FFF1E7', '#E47D38'),
        (10.35, 1.3, 2.55, 1.35, 'M6\n+ APG-HR', '#F3EDFA', '#7D58A6'),
    ]
    for x,y,w,h,label,fill,edge in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06,rounding_size=0.14',fc=fill,ec=edge,lw=2.2))
        ax.text(x+w/2,y+h*0.62,label,ha='center',va='center',fontsize=15,fontweight='bold',color='#1E2934')
    for i in range(3):
        x1=boxes[i][0]+boxes[i][2]; x2=boxes[i+1][0]
        ax.add_patch(FancyArrowPatch((x1+0.10,1.98),(x2-0.10,1.98),arrowstyle='-|>',mutation_scale=18,lw=2.0,color='#5B6773'))
    notes=[('替换主干并统一尺度',4.93,'#168C83'),('恢复 320×320 细节',8.28,'#E47D38'),('注入 p / u 先验',11.63,'#7D58A6')]
    for t,x,c in notes: ax.text(x,3.25,t,ha='center',fontsize=11.5,color=c,fontweight='bold')
    ax.text(6.9,0.42,'每一步以前一模型为基础，仅增加对应模块；其余训练设置保持一致。',ha='center',fontsize=11,color='#5B6773')
    fig.tight_layout(pad=0.1); fig.savefig(FIGDIR/'ablation_progress.png',bbox_inches='tight',facecolor='white'); plt.close(fig)
    # scores chart
    names=[r[0] for r in RESULTS]; iou=[r[2] for r in RESULTS]; f1=[r[3] for r in RESULTS]; bnd=[r[6] for r in RESULTS]
    x=np.arange(len(names)); width=0.23
    fig, ax=plt.subplots(figsize=(10.8,4.5),dpi=180)
    bars1=ax.bar(x-width,iou,width,label='Binary IoU',color='#2D70B3')
    bars2=ax.bar(x,f1,width,label='Binary F1',color='#168C83')
    bars3=ax.bar(x+width,bnd,width,label='Boundary F1@1px',color='#E47D38')
    ax.set_ylim(82,99.1); ax.set_xticks(x,names); ax.set_ylabel('指标（%）'); ax.grid(axis='y',linestyle='--',alpha=.35); ax.legend(ncol=3,loc='lower center',bbox_to_anchor=(.5,1.02),frameon=False)
    for bars in (bars1,bars2,bars3):
        for b in bars: ax.text(b.get_x()+b.get_width()/2,b.get_height()+.11,f'{b.get_height():.2f}',ha='center',va='bottom',fontsize=8)
    ax.text(3,82.35,'M6 相对 C3Net：IoU +1.610，F1 +0.934，Boundary +0.389',ha='right',fontsize=9,color='#173B63',fontweight='bold')
    fig.tight_layout(); fig.savefig(FIGDIR/'metric_comparison.png',bbox_inches='tight',facecolor='white'); plt.close(fig)
    # time chart
    time=[r[8] for r in RESULTS]
    fig, ax=plt.subplots(figsize=(10.8,3.7),dpi=180)
    colors=['#2D70B3','#168C83','#E47D38','#7D58A6']; b=ax.bar(names,time,color=colors,width=.58)
    ax.set_ylabel('单张推理时间（ms，batch=1）'); ax.set_ylim(0,62); ax.grid(axis='y',linestyle='--',alpha=.35)
    for k,v in zip(b,time): ax.text(k.get_x()+k.get_width()/2,v+1,f'{v:.2f}',ha='center',fontsize=10,fontweight='bold')
    ax.text(1.5,57,'M6 相比原始 C3Net：−37.08 ms / −69.2%',ha='center',fontsize=10,color='#173B63',fontweight='bold')
    fig.tight_layout(); fig.savefig(FIGDIR/'time_comparison.png',bbox_inches='tight',facecolor='white'); plt.close(fig)


def add_caption(doc, text):
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before=Pt(3); p.paragraph_format.space_after=Pt(8)
    r=p.add_run(text); font_prop(r,size=9.3,color='4B5563')


def add_para(doc, text, first=True, italic=False, color=TEXT):
    p=doc.add_paragraph(); p.paragraph_format.line_spacing=1.35; p.paragraph_format.space_after=Pt(6)
    if first: p.paragraph_format.first_line_indent=Cm(.74)
    r=p.add_run(text); font_prop(r,size=10.5,color=color); r.italic=italic
    return p


def add_callout(doc, title, text, fill='F3F7FB'):
    table=doc.add_table(rows=1,cols=1); table.alignment=WD_TABLE_ALIGNMENT.CENTER; table.autofit=False
    cell=table.cell(0,0); set_cell_width(cell,16.0); set_cell_shading(cell,fill); set_cell_border(cell,'B8C7D9','8')
    cell.text=''; p=cell.paragraphs[0]; p.paragraph_format.space_before=Pt(5); p.paragraph_format.space_after=Pt(5)
    r=p.add_run(title+'：'); font_prop(r,size=10.3,bold=True,color=NAVY)
    r=p.add_run(text); font_prop(r,size=10.3,color=TEXT)
    finalize_table_geometry(table,[16.0])
    doc.add_paragraph().paragraph_format.space_after=Pt(1)


def add_results_table(doc):
    headers=['编号','模型/增量','Binary\nIoU','Binary\nF1','Precision','Recall','Boundary\nF1@1px','Trajectory\nF1@1px','时间\n(ms)']
    widths=[1.0,2.6,1.25,1.25,1.2,1.2,1.55,1.55,1.25]
    table=doc.add_table(rows=1,cols=len(headers)); table.alignment=WD_TABLE_ALIGNMENT.CENTER; table.autofit=False
    table.style='Table Grid'; set_repeat_table_header(table.rows[0])
    for c,h,w in zip(table.rows[0].cells,headers,widths):
        set_cell_width(c,w); set_cell_shading(c,'E8EEF5'); set_cell_border(c); cell_text(c,h,True,NAVY,8.7)
    for name,mod,iou,f1,pre,rec,bnd,traj,time in RESULTS:
        cells=table.add_row().cells
        values=[name,mod,f'{iou:.3f}',f'{f1:.3f}',f'{pre:.3f}',f'{rec:.3f}',f'{bnd:.3f}',f'{traj:.3f}',f'{time:.2f}']
        for j,(c,v,w) in enumerate(zip(cells,values,widths)):
            set_cell_width(c,w); set_cell_border(c)
            if name=='M6': set_cell_shading(c,'EDF7F5')
            cell_text(c,v,bold=(name=='M6' or j==0),color=TEXT,size=8.5)
    finalize_table_geometry(table,widths)
    return table


def main():
    make_figures()
    doc=Document()
    sec=doc.sections[0]
    sec.page_width=Cm(21); sec.page_height=Cm(29.7)
    sec.top_margin=Cm(2.3); sec.bottom_margin=Cm(2.2); sec.left_margin=Cm(2.5); sec.right_margin=Cm(2.5)
    sec.header_distance=Cm(1.2); sec.footer_distance=Cm(1.2)
    # styles
    normal=doc.styles['Normal']; normal.font.name='Microsoft YaHei'; normal._element.rPr.rFonts.set(qn('w:eastAsia'),'Microsoft YaHei'); normal.font.size=Pt(10.5)
    normal.paragraph_format.line_spacing=1.35; normal.paragraph_format.space_after=Pt(6)
    for style_name,size,col,before,after in [('Heading 1',15,NAVY,14,8),('Heading 2',12.5,BLUE,11,6),('Heading 3',11,PURPLE,8,4)]:
        st=doc.styles[style_name]; st.font.name='Microsoft YaHei'; st._element.rPr.rFonts.set(qn('w:eastAsia'),'Microsoft YaHei'); st.font.size=Pt(size); st.font.bold=True; st.font.color.rgb=RGBColor.from_string(col); st.paragraph_format.space_before=Pt(before); st.paragraph_format.space_after=Pt(after); st.paragraph_format.keep_with_next=True
    # header/footer
    hp=sec.header.paragraphs[0]; hp.alignment=WD_ALIGN_PARAGRAPH.RIGHT; hr=hp.add_run('APH-StitchNet 线迹分割网络｜消融实验'); font_prop(hr,size=8.5,color='6B7280')
    add_page_number(sec.footer.paragraphs[0])
    # title
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_before=Pt(4); p.paragraph_format.space_after=Pt(4)
    r=p.add_run('4.X 分割网络消融实验与分析'); font_prop(r,size=18,bold=True,color=NAVY)
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_after=Pt(14)
    r=p.add_run('以原始 C3Net 为基线的逐步结构验证（当前实验结果汇总版）'); font_prop(r,size=10.5,color='5B6773')
    add_callout(doc,'阅读说明','本节依据当前已完成的全量训练与推理记录撰写。四项数据集（DB、XJ、SD、RX）按宏平均汇总；RX 始终仅用于外部测试，不参与训练、阈值选择或模型选择。')
    doc.add_heading('4.X.1 消融目标与递进设计',level=1)
    add_para(doc,'为验证所提出网络中各结构的独立贡献，消融实验以原始 C3Net 为起点，按照“轻量跨尺度表征重建—高分辨率细化—AFM 概率—不确定性先验引导”的顺序逐步引入模块。每一步均在前一模型的基础上增加一个清晰可归因的结构单元，保持输入尺寸、训练数据划分、损失组合、优化器和评估流程一致，从而使性能变化可对应到新增模块。')
    doc.add_picture(str(FIGDIR/'ablation_progress.png'),width=Cm(15.7)); add_caption(doc,'图 4-X 逐步消融结构关系。M0、M4 和 M6 分别对应三项结构改进的累积版本。')
    add_para(doc,'具体而言，M0 采用 LMFR 替换原始较重的编码—解码表征链路，以 MobileNetV4 Hybrid Medium 提取四级特征，并通过 192 通道 Mobile-FPN 完成自顶向下重建；M4 在 M0 上增加 HSR，将边缘、定位与 RGB 细节恢复至 320×320；M6 进一步将 AFM 的粗分割结果编码为概率先验 p 与不确定性先验 u，使高分辨率细化在高置信区域保持语义约束、在模糊区域强化局部修正。')
    doc.add_heading('4.X.2 实验设置与评价协议',level=1)
    add_para(doc,'所有模型输入均为 320×320 RGB 织物图像，采用相同的数据预处理与训练轮次设置（全量数据、150 轮、随机种子 42）。训练集由 DB、XJ 和 SD 构成，RX 作为纯外部泛化测试集；最终统计在 DB、XJ、SD 与 RX 四个数据集上分别计算后取宏平均。为兼顾区域完整性与极细线迹边界质量，主指标采用 Binary IoU，同时报告 Binary F1、Precision、Recall、Boundary F1@1px、Trajectory F1@1px 和 batch=1 单张推理时间。')
    add_callout(doc,'模型选择一致性','正式论文定稿时，应保证四个模型采用同一种验证集监控指标保存最佳权重，建议统一为验证集 Binary IoU；若当前历史运行采用了不同监控指标，应以统一规则补跑或在表注中如实说明。','FFF8DD')
    doc.add_heading('4.X.3 定量结果与总体比较',level=1)
    add_results_table(doc)
    add_caption(doc,'表 4-X 不同消融模型在四个数据集宏平均上的性能比较。加粗表示当前表内最优值。时间为 batch=1 下单张推理平均时间。')
    doc.add_picture(str(FIGDIR/'metric_comparison.png'),width=Cm(15.6)); add_caption(doc,'图 4-X 区域、像素与边界指标的逐步变化。')
    doc.add_picture(str(FIGDIR/'time_comparison.png'),width=Cm(15.4)); add_caption(doc,'图 4-X 单张推理时间比较。原始 C3Net 与各改进模型应使用相同硬件、batch=1 与预热策略测量。')
    doc.add_heading('4.X.4 创新点一：LMFR 的消融分析',level=1)
    add_para(doc,'由 C3Net 到 M0，Binary IoU 从 84.201% 提升至 84.853%，增幅为 0.652 个百分点；Binary F1 同步提高 0.378 个百分点，Recall 提高 1.051 个百分点。该结果说明，以轻量骨干和统一通道 FPN 重建多尺度特征后，模型对弱线迹与细窄区域的召回更充分。与此同时，单张推理时间由 53.55 ms 降至 15.19 ms，表明 LMFR 不只是精度替换，也显著降低了编码—解码链路的计算开销。')
    doc.add_picture(str(ROOT/'outputs'/'m6_network_vectors'/'preview_fig_innovation1_lmfr_detailed.png'),width=Cm(15.7)); add_caption(doc,'图 4-X LMFR 的结构展开与其在消融中的对应位置。')
    add_para(doc,'值得注意的是，M0 的 Precision 较基线略降 0.337 个百分点，而 Recall 增幅更大，说明轻量跨尺度重建更倾向于减少弱线迹漏检；其带来的假阳性变化需要通过后续细化模块进一步约束。因此，LMFR 的贡献应表述为“提高细线可见性并显著改善效率”，而不宜夸大为单独解决全部背景纹理干扰。')
    doc.add_heading('4.X.5 创新点二：HSR 的消融分析',level=1)
    add_para(doc,'在 M0 上加入 HSR 后，M4 的 Binary IoU 达到 85.663%，相对 M0 增加 0.810 个百分点；Binary F1 增加 0.473 个百分点，Recall 增加 1.034 个百分点，Boundary F1@1px 增加 0.208 个百分点。上述结果表明，当低分辨率定位语义与边缘特征被投影并上采样到原图尺度后，模型能够更稳定地恢复线迹宽度、断裂处连续性和边界贴合性。')
    doc.add_picture(str(ROOT/'outputs'/'m6_network_vectors'/'preview_fig_innovation2_hsr_detailed.png'),width=Cm(15.7)); add_caption(doc,'图 4-X HSR 的结构展开。边缘、定位和 RGB 细节在 320×320 尺度融合。')
    add_para(doc,'从效率看，M4 相比 M0 仅增加 0.46 ms，说明 1×1 投影、双线性上采样与深度可分离细化能够在较低代价下实现高分辨率恢复。因而，HSR 的主要消融结论是：它直接改善了细窄线迹的区域召回与边界质量，并未显著破坏轻量化推理优势。')
    doc.add_heading('4.X.6 创新点三：APG-HR 的消融分析',level=1)
    add_para(doc,'M6 在 M4 的基础上进一步引入 APG-HR。AFM 粗分割 logits 经过 Sigmoid 形成概率先验 p，并由 p 构造不确定性先验 u；二者与高分辨率边缘、定位和 RGB 细节共同输入细化头。相较 M4，M6 的 Binary IoU 继续提高 0.148 个百分点，Binary F1 提高 0.083 个百分点，Boundary F1@1px 提高 0.056 个百分点，Trajectory F1@1px 提高 0.024 个百分点；最终 M6 在表中取得最高的 IoU、F1、Boundary F1 与 Trajectory F1。')
    doc.add_picture(str(ROOT/'outputs'/'m6_network_vectors'/'preview_fig_innovation3_apg_hr_detailed.png'),width=Cm(15.7)); add_caption(doc,'图 4-X APG-HR 的结构展开。p 与 u 是细化头的显式输入，而非另行增加的预测网络。')
    add_para(doc,'该模块的增益虽小于 LMFR 与 HSR，但变化方向具有明确的结构解释：M6 的 Recall 相对 M4 增加 0.950 个百分点，而 Precision 降低 0.765 个百分点，说明先验引导更倾向于在 AFM 不确定区域保留并校正弱线迹细节，从而降低漏检风险。因此，论文中应将 APG-HR 定位为“可置信度约束的局部校正机制”，而非笼统表述为全面抑制纹理噪声。其时间相对 M4 增加 0.82 ms，仍显著低于原始 C3Net。')
    doc.add_heading('4.X.7 消融结论、局限与待补充可视化',level=1)
    add_para(doc,'综合来看，三项结构形成连续增益：LMFR 主要完成高效多尺度表征重建，HSR 负责原图尺度的细线与边界恢复，APG-HR 则利用 AFM 的概率—不确定性信息约束细化过程。最终 M6 相对原始 C3Net 的 Binary IoU 提升 1.610 个百分点、Binary F1 提升 0.934 个百分点、Boundary F1@1px 提升 0.389 个百分点，同时单张推理时间缩短 37.08 ms（69.2%）。')
    add_callout(doc,'论文表述边界','本节目前为种子 42 的单次全量训练汇总，不能据此声称具有统计显著性。若版面和时间允许，应至少对最终 M6 增加 2 个随机种子复现，并报告均值±标准差；若不补跑，应如实写为“在当前随机种子和统一协议下的比较结果”。','FFF1E7')
    doc.add_heading('4.X.8 定性可视化结果（待替换）',level=1)
    add_para(doc,'请从 DB、XJ、SD 与 RX 中各选取 1—2 张具有代表性的样本，按“输入图像—GT—C3Net—M0—M4—M6”六列排版。建议优先覆盖：弱对比白线、复杂织纹、装饰性亮点/金粉干扰、线迹交叉或宽度变化四类场景。图注中只描述可观察到的现象，例如“减少断裂”“边界更贴合”或“抑制局部误检”，避免使用没有统计依据的绝对化措辞。')
    table=doc.add_table(rows=2,cols=6); table.alignment=WD_TABLE_ALIGNMENT.CENTER; table.autofit=False
    labs=['输入图像','GT','C3Net','M0','M4','M6']
    for c,l in zip(table.rows[0].cells,labs): set_cell_width(c,2.62); set_cell_shading(c,'E8EEF5'); set_cell_border(c); cell_text(c,l,True,NAVY,9)
    for c in table.rows[1].cells:
        set_cell_width(c,2.62); set_cell_border(c); c.text=''; p=c.paragraphs[0]; p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_before=Pt(35); p.paragraph_format.space_after=Pt(35); r=p.add_run('替换为对应预测图'); font_prop(r,size=8.5,color='7A8795')
    finalize_table_geometry(table,[2.62]*6)
    add_caption(doc,'图 4-X 定性对比排版模板。请将六列同一样本的图像替换后再用于论文正文。')
    doc.save(DOCX)
    print(DOCX)


if __name__=='__main__': main()

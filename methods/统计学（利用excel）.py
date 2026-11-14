import numpy as np
import pandas as pd
from scipy import stats


def calculate_statistics_for_color(color_name, actual_list, detect_list):
    """
    计算特定颜色下的统计学指标
    """
    actual = np.array(actual_list)
    detect = np.array(detect_list)

    # 1. 计算 MAE
    abs_errors = np.abs(detect - actual)
    mae = np.mean(abs_errors)

    # 2. 计算 MAE 的 95% 置信区间 (95% CI)
    n = len(abs_errors)
    if n > 1:
        se = stats.sem(abs_errors)  # 标准误
        h = se * stats.t.ppf((1 + 0.95) / 2., n - 1)
        ci_lower, ci_upper = mae - h, mae + h
    else:
        ci_lower, ci_upper = mae, mae

    # 3. 计算 Pearson 相关系数 r 和 p-value
    if n > 1:
        r_stat, p_value = stats.pearsonr(detect, actual)
    else:
        r_stat, p_value = 0.0, 1.0

    # 打印格式化结果 (统一保留三位小数)
    print(f"--- 颜色: {color_name} (样本数: {n}) ---")
    print(f"MAE: {mae:.3f}")
    print(f"95% CI of MAE: [{ci_lower:.3f}, {ci_upper:.3f}]")
    print(f"Pearson r: {r_stat:.3f}")

    if p_value < 0.001:
        print(f"p-value: < 0.001")
    else:
        print(f"p-value: {p_value:.3f}")
    print("-" * 30)


# ==========================================
# 以下为自动化读取 Excel 表格并计算的代码
# ==========================================

# 1. 读取 Excel 文件 (.xlsx)
file_path = "E:/Line/TigerSewDataset(TSD)/tongji.xlsx"  # 确保文件名与你的实际文件名一致

try:
    # 读取 xlsx 必须用 read_excel，不需要 encoding 参数
    df = pd.read_excel(file_path)
except Exception as e:
    raise ValueError(f"读取 Excel 失败！请确保安装了 openpyxl 库。错误详情: {e}")

# 2. 定义你的 8 个颜色名称
# 【非常重要】：请确保这里的颜色顺序，与你表格中从左到右的列组顺序完全一一对应！
colors = ['Blue', 'Brown', 'Green', 'Purple', 'Red', 'ZPink', 'ZWhite', 'ZYellow']

actual_all = []
detect_all = []

# 3. 遍历颜色，每次取出两列
for i, color in enumerate(colors):

    # 按照你的要求：第一列为真实值(True)，第二列为测试值(Detect)
    col_idx_true = 2 * i  # 第 0, 2, 4... 列
    col_idx_detect = 2 * i + 1  # 第 1, 3, 5... 列

    # 防止因为 Excel 中的列数不足而报错中断
    if col_idx_detect >= df.shape[1]:
        print(f"\n⚠️ 警告: Excel 文件中的列数不足，只处理了前 {i} 个颜色。")
        break

    # 提取当前颜色的真实值和预测值
    # 加入 pd.to_numeric(..., errors='coerce') 的作用是：万一表格里混入了表头文字或空格，强制转为空值，再被 dropna() 完美剔除
    actual_data = pd.to_numeric(df.iloc[:, col_idx_true], errors='coerce').dropna().values.tolist()
    detect_data = pd.to_numeric(df.iloc[:, col_idx_detect], errors='coerce').dropna().values.tolist()

    # 确保两列的数据量是对齐的（防止错位，按数量较少的截断）
    min_len = min(len(detect_data), len(actual_data))
    if min_len == 0:
        print(f"⚠️ 警告: 颜色 {color} 对应列没有有效数据，跳过计算。")
        continue

    actual_data = actual_data[:min_len]
    detect_data = detect_data[:min_len]

    # 调用计算函数
    calculate_statistics_for_color(color, actual_data, detect_data)

    # 收集进汇总列表，用于最后计算 Overall
    actual_all.extend(actual_data)
    detect_all.extend(detect_data)

# 4. 计算所有颜色的总计 (Overall)
if actual_all and detect_all:
    print("\n" + "=" * 30)
    print("开始计算所有数据集的总平均统计...")
    calculate_statistics_for_color("Overall (总计)", actual_all, detect_all)
import pandas as pd

# 替换为你的真实 xlsx 文件路径
file_path = 'E:/Line/TigerSewDataset(TSD)/tongji.xlsx'

try:
    # 直接读取 Excel 文件（不需要猜编码，绝对不会乱码）
    df = pd.read_excel(file_path)
    print("✅ 成功读取 Excel 文件！\n")
except Exception as e:
    raise ValueError(f"❌ 读取失败，请确保文件没被其他程序占用，且安装了 openpyxl 库。错误信息：{e}")

# 遍历每一列提取数据
for i in range(df.shape[1]):
    col_name = df.columns[i]

    # 提取当前列数据：去除空值(NaN) -> 转换为字符串列表
    col_data = df.iloc[:, i].dropna().astype(str).tolist()

    # 如果这一列全是空的，跳过
    if not col_data:
        continue

    # 将列表中的所有元素用逗号隔开
    joined_string = ",".join(col_data)

    print(f"=== 第 {i + 1} 列 (表头名: {col_name}) ===")
    print(joined_string)
    print("-" * 50)
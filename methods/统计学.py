import numpy as np
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

    # 打印格式化结果
    print(f"--- 颜色: {color_name} (样本数: {n}) ---")
    print(f"MAE: {mae:.3f}")
    print(f"95% CI of MAE: [{ci_lower:.3f}, {ci_upper:.3f}]")
    print(f"Pearson r: {r_stat:.3f}")

    if p_value < 0.001:
        print(f"p-value: < 0.001")
    else:
        print(f"p-value: {p_value:.4f}")
    print("-" * 30)


# ==========================================
# 在这里填入您某个颜色下的所有数据 (别只填表里显示的这几个，要把省略号里的也填进去)
# 提示：Actual 1 和 2 如果一样，随便取一列就行；Detect 同理
# ==========================================

# 示例：Blue 颜色的数据
actual_blue = [3.503, 3.618, 3.474, 3.470, 3.521]  # 把 Blue 所有的真实针距放进来
detect_blue = [3.308, 3.515, 3.491, 3.415, 3.367]  # 把 Blue 对应的预测针距放进来
calculate_statistics_for_color("Blue", actual_blue, detect_blue)

# 示例：Brown 颜色的数据
actual_brown = [3.821, 3.712, 3.737, 3.565, 3.498]
detect_brown = [3.694, 3.848, 3.676, 3.709, 3.401]
calculate_statistics_for_color("Brown", actual_brown, detect_brown)

# 示例：Green 颜色的数据
actual_Green = [3.821, 3.712, 3.737, 3.565, 3.498]
detect_Green = [3.694, 3.848, 3.676, 3.709, 3.401]
calculate_statistics_for_color("Green", actual_Green, detect_Green)

# 示例：Purple 颜色的数据
actual_Purple = [3.821, 3.712, 3.737, 3.565, 3.498]
detect_Purple = [3.694, 3.848, 3.676, 3.709, 3.401]
calculate_statistics_for_color("Purple", actual_Purple, detect_Purple)

# 示例：Red 颜色的数据
actual_Red = [3.821, 3.712, 3.737, 3.565, 3.498]
detect_Red = [3.694, 3.848, 3.676, 3.709, 3.401]
calculate_statistics_for_color("Red", actual_Red, detect_Red)

# 示例：Pink 颜色的数据
actual_Pink = [3.821, 3.712, 3.737, 3.565, 3.498]
detect_Pink = [3.694, 3.848, 3.676, 3.709, 3.401]
calculate_statistics_for_color("Pink", actual_Pink, detect_Pink)

# 示例：White 颜色的数据
actual_White = [3.821, 3.712, 3.737, 3.565, 3.498]
detect_White = [3.694, 3.848, 3.676, 3.709, 3.401]
calculate_statistics_for_color("White", actual_White, detect_White)

# 示例：Yellow 颜色的数据
actual_Yellow = [3.821, 3.712, 3.737, 3.565, 3.498]
detect_Yellow = [3.694, 3.848, 3.676, 3.709, 3.401]
calculate_statistics_for_color("Yellow", actual_Yellow, detect_Yellow)



# 示例：Overall (把所有颜色的数据拼接在一起算一次总的)
actual_all = actual_blue + actual_brown + actual_Green +actual_Purple + actual_Red +actual_Pink +actual_White +actual_Yellow# + actual_green + ...
detect_all = detect_blue + detect_brown + detect_Green +detect_Purple + detect_Red +detect_Pink +detect_White +detect_Yellow # + detect_green + ...
calculate_statistics_for_color("Overall", actual_all, detect_all)
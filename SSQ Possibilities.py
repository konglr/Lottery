import itertools
import pandas as pd
from tqdm import tqdm  # 导入 tqdm 库

def calculate_combinations_step_by_step():
    """
    逐步计算 C(33, 6) 并打印中间结果。
    """
    n = 33
    k = 6

    numerator_parts = [n - i for i in range(k)]
    denominator_parts = [i + 1 for i in range(k)]

    numerator = 1
    print("分子部分:", end=" ")
    for part in numerator_parts:
        numerator *= part
        print(part, end=" x " if part != numerator_parts[-1] else " ")
    print(f"= {numerator}")

    denominator = 1
    print("分母部分:", end=" ")
    for part in denominator_parts:
        denominator *= part
        print(part, end=" x " if part != denominator_parts[-1] else " ")
    print(f"= {denominator}")

    combinations_count = numerator // denominator
    print(f"C({n}, {k}) = {numerator} / {denominator}")
    print(f"         = {combinations_count}")
    return combinations_count

def is_arithmetic_progression(nums):
    """
    检查列表中是否有超过3个数字构成等差数列 (即 4个或更多)。
    """
    if len(nums) < 4: # 修改为检查至少4个数字
        return False
    nums_sorted = sorted(nums)
    for i in range(len(nums_sorted) - 3): # 修改循环范围
        for j in range(i + 1, len(nums_sorted) - 2): # 修改循环范围
            diff = nums_sorted[j] - nums_sorted[i]
            count = 2
            current_val = nums_sorted[j]
            for k in range(j + 1, len(nums_sorted)):
                if nums_sorted[k] == current_val + diff:
                    count += 1
                    current_val = nums_sorted[k]
            if count > 3: # 修改为 count > 3，排除4个及以上
                return True
    return False

def is_geometric_progression(nums):
    """
    检查列表中是否有超过3个数字构成等比数列 (即 4个或更多)。
    (简化版，只考虑整数等比，且排除公比为1的情况，因为公比为1的等比数列也是等差数列)
    """
    if len(nums) < 4: # 修改为检查至少4个数字
        return False
    nums_sorted = sorted(nums)
    for i in range(len(nums_sorted) - 3): # 修改循环范围
        for j in range(i + 1, len(nums_sorted) - 2): # 修改循环范围
            if nums_sorted[i] == 0: # 避免除以零
                continue
            ratio = nums_sorted[j] / nums_sorted[i]
            if ratio == 1: # 排除公比为1的情况，避免重复计算等差数列
                continue
            if ratio != int(ratio): # 只考虑整数等比
                continue
            ratio = int(ratio)

            count = 2
            current_val = nums_sorted[j]
            for k in range(j + 1, len(nums_sorted)):
                if nums_sorted[i] != 0 and nums_sorted[k] == current_val * ratio:
                    count += 1
                    current_val = nums_sorted[k]
            if count > 3: # 修改为 count > 3，排除4个及以上
                return True
    return False


def has_consecutive_numbers(nums):
    """
    检查列表中是否有超过3个连号 (即 4个或更多)。
    """
    if len(nums) < 4: # 修改为检查至少4个数字
        return False
    nums_sorted = sorted(nums)
    max_consecutive_count = 1 # 初始化最大连号计数为1 (因为至少有1个数字)
    current_consecutive_count = 1

    for i in range(len(nums_sorted) - 1):
        if nums_sorted[i+1] == nums_sorted[i] + 1:
            current_consecutive_count += 1
        else:
            max_consecutive_count = max(max_consecutive_count, current_consecutive_count) # 更新最大计数
            current_consecutive_count = 1 # 重置当前计数

    max_consecutive_count = max(max_consecutive_count, current_consecutive_count) # 最后再更新一次，处理末尾的连号

    return max_consecutive_count > 3 # 修改为 max_consecutive_count > 3， 排除4个及以上连号

def has_multiple_consecutive_sets(nums):
    """
    检查是否同时存在 3个连号，且另外还有其他连号组合（无论多少个连号）。
    """
    if len(nums) < 3:
        return False

    nums_sorted = sorted(nums)
    consecutive_sets_count = 0
    has_three_consecutive = False
    current_consecutive_count = 1

    for i in range(len(nums_sorted) - 1):
        if nums_sorted[i+1] == nums_sorted[i] + 1:
            current_consecutive_count += 1
        else:
            if current_consecutive_count >= 2: # 只要是连号（至少2个），就算作一个连号组合
                consecutive_sets_count += 1
                if current_consecutive_count == 3: # 检查是否有正好3个连号的组合
                    has_three_consecutive = True
            current_consecutive_count = 1 # 重置计数器

    if current_consecutive_count >= 2: # 检查末尾的连号组合
        consecutive_sets_count += 1
        if current_consecutive_count == 3:
            has_three_consecutive = True

    return has_three_consecutive and consecutive_sets_count >= 2 # 只有当有至少两个连号组合，且其中一个是3连号时，才排除


def has_three_double_consecutive_sets(nums):
    """
    检查是否包含 3个或更多双连号 (例如 [1,2], [4,5], [7,8] 算3个双连号).
    """
    if len(nums) < 4: # 双连号至少需要4个数字
        return False

    nums_sorted = sorted(nums)
    double_consecutive_sets_count = 0

    i = 0
    while i < len(nums_sorted) - 1:
        if nums_sorted[i+1] == nums_sorted[i] + 1: # 发现双连号
            double_consecutive_sets_count += 1
            i += 2 # 跳过已计入双连号的两个数字
        else:
            i += 1 # 继续检查下一个数字

    return double_consecutive_sets_count >= 3 # 检查双连号数量是否达到3个或更多


def is_valid_ratio(nums, ratio_type, valid_ratios):
    """
    检查奇偶比例或大小号比例是否在允许范围内。
    ratio_type: 'odd_even' 或 'small_large'
    valid_ratios: 允许的比例列表，例如 [(2, 4), (3, 3), (4, 2)]
    """
    if ratio_type == 'odd_even':
        odd_count = sum(1 for num in nums if num % 2 != 0)
        even_count = len(nums) - odd_count
        current_ratio = (odd_count, even_count)
    elif ratio_type == 'small_large':
        small_count = sum(1 for num in nums if num <= 16)
        large_count = len(nums) - small_count
        current_ratio = (small_count, large_count)
    else:
        raise ValueError("ratio_type must be 'odd_even' or 'small_large'")

    return current_ratio in valid_ratios

def is_valid_sum_range(nums, min_sum=30, max_sum=170):
    """
    检查红球号码的和是否在指定范围内。
    """
    ball_sum = sum(nums)
    return min_sum <= ball_sum <= max_sum


def generate_valid_combinations_dataframe():
    """
    生成所有C(33, 6)组合，并筛选符合条件的组合，存储到 Pandas DataFrame 中, 并统计每种条件排除的组合数量, 增加进度条显示。
    """
    all_combinations = list(itertools.combinations(range(1, 34), 6)) # 生成所有红球组合并转换为列表
    initial_combinations_count = len(all_combinations)
    print(f"初始组合总数 (C(33, 6)): {initial_combinations_count}")

    valid_combinations = []
    valid_odd_even_ratios = [(2, 4), (3, 3), (4, 2)] # 奇偶比例保持不变
    valid_small_large_ratios = [(3, 3), (2, 4)] # 修改大小号比例 (冷热号比例) 为 3:3 和 2:4

    # 初始化计数器
    arithmetic_geometric_count = 0
    consecutive_count = 0
    invalid_odd_even_count = 0
    invalid_small_large_count = 0
    invalid_sum_range_count = 0 # 新增和值范围计数器
    multiple_consecutive_sets_count = 0 # 新增 “3连号+其他连号” 计数器
    three_double_consecutive_sets_count = 0 # 新增 “3个双连号” 计数器


    # 使用 tqdm 添加进度条
    for combination in tqdm(all_combinations, desc="筛选组合"):
        is_valid = True # 假设组合有效，除非被排除条件否定

        if is_arithmetic_progression(combination) or is_geometric_progression(combination):
            arithmetic_geometric_count += 1
            is_valid = False
        if has_consecutive_numbers(combination):
            consecutive_count += 1
            is_valid = False
        if not is_valid_ratio(combination, 'odd_even', valid_odd_even_ratios):
            invalid_odd_even_count += 1
            is_valid = False
        if not is_valid_ratio(combination, 'small_large', valid_small_large_ratios):
            invalid_small_large_count += 1
            is_valid = False
        if not is_valid_sum_range(combination): # 新增和值范围检查
            invalid_sum_range_count += 1
            is_valid = False
        if has_multiple_consecutive_sets(combination): # 新增 “3连号+其他连号” 检查
            multiple_consecutive_sets_count += 1
            is_valid = False
        if has_three_double_consecutive_sets(combination): # 新增 “3个双连号” 检查
            three_double_consecutive_sets_count += 1
            is_valid = False


        if is_valid: # 只有当组合通过所有检查时才加入有效列表
            valid_combinations.append(list(combination))

    valid_combinations_count = len(valid_combinations)
    print(f"符合所有条件的组合数量: {valid_combinations_count}")
    print(f"总共减少了 {initial_combinations_count - valid_combinations_count} 种组合")

    print("\n各种条件排除的组合数量统计:")
    print(f"  - 包含4个或以上数字的等差或等比数列: {arithmetic_geometric_count} 种") # 修改了描述
    print(f"  - 包含4个或以上连号: {consecutive_count} 种") # 修改了描述
    print(f"  - 奇偶比例不符合 2:4, 3:3, 4:2: {invalid_odd_even_count} 种")
    print(f"  - 大小号比例 (冷热号比例) 不符合 3:3, 2:4: {invalid_small_large_count} 种") # 修改了描述
    print(f"  - 红球和值小于30或大于170: {invalid_sum_range_count} 种") # 新增和值范围排除统计
    print(f"  - 同时包含3连号和另一组连号: {multiple_consecutive_sets_count} 种") # 新增统计
    print(f"  - 包含3个或更多双连号: {three_double_consecutive_sets_count} 种") # 新增统计


    df_valid_combinations = pd.DataFrame(valid_combinations, columns=['红球1', '红球2', '红球3', '红球4', '红球5', '红球6'])
    print(f"\nDataFrame 的形状: {df_valid_combinations.shape}") # 打印DataFrame的形状，确认行数和列数
    return df_valid_combinations

if __name__ == "__main__":
    print("计算 C(33, 6) 的步骤:")
    calculate_combinations_step_by_step()
    print("\n开始生成并筛选组合，请稍候...")
    valid_df = generate_valid_combinations_dataframe()
    print("\n符合条件的组合已经存储在 DataFrame 'valid_df' 中。")
    print("DataFrame 的前5行示例:")
    print(valid_df.head())
    # 可以选择将 DataFrame 保存到 CSV 文件
    # valid_df.to_csv("valid_combinations.csv", index=False, encoding="utf-8-sig")
    # print("\n有效组合已保存到文件 valid_combinations.csv")
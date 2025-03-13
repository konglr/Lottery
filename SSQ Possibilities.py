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
    连号也被视为公差为1的等差数列。
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

def has_two_arithmetic_progressions_of_length_three(nums):
    """
    检查列表中是否包含两个或更多套，每套由3个数字组成的等差数列。
    例如: [1, 2, 3, 17, 18, 19] 包含两套3个数字的等差数列。
    """
    count_ap_3 = 0
    for combo in itertools.combinations(nums, 3):
        if len(combo) == 3:
            diff = combo[1] - combo[0]
            if combo[2] - combo[1] == diff and diff != 0: # 确保是等差数列，且公差不为0 (避免 [1,1,1] 这种情况)
                count_ap_3 += 1
    return count_ap_3 >= 2


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
    修改：连号判断已去除，等差数列判断已包含公差为1的连号。
    新增：排除包含两套3个数字组成的等差数列的组合。
    新增：根据最新一期开奖数据进行排除。
    """
    all_combinations = list(itertools.combinations(range(1, 34), 6)) # 生成所有红球组合并转换为列表
    initial_combinations_count = len(all_combinations)
    print(f"初始组合总数 (C(33, 6)): {initial_combinations_count}")

    valid_combinations = []
    valid_odd_even_ratios = [(2, 4), (3, 3), (4, 2)] # 奇偶比例保持不变
    valid_small_large_ratios = [(3, 3), (2, 4)] # 修改大小号比例 (冷热号比例) 为 3:3 和 2:4

    # 初始化计数器
    arithmetic_geometric_count = 0 # 等差等比数列计数器 (包含连号)
    invalid_odd_even_count = 0
    invalid_small_large_count = 0
    invalid_sum_range_count = 0 # 新增和值范围计数器
    two_ap_3_count = 0 # 新增 “两套3个数字等差数列” 计数器
    no_overlap_latest_count = 0 # 新增 “与最新一期开奖红球没有相同组合” 计数器
    two_or_more_overlap_latest_count = 0 # 新增 “与最新一期开奖红球有2个及以上相同组合” 计数器

    # 读取 Excel 文件获取开奖数据
    df_lottery = pd.read_excel('双色球开奖情况.xlsx', usecols=["期号", "开奖日期", "红球1", "红球2", "红球3", "红球4", "红球5", "红球6"])

    # 找到最新一期开奖数据，假设期号越大，日期越新。可以根据 '期号' 列排序后取最后一行
    latest_lottery_data = df_lottery.sort_values(by='期号', ascending=False).iloc[0]

    # 获取最新一期的红球号码
    latest_red_balls = [latest_lottery_data['红球{}'.format(i)] for i in range(1, 7)]
    print(f"最新一期开奖红球号码: {latest_red_balls}")


    # 使用 tqdm 添加进度条
    for combination in tqdm(all_combinations, desc="筛选组合"):
        is_valid = True # 假设组合有效，除非被排除条件否定

        # 检查与最新一期开奖红球的重复情况
        overlap_count = sum(1 for ball in combination if ball in latest_red_balls)

        if is_arithmetic_progression(combination) or is_geometric_progression(combination):
            arithmetic_geometric_count += 1
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
        if has_two_arithmetic_progressions_of_length_three(combination): # 新增 “两套3个数字等差数列” 检查
            two_ap_3_count += 1
            is_valid = False

        # 新增的排除规则：
        if overlap_count == 0: # 排除红球与新一期开奖红球中没有相同的组合
            no_overlap_latest_count += 1
            is_valid = False
        if overlap_count >=2: # 排除红球与最新一期开奖红球中有2个及以上相同的组合
            two_or_more_overlap_latest_count += 1
            is_valid = False


        if is_valid: # 只有当组合通过所有检查时才加入有效列表
            valid_combinations.append(list(combination))

    valid_combinations_count = len(valid_combinations)
    print(f"符合所有条件的组合数量: {valid_combinations_count}")
    print(f"总共减少了 {initial_combinations_count - valid_combinations_count} 种组合")

    print("\n各种条件排除的组合数量统计:")
    print(f"  - 包含4个或以上数字的等差或等比数列 (包含连号): {arithmetic_geometric_count} 种") # 修改了描述，包含连号
    print(f"  - 奇偶比例不符合 2:4, 3:3, 4:2: {invalid_odd_even_count} 种")
    print(f"  - 大小号比例 (冷热号比例) 不符合 3:3, 2:4: {invalid_small_large_count} 种") # 修改了描述
    print(f"  - 红球和值小于30或大于170: {invalid_sum_range_count} 种") # 新增和值范围排除统计
    print(f"  - 包含两套3个数字组成的等差数列: {two_ap_3_count} 种") # 新增统计
    print(f"  - 红球与最新一期开奖红球没有相同组合: {no_overlap_latest_count} 种") # 新增统计
    print(f"  - 红球与最新一期开奖红球有2个及以上相同组合: {two_or_more_overlap_latest_count} 种") # 新增统计


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
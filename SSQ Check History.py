import pandas as pd

# 预测号码
predictions = [
    # 第1组：包含连号 11, 12，热号主导
    ([2, 4, 11, 12, 18, 25], 15),
    # 选择原因：冷热号比例4:2(热号：04, 11, 12, 18, 25；冷号：02)，奇偶比例3:3，包含连号11-12，无同尾号，无3个以上等差数列，无3个以上连号。热号占主导，包含一对连号，奇偶平衡。
    # 第2组：包含同尾号 10, 30，冷热均衡
    ([5, 10, 16, 23, 28, 30], 14),
    # 选择原因：冷热号比例3:3(热号：16, 28, 30；冷号：05, 10, 23)，奇偶比例2:4(偶数占优)，无连号，同尾号10, 30(尾号0)，无3个以上等差数列，无3个以上连号。冷热均衡，偶数占主导，有一对同尾号。
    # 第3组：包含连号 07, 08，热号偏多
    ([7, 8, 11, 18, 25, 29], 11),
    # 选择原因：冷热号比例4:2(热号：11, 18, 25, 29；冷号：07, 08)，奇偶比例4:2(奇数占优)，包含连号07-08，同尾号08, 18(尾号8)，无3个以上等差数列，无3个以上连号。热号占主导，包含一对连号，奇数略占主导，有一对同尾号。
    # 第4组：包含连号 02, 03，冷号稍多
    ([2, 3, 12, 16, 21, 26], 7),
    # 选择原因：冷热号比例2:4(热号：12, 16；冷号：02, 03, 21, 26)，奇偶比例2:4(偶数占优)，包含连号02-03，同尾号02, 12(尾号2)，无3个以上等差数列，无3个以上连号。冷号略占主导，包含一对连号，偶数占主导，有一对同尾号。
    # 第5组：包含连号 04, 05，热号主导
    ([4, 5, 11, 18, 23, 28], 10),
    # 选择原因：冷热号比例4:2(热号：04, 11, 18, 28；冷号：05, 23)，奇偶比例3:3，包含连号04-05，无同尾号，无3个以上等差数列，无3个以上连号。热号占主导，包含一对连号，奇偶平衡。
    # 第6组：包含同尾号 10, 30，冷热均衡
    ([8, 10, 16, 25, 29, 30], 5),
    # 选择原因：冷热号比例3:3(热号：16, 25, 29, 30；冷号：08, 10)，奇偶比例2:4(偶数占优)，无连号，同尾号10, 30(尾号0)，无3个以上等差数列，无3个以上连号。冷热均衡，偶数占主导，有一对同尾号。
    # 第7组：包含同尾号 02, 12，冷号稍多
    ([2, 7, 12, 18, 21, 26], 15),
    # 选择原因：冷热号比例2:4(热号：12, 18；冷号：02, 07, 21, 26)，奇偶比例2:4(偶数占优)，无连号，同尾号02, 12(尾号2)，无3个以上等差数列，无3个以上连号。冷号略占主导，偶数占主导，有一对同尾号。
    # 第8组：包含同尾号 03, 33，冷热均衡
    ([3, 11, 16, 23, 28, 33], 14),
    # 选择原因：冷热号比例3:3(热号：11, 16, 28；冷号：03, 23, 33)，奇偶比例3:3，无连号，同尾号03, 33(尾号3)，无3个以上等差数列，无3个以上连号。冷热均衡，奇偶平衡，有一对同尾号。
    # 第9组：包含同尾号 08, 18，冷热均衡
    ([4, 8, 11, 18, 21, 26], 11),
    # 选择原因：冷热号比例3:3(热号：04, 11, 18；冷号：08, 21, 26)，奇偶比例2:4(偶数占优)，无连号，同尾号08, 18(尾号8)，无3个以上等差数列，无3个以上连号。冷热均衡，偶数占主导，有一对同尾号。
    # 第10组：包含同尾号 10, 30，冷热均衡
    ([5, 10, 16, 23, 28, 30], 7)
    # 选择原因：冷热号比例3:3(热号：16, 28, 30；冷号：05, 10, 23)，奇偶比例2:4(偶数占优)，无连号，同尾号10, 30(尾号0)，无3个以上等差数列，无3个以上连号。冷热均衡，偶数占主导，有一对同尾号。
]

# 读取 Excel 文件
df = pd.read_excel('双色球开奖情况.xlsx')

# 将红球号码和蓝球号码转换为列表
df['红球号码'] = df['前区号码'].apply(lambda x: sorted([int(i) for i in x.split(' ')]))
df['蓝球号码'] = df['后区号码'].apply(lambda x: int(x))

def check_prediction(prediction, df):
    red_balls, blue_ball = prediction
    red_balls = sorted(red_balls)  # 对预测的红球号码进行排序

    # 1. 检查是否完全匹配
    match = df[df.apply(lambda row: row['红球号码'] == red_balls and row['蓝球号码'] == blue_ball, axis=1)]
    if not match.empty:
        return match, "完全匹配"

    # 2. 检查 6 个红球相同
    six_red_match = df[df.apply(lambda row: set(row['红球号码']) == set(red_balls), axis=1)]
    if not six_red_match.empty:
        return six_red_match, "6 个红球相同"

    # 3. 检查 5 个红球 + 1 个蓝球相同
    five_red_one_blue_match = []
    for _, row in df.iterrows():
        red_matches = len(set(red_balls) & set(row['红球号码']))
        if red_matches == 5 and row['蓝球号码'] == blue_ball:
            five_red_one_blue_match.append(row)
    five_red_one_blue_match_df = pd.DataFrame(five_red_one_blue_match)
    if not five_red_one_blue_match_df.empty:
        return five_red_one_blue_match_df, "5 个红球 + 1 个蓝球相同"

    return pd.DataFrame(), "无匹配"

for prediction in predictions:
    red_balls, blue_ball = prediction
    print(f"预测号码：{red_balls} | {blue_ball}")

    match, match_type = check_prediction(prediction, df)

    if not match.empty:
        print(f"{match_type}的历史记录：")
        print(match.iloc[:, :5])
    else:
        print("没有找到匹配的历史记录。")
    print("-" * 20)

def analyze_red_ball_sum(df):
    """
    分析双色球历史开奖记录中红球号码的和，并返回包含 '红球和' 列的 DataFrame。

    Args:
        df (pd.DataFrame): 包含双色球开奖数据的 DataFrame，
                             需要包含 '红球号码' 列，该列为列表类型，存储每期开奖的 6 个红球号码。

    Returns:
        pd.DataFrame:  返回修改后的 DataFrame，新增了 '红球和' 列。
                       函数同时打印红球和的最大值、最小值和分布情况（频率统计）。
    """

    # 1. 计算每期红球号码之和，并添加到新的 '红球和' 列
    df['红球和'] = df['红球号码'].apply(sum)

    # 2. 找出红球和的最大值和最小值
    max_sum = df['红球和'].max()
    min_sum = df['红球和'].min()

    print(f"红球号码之和的最大值: {max_sum}")
    print(f"红球号码之和的最小值: {min_sum}")
    print("\n红球号码之和的分布情况:")

    # 3. 分析红球和的分布情况 (使用 value_counts 统计频率)
    sum_distribution = df['红球和'].value_counts().sort_index() # 统计每个和值出现的次数并按和值排序
    print(sum_distribution) # 打印和值的分布情况

    return df # 修改：函数返回 DataFrame


def print_top_bottom_sum_results(df, top_n=10, bottom_n=10):
    """
    打印红球和值最大和最小的指定期数开奖信息。

    Args:
        df (pd.DataFrame): 包含双色球开奖数据的 DataFrame，需要包含 '红球和', '开奖日期', '期号', '前区号码' 列。
        top_n (int):  打印和值最大的前 N 期，默认为 10。
        bottom_n (int): 打印和值最小的前 N 期，默认为 10。
    """

    # 确保 '红球和' 列已存在
    if '红球和' not in df.columns:
        raise ValueError("DataFrame 必须包含 '红球和' 列。请先调用 analyze_red_ball_sum 函数。")

    # 按 '红球和' 列排序，分别获取最大和最小的 N 期
    df_sorted_asc = df.sort_values(by='红球和', ascending=True) # 升序排序，和值最小在前
    df_sorted_desc = df.sort_values(by='红球和', ascending=False) # 降序排序，和值最大在前

    top_sum_results = df_sorted_desc.head(top_n) # 获取和值最大的前 N 期
    bottom_sum_results = df_sorted_asc.head(bottom_n) # 获取和值最小的前 N 期

    print(f"\n--- 红球和值 最大的 {top_n} 期开奖信息 ---")
    for index, row in top_sum_results.iterrows():
        print(f"  日期: {row['开奖日期']}, 期号: {row['期号']}, 开奖号码: {row['前区号码']} (和值: {row['红球和']})")

    print(f"\n--- 红球和值 最小的 {bottom_n} 期开奖信息 ---")
    for index, row in bottom_sum_results.iterrows():
        print(f"  日期: {row['开奖日期']}, 期号: {row['期号']}, 开奖号码: {row['前区号码']} (和值: {row['红球和']})")


# 读取 Excel 文件
df = pd.read_excel('双色球开奖情况.xlsx')

# 将红球号码和蓝球号码转换为列表 (如果您的 DataFrame 还没有 '红球号码' 列，则运行此代码)
df['红球号码'] = df['前区号码'].apply(lambda x: sorted([int(i) for i in x.split(' ')]))

# 调用 analyze_red_ball_sum 函数进行分析，并获取包含 '红球和' 列的 DataFrame
df_analyzed = analyze_red_ball_sum(df)

# 调用 print_top_bottom_sum_results 函数，打印和值最大和最小的各10期开奖信息
print_top_bottom_sum_results(df_analyzed, top_n=10, bottom_n=10)


#连号判断
def analyze_consecutive_numbers_in_lottery(excel_file):
    """
    分析彩票开奖记录，查找出现3连号、4连号和5连号的次数，并返回相关记录。

    参数:
    excel_file (str): 包含彩票开奖记录的 Excel 文件路径。

    返回:
    tuple: 包含以下元素的元组：
           - int: 3连号出现的次数
           - int: 4连号出现的次数
           - int: 5连号出现的次数  (新增)
           - pandas.DataFrame: 包含出现3连号的开奖记录
           - pandas.DataFrame: 包含出现4连号的开奖记录
           - pandas.DataFrame: 包含出现5连号的开奖记录  (新增)
    """

    # 确保期号是字符串类型，并移除逗号
    df['期号'] = df['期号'].astype(str).str.replace(',', '')

    # 将前区号码转换为数字列表
    df['前区号码'] = df['前区号码'].apply(lambda x: [int(num) for num in str(x).split()])

    # 定义检测连号的函数 (函数保持不变)
    def check_consecutive_numbers(numbers, consecutive_length):
        """检测号码列表中是否包含指定长度的连号"""
        numbers.sort()  # 确保号码排序
        count = 1
        for i in range(len(numbers) - 1):
            if numbers[i+1] == numbers[i] + 1:
                count += 1
                if count == consecutive_length:
                    return True
            else:
                count = 1
        return False

    # 查找3连号 (代码保持不变)
    df['包含3连号'] = df['前区号码'].apply(lambda nums: check_consecutive_numbers(nums, 3))
    df_3_consecutive = df[df['包含3连号']]
    count_3_consecutive = len(df_3_consecutive)

    # 查找4连号 (代码保持不变)
    df['包含4连号'] = df['前区号码'].apply(lambda nums: check_consecutive_numbers(nums, 4))
    df_4_consecutive = df[df['包含4连号']]
    count_4_consecutive = len(df_4_consecutive)

    # ----- 新增：查找 5 连号 -----
    df['包含5连号'] = df['前区号码'].apply(lambda nums: check_consecutive_numbers(nums, 5))
    df_5_consecutive = df[df['包含5连号']]
    count_5_consecutive = len(df_5_consecutive)
    # ----- 新增结束 -----

    # 返回结果和相关开奖记录 (返回值修改)
    return count_3_consecutive, count_4_consecutive, count_5_consecutive, df_3_consecutive, df_4_consecutive, df_5_consecutive

# 设置 Excel 文件路径 (代码保持不变)
excel_file_path = '双色球开奖情况.xlsx'

# 分析连号情况并获取结果 (代码修改，增加 count_5 和 records_5)
count_3, count_4, count_5, records_3_consecutive, records_4_consecutive, records_5_consecutive = analyze_consecutive_numbers_in_lottery(excel_file_path)

print(f"在开奖记录中：")
print(f"  - 出现 3 连号的次数: {count_3} 次")
print(f"  - 出现 4 连号的次数: {count_4} 次")
print(f"  - 出现 5 连号的次数: {count_5} 次") # 新增 5 连号的次数
print("\n----- 出现 3 连号的开奖记录 -----")
if not records_3_consecutive.empty:
    print(records_3_consecutive[['期号', '开奖日期', '前区号码', '后区号码']].to_string(index=False))
else:
    print("没有找到包含 3 连号的开奖记录。")

print("\n----- 出现 4 连号的开奖记录 -----")
if not records_4_consecutive.empty:
    print(records_4_consecutive[['期号', '开奖日期', '前区号码', '后区号码']].to_string(index=False))
else:
    print("没有找到包含 4 连号的开奖记录。")

print("\n----- 出现 5 连号的开奖记录 -----") # 新增 5 连号的开奖记录输出
if not records_5_consecutive.empty:
    print(records_5_consecutive[['期号', '开奖日期', '前区号码', '后区号码']].to_string(index=False))
else:
    print("没有找到包含 5 连号的开奖记录。")


def process_lottery_data(df, lottery_type):
    """根据彩票类型处理数据，将号码列转换为列表"""
    if lottery_type == "双色球":
        df['前区号码'] = df['前区号码'].apply(lambda x: [int(num) for num in str(x).split()])
        df['后区号码'] = df['后区号码'].apply(lambda x: [int(x)])  # 后区现在是单个数字，需要转换为列表
    elif lottery_type == "快乐8":
        df['前区号码'] = df['前区号码'].apply(lambda x: [int(num) for num in str(x).split()])
    return df

def find_arithmetic_progressions_of_length_three(nums):
    """
    查找列表中所有由3个数字组成的等差数列。
    返回一个列表，其中包含所有找到的等差数列组合。
    """
    ap_sets = []
    for combo in itertools.combinations(nums, 3):
        if len(combo) == 3:
            diff = combo[1] - combo[0]
            if combo[2] - combo[1] == diff and diff != 0:  # 确保是等差数列，且公差不为0
                ap_sets.append(list(combo)) # 将找到的等差数列组合添加到列表
    return ap_sets


def find_ssq_records_with_exactly_two_ap3(filename="双色球开奖情况.xlsx"):
    """
    分析双色球历史开奖数据，找出前区号码包含“正好两套3个数字等差数列”的中奖记录，
    并列出每组号码中的这两套等差数列。
    返回包含符合条件的记录的 DataFrame。
    """
    try:
        df = pd.read_excel(filename)
    except FileNotFoundError:
        return f"找不到文件: {filename}，请确保文件与脚本在同一目录下", None

    df['期号'] = df['期号'].astype(str).str.replace(',', '')  # 清理期号列
    df = process_lottery_data(df, "双色球")  # 指定彩票类型为双色球

    matching_records = []  # 存储符合条件的记录

    for index, row in df.iterrows():
        front_nums = row['前区号码']
        ap_sets = find_arithmetic_progressions_of_length_three(front_nums) # 查找当前号码中的所有3个数字等差数列

        if len(ap_sets) == 2: # 检查是否正好有两套
            record = row.to_dict() # 将 Series 转换为字典，方便添加新的键值对
            record['等差数列组合1'] = str(ap_sets[0]) # 存储第一套等差数列
            record['等差数列组合2'] = str(ap_sets[1]) # 存储第二套等差数列
            matching_records.append(record)  # 如果符合条件，将记录添加到列表

    if matching_records:
        result_df = pd.DataFrame(matching_records)  # 将列表转换为 DataFrame
        return "以下是前区号码包含 '正好两套3个数字等差数列' 的双色球中奖记录，并列出每组号码中的这两套等差数列（仅显示 '开奖日期', '期号', '前区号码', '等差数列组合1', '等差数列组合2' 列）：", result_df
    else:
        return "在历史开奖数据中，没有找到前区号码包含 '正好两套3个数字等差数列' 的双色球中奖记录。", pd.DataFrame()  # 返回空 DataFrame



print("开始分析双色球历史开奖数据，查找前区号码包含 '正好两套3个数字等差数列' 的中奖记录，并列出等差数列组合...\n")
analysis_text, result_df = find_ssq_records_with_exactly_two_ap3() # 调用新的分析函数
print(analysis_text)

    if not result_df.empty:  # 如果 result_df 不为空，则显示 DataFrame
        # 修改：只打印指定的列, 包括新的等差数列组合列
        columns_to_display = ['开奖日期', '期号', '前区号码', '等差数列组合1', '等差数列组合2']
        print("\n符合条件的开奖记录（仅显示 '开奖日期', '期号', '前区号码', '等差数列组合1', '等差数列组合2' 列）：")
        print(result_df[columns_to_display].to_string(index=False))  # 在命令行终端打印指定列, 使用 to_string 避免省略
    else:
        print("\n没有找到符合条件的开奖记录。")

    print("\n分析完成。")
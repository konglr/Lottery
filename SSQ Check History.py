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
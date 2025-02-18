import pandas as pd

# 预测号码
predictions = [
    ([1, 4, 12, 23, 29, 33], 15),
    ([2, 6, 12, 16, 25, 30], 12),
    ([3, 9, 18, 20, 27, 31], 16),
    ([1, 7, 12, 24, 28, 30], 3),
    ([4, 10, 15, 22, 26, 33], 12),
    ([6, 8, 11, 18, 22, 27], 10),
    ([3, 9, 12, 23, 25, 30], 15),
    ([5, 10, 16, 22, 28, 30], 16),
    ([1, 7, 12, 24, 27, 31], 12),
    ([4, 10, 12, 22, 26, 30], 3),
]

# 读取 Excel 文件
df = pd.read_excel('双色球开奖情况.xls')

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


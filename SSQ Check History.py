import pandas as pd

# 预测号码

predictions = [
    # 第1组：同尾16-26（高频组合）+ 蓝球12（热号）
    ([4, 11, 13, 16, 26, 30], 12),
    # 第2组：尾3组合13-23 + 蓝球15（热号）
    ([3, 9, 13, 18, 23, 31], 15),
    # 第3组：尾6补充16-26 + 蓝球16（大号）
    ([6, 11, 16, 22, 26, 33], 16),
    # 第4组：连号11-13 + 蓝球09（温号）
    ([4, 11, 13, 19, 24, 28], 9),
    # 第5组：冷尾5补充05-25 + 蓝球03（小号）
    ([5, 10, 15, 20, 25, 30], 3),
    # 第6组：二区主导12-16 + 蓝球08（温号）
    ([2, 12, 14, 16, 20, 27], 8),
    # 第7组：奇偶平衡3:3 + 蓝球14（大号）
    ([3, 8, 14, 19, 23, 28], 14),
    # 第8组：尾1组合11-31 + 蓝球11（热号）
    ([7, 11, 17, 22, 26, 31], 11),
    # 第9组：三区强化23-33 + 蓝球06（温号）
    ([4, 12, 16, 23, 29, 33], 6),
    # 第10组：全区间覆盖 + 蓝球10（冷号补充）
    ([1, 7, 12, 20, 25, 30], 10)
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


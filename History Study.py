import pandas as pd

# 读取 Excel 文件
df = pd.read_excel('双色球开奖情况.xls')

# 将红球号码和蓝球号码转换为列表
df['红球号码'] = df['前区号码'].apply(lambda x: sorted([int(i) for i in x.split(' ')]))
df['蓝球号码'] = df['后区号码'].apply(lambda x: int(x))

def find_similar_records(df):
    """
    查找双色球历史开奖记录中，两次开奖相同的记录，或者相似的记录。

    Args:
        df: 包含历史开奖记录的 DataFrame。

    Returns:
        一个包含相同或相似开奖记录的 DataFrame，包含日期、期数、号码和匹配类型。
    """

    similar_records = []
    n_rows = len(df)
    match_counts = {'6+1 都相同': 0, '6+0 相同': 0, '5+1 相同': 0}  # 存储匹配类型计数

    for i in range(n_rows):
        row1 = df.iloc[i]
        red_balls1 = row1['红球号码']
        blue_ball1 = row1['蓝球号码']
        date1 = row1['开奖日期']
        issue1 = row1['期号']

        for j in range(i + 1, n_rows):  # 避免与自身比较，且只比较后面的记录
            row2 = df.iloc[j]
            red_balls2 = row2['红球号码']
            blue_ball2 = row2['蓝球号码']
            date2 = row2['开奖日期']
            issue2 = row2['期号']

            if red_balls1 == red_balls2 and blue_ball1 == blue_ball2:  # 6+1 都相同
                similar_records.append({
                    '日期1': date1, '期数1': issue1, '号码1': red_balls1 + [blue_ball1],
                    '日期2': date2, '期数2': issue2, '号码2': red_balls2 + [blue_ball2],
                    '匹配类型': '6+1 都相同'
                })
                match_counts['6+1 都相同'] += 1
            elif red_balls1 == red_balls2 and blue_ball1 != blue_ball2:  # 6+0 相同
                similar_records.append({
                    '日期1': date1, '期数1': issue1, '号码1': red_balls1 + [blue_ball1],
                    '日期2': date2, '期数2': issue2, '号码2': red_balls2 + [blue_ball2],
                    '匹配类型': '6+0 相同'
                })
                match_counts['6+0 相同'] += 1
            elif blue_ball1 == blue_ball2:  # 蓝球相同，继续比较红球
                red_diff = set(red_balls1) ^ set(red_balls2)  # 红球差异
                red_intersection = set(red_balls1) & set(red_balls2)  # 红球交集
                if len(red_diff) == 2 and len(red_intersection) == 5:  # 5+1 相同
                    similar_records.append({
                        '日期1': date1, '期数1': issue1, '号码1': red_balls1 + [blue_ball1],
                        '日期2': date2, '期数2': issue2, '号码2': red_balls2 + [blue_ball2],
                        '匹配类型': '5+1 相同'
                    })
                    match_counts['5+1 相同'] += 1

    return pd.DataFrame(similar_records), match_counts

similar_records, match_counts = find_similar_records(df)

if not similar_records.empty:
    print("找到相同或相似的历史记录：")
    for index, row in similar_records.iterrows():
        print(row)

    print("\n匹配类型统计：")
    for match_type, count in match_counts.items():
        print(f"{match_type}: {count} 次")
else:
    print("没有找到相同或相似的历史记录。")
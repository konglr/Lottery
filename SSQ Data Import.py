import json
import math
import requests
import openpyxl  # 替换 xlwt
import re
import logging
import pandas as pd
import os
import time
from tqdm import tqdm
from funcs.requestsdata import requests_data, get_latest_issue_from_system

Lottry_ID = 1  # 快乐8的ID为6; 7乐彩 ID为3；双色球 ID为1，
# 配置日志记录
logging.basicConfig(filename='my_log_file.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    df_existing = pd.read_excel('双色球开奖情况.xlsx') #替换xls为xlsx
    last_issue_in_excel = int(df_existing['期号'].max())
except FileNotFoundError:
    last_issue_in_excel = 0

latest_issue_in_system = get_latest_issue_from_system(Lottry_ID)
if latest_issue_in_system is None:
    print("无法获取最新期号，程序终止。")
    exit()

current_2025_times = latest_issue_in_system - 2025000
total_issueCount = 3299 + current_2025_times
# 89 + 122 + 153 + 154 + 153 + 154 + 154 + 153 + 153 + 154 + 154 + 152 + 154
# + 153 + 154 + 153 + 151 + 134 + 150 + 150 + 151 + 151 = 3299

# 如果本地文件最后一期与系统最新期号相同，则跳过下载
if last_issue_in_excel == latest_issue_in_system:
    print(f"本地数据已是最新，跳过下载。最新期号: {latest_issue_in_system}")
else:
    wb = openpyxl.Workbook() #替换xlwt为openpyxl
    sheet = wb.active  # 使用活动工作表
    sheet.title = '双色球'  # 设置工作表标题

    row = ["期号", "开奖日期", "WeekDay", "前区号码", "后区号码", "总销售额(元)", "奖池金额(元)",
           "一等奖注数", "一等奖奖金", "二等奖注数", "二等奖奖金", "三等奖注数", "三等奖奖金",
           "四等奖注数", "四等奖奖金", "五等奖注数", "五等奖奖金", "六等奖注数", "六等奖奖金"]
    for i, title in enumerate(row):
        sheet.cell(row=1, column=i + 1, value=title) #修改写入方式

    i = 2 #修改i的初始值
    range_max = math.floor(total_issueCount / 30 + 1) if total_issueCount % 30 == 0 else math.floor(total_issueCount / 30 + 2)
    for pageNum_i in tqdm(range(1, range_max), desc="下载进度"):  # 添加 tqdm 进度条
        tony_dict = requests_data(pageNum_i, total_issueCount, Lottry_ID
        for j in tony_dict:
            if j != '{':
                tony_dict = tony_dict[-(len(tony_dict) - 1):]
            else:
                break
        if tony_dict[len(tony_dict) - 1] == ')':
            tony_dict = tony_dict[:len(tony_dict) - 1]
        content = json.loads(tony_dict)
        content_data = content['data']
        for item in content_data:
            sheet.cell(row=i, column=1, value=item['issue'])
            sheet.cell(row=i, column=2, value=item['openTime'])
            sheet.cell(row=i, column=3, value=item['week'])
            sheet.cell(row=i, column=4, value=item['frontWinningNum'])
            sheet.cell(row=i, column=5, value=item['backWinningNum'])
            sheet.cell(row=i, column=6, value=item['saleMoney'])
            sheet.cell(row=i, column=7, value=item['prizePoolMoney'])
            winner_details = item.get('winnerDetails', [])
            for award in winner_details:
                award_etc = award.get('awardEtc', '')
                base_bet_winner = award.get('baseBetWinner', {})
                try:
                    award_level = int(award_etc)
                    if 1 <= award_level <= 6:
                        col_index = 8 + (award_level - 1) * 2
                        sheet.cell(row=i, column=col_index, value=base_bet_winner.get('awardNum', ''))
                        sheet.cell(row=i, column=col_index + 1, value=base_bet_winner.get('awardMoney', ''))
                except ValueError:
                    print(f"awardEtc: {award_etc} 不是有效的数字")
                    continue
            i += 1
    wb.save("双色球开奖情况.xlsx") #保存为xlsx格式
    print("数据已成功下载并保存。")

# 数据检验部分代码
# 读取 Excel 文件
df = pd.read_excel('双色球开奖情况.xlsx', header=0) #替换xls为xlsx

# 检查最早的数据是否为 2003001
earliest_issue = df['期号'].min()
if earliest_issue != 2003001:
    print(f"警告：最早的数据不是2003001，而是{earliest_issue}")
else:
    print(f"1.最早的数据是2003001，符合预期。")

# 检查最新一期的数据是否与系统里的相同
# Configure logging (as before)
logging.basicConfig(filename='my_log_file.log', level=logging.INFO)

last_issue_in_excel = df['期号'].max()  # Calculate AFTER reading and processing the Excel file
last_issue_in_excel = int(last_issue_in_excel)  # Convert to int for comparison

# --- Comparison ---
latest_issue_in_system = get_latest_issue_from_system(Lottry_ID)

if latest_issue_in_system is None:
    print("Failed to get latest issue from system.")
else:
    if last_issue_in_excel == int(latest_issue_in_system):  # Convert to int for comparison
        print(f"2.Excel与系统数据同步. 最新期数为: {last_issue_in_excel}")
        logging.info(f"Data synchronized: {last_issue_in_excel}")
    else:
        print(f"WARNING: Excel and system data are NOT synchronized!")
        print(f"Excel: {last_issue_in_excel}, System: {latest_issue_in_system}")
        logging.warning(f"Data mismatch: Excel: {last_issue_in_excel}, System: {latest_issue_in_system}")

# 检查是否有重复数据
duplicate_issues = df[df.duplicated(['期号'], keep=False)]
if not duplicate_issues.empty:
    print("3.警告：发现重复的期号:")
    print(duplicate_issues[['期号']])
else:
    print("3.未发现重复数据。")

# 统计每年的开奖次数
# 读取 Excel 文件
df = pd.read_excel('双色球开奖情况.xlsx', header=0) #替换xls为xlsx

# 将“开奖日期”列转换为 datetime 类型
df['开奖日期'] = pd.to_datetime(df['开奖日期'])

# 提取年份
df['年份'] = df['开奖日期'].dt.year

# 按年份分组并统计期数
yearly_counts = df.groupby('年份')['期号'].count()

# 打印结果
print(yearly_counts)


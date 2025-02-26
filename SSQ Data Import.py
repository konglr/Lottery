import json
import math
import requests
import xlwt
import re
import logging
import pandas as pd
import os
import time
from tqdm import tqdm

ID = 1 #快乐8的ID为6; 7乐彩 ID为3；双色球 ID为1，
# 配置日志记录
logging.basicConfig(filename='my_log_file.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def requests_data(index, issue_count,ID):
    headers = {
        'Connection': 'keep-alive',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.117 Safari/537.36',
        'Accept': '*/*',
        'Sec-Fetch-Site': 'same-site',
        'Sec-Fetch-Mode': 'no-cors',
        'Referer': 'https://www.zhcw.com/kjxx/kl8/',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'zh-CN,zh;q=0.9',
    }

    timestamp = int(time.time() * 1000)
    callback = f'jQuery1122035713028555611515_{timestamp}'
    tt = '0.123456789' # 需要分析网页请求得到正确的 tt 值
    _ = timestamp + 10 # 需要分析网页请求得到正确的 _ 值

    params = (
        ('callback', callback),
        ('transactionType', '10001001'),
        ('lotteryId', ID),
        ('issueCount', issue_count),
        ('startIssue', ''),
        ('endIssue', ''),
        ('startDate', ''),
        ('endDate', ''),
        ('type', '0'),
        ('pageNum', index),
        ('pageSize', '30'),
        ('tt', tt),
        ('_', _),
    )
    try:
        response = requests.get('https://jc.zhcw.com/port/client_json.php', headers=headers, params=params).content.decode('utf-8')
        return response
    except requests.exceptions.RequestException as e:
        logging.error(f"网络请求错误: {e}")
        return None

def get_latest_issue_from_system():
    try:
        response = requests_data(1, 1, ID)
        if response is None:
            return None

        match = re.search(r"\((.*)\)", response)
        if match:
            response = match.group(1)

        content = json.loads(response)
        latest_issue = int(content['data'][0]['issue'])
        return latest_issue
    except json.JSONDecodeError as e:
        logging.error(f"JSON解析错误: {e}, 数据: {response if 'response' in locals() else 'N/A'}")
        return None
    except (KeyError, IndexError) as e:
        logging.error(f"JSON数据访问错误: {e}, 数据: {response if 'response' in locals() else 'N/A'}")
        return None
    except Exception as e:
        logging.error(f"获取系统最新期号出错: {e}")
        return None

def write_to_excel(sheet, row_index, item):
    sheet.write(row_index, 0, item['issue'])
    sheet.write(row_index, 1, item['openTime'])
    sheet.write(row_index, 2, item['week'])
    sheet.write(row_index, 3, item['frontWinningNum'])
    sheet.write(row_index, 4, item['backWinningNum'])
    sheet.write(row_index, 5, item['saleMoney'])
    sheet.write(row_index, 6, item['prizePoolMoney'])
    winner_details = item.get('winnerDetails', [])
    for award in winner_details:
        award_etc = award.get('awardEtc', '')
        base_bet_winner = award.get('baseBetWinner', {})
        try:
            award_level = int(award_etc)
            if 1 <= award_level <= 6:
                col_index = 7 + (award_level - 1) * 2
                sheet.write(row_index, col_index, base_bet_winner.get('awardNum', ''))
                sheet.write(row_index, col_index + 1, base_bet_winner.get('awardMoney', ''))
        except ValueError as e:
            logging.error(f"award_etc 类型错误: {e}, 数据: {award_etc}")
            continue
        except (KeyError, TypeError) as e:
            logging.error(f"中奖详情数据访问错误: {e}, 数据: {award}")
            continue

latest_issue = get_latest_issue_from_system()
if latest_issue is None:
    print("无法获取最新期号，程序终止。")
    exit()


current_2025_times = latest_issue - 2025000
total_issueCount = 3247 + current_2025_times

wb = xlwt.Workbook()
sheet = wb.add_sheet('双色球')

row = ["期号", "开奖日期", "WeekDay", "前区号码", "后区号码", "总销售额(元)", "奖池金额(元)",
       "一等奖注数", "一等奖奖金", "二等奖注数", "二等奖奖金", "三等奖注数", "三等奖奖金",
       "四等奖注数", "四等奖奖金", "五等奖注数", "五等奖奖金", "六等奖注数", "六等奖奖金"]
for i in range(len(row)):
    sheet.write(0, i, row[i])

i = 1
range_max = math.ceil(total_issueCount / 30)

for pageNum_i in tqdm(range(1, range_max + 1), desc="爬取进度"):
    tony_dict = requests_data(pageNum_i, total_issueCount,ID)
    if tony_dict is None:
        print(f"第 {pageNum_i} 页数据获取失败，跳过。")
        continue

    tony_dict = re.search(r'\((.*)\)', tony_dict).group(1)

    try:
        content = json.loads(tony_dict)
        content_data = content['data']
    except json.JSONDecodeError as e:
        logging.error(f"JSON解析错误: {e}, 数据: {tony_dict if 'tony_dict' in locals() else 'N/A'}")
        continue

    for item in content_data:
        write_to_excel(sheet, i, item)
        i += 1

wb.save("双色球开奖情况.xls")
print("数据爬取完成，已保存到 双色球开奖情况.xls")


# 数据检验部分代码
# 读取 Excel 文件
df = pd.read_excel('双色球开奖情况.xls')

# 检查最早的数据是否为 2003001
earliest_issue = df['期号'].min()
if earliest_issue != 2003001:
    print(f"警告：最早的数据不是2003001，而是{earliest_issue}")
else:
    print(f"1.最早的数据是2003001，符合预期。")

#检查最新一期的数据是否与系统里的相同
import re
import logging

# Configure logging (as before)
logging.basicConfig(filename='my_log_file.log', level=logging.INFO)

last_issue_in_excel = df['期号'].max()  # Calculate AFTER reading and processing the Excel file
last_issue_in_excel = int(last_issue_in_excel) #Convert to int for comparison


# --- Comparison ---
latest_issue_in_system = get_latest_issue_from_system()

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


#统计每年的开奖次数
# 读取 Excel 文件
df = pd.read_excel('双色球开奖情况.xls')

# 将“开奖日期”列转换为 datetime 类型
df['开奖日期'] = pd.to_datetime(df['开奖日期'])

# 提取年份
df['年份'] = df['开奖日期'].dt.year

# 按年份分组并统计期数
yearly_counts = df.groupby('年份')['期号'].count()

# 打印结果
print(yearly_counts)


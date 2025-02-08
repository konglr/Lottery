import json
import math as math
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import pandas as pd
import requests
import xlwt
# import xlrdtimizers, utils, datasets
#from utils import display


def requests_data(index):
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

    params = (
        ('callback', 'jQuery1122035713028555611515_1607745050216'),
        ('transactionType', '10001001'),
        ('lotteryId', '1'),#快乐8的ID为6; 7乐彩 ID为3；双色球 ID为1，
        ('issueCount', issueCount),
        ('startIssue', ''),
        ('endIssue', ''),
        ('startDate', ''),
        ('endDate', ''),
        ('type', '0'),
        ('pageNum', index),
        ('pageSize', '30'),
        ('tt', '0.24352317020584802'),
        ('_', '1607745050225'),
    )
    # 获取服务器返回数据
    response = requests.get('https://jc.zhcw.com/port/client_json.php', headers=headers, params=params).content.decode('utf-8')
    #print(response)
    return response

#更新最新的开奖期数，并更新开奖总数 issueCount

def get_latest_issue_from_system():
    # ... (The improved get_latest_issue_from_system() function from the previous response) ...
    pass #This is just a placeholder, you must paste the get_latest_issue_from_system() function here

def get_latest_issue_from_system():
    try:
        response = requests_data(1)
        if response is None:
            return None

        match = re.search(r"\((.*)\)", response)
        if match:
            response = match.group(1)

        content = json.loads(response)
        latest_issue = int(content['data'][0]['issue'])  # 转换为整数
        return latest_issue
    except json.JSONDecodeError as e:
        logging.error(f"JSON解析错误: {e}, 数据: {response if 'response' in locals() else 'N/A'}")  # 打印response，如果存在
        return None
    except (KeyError, IndexError) as e:
        logging.error(f"JSON数据访问错误: {e}, 数据: {response if 'response' in locals() else 'N/A'}")
        return None
    except Exception as e:
        logging.error(f"获取系统最新期号出错: {e}")
        return None



current_2025_times=get_latest_issue_from_system()-2025000#  SSQ open times in 2024
#times_2003=89
#time_2004=122
#time_2005=153

issueCount = 3247+current_2025_times
# 2002年 -2024年 总共发行了3246期， 可以在运行本代码时根据实际日期修改本变量。


wb = xlwt.Workbook()
sheet= wb.add_sheet('双色球')
# 存储表头文件
row=["期号","开奖日期","WeekDay","前区号码","后区号码","总销售额(元)","奖池金额(元)",
     "一等奖注数","一等奖奖金","二等奖注数","二等奖奖金","三等奖注数","三等奖奖金",
     "四等奖注数","四等奖奖金","五等奖注数","五等奖奖金","六等奖注数","六等奖奖金"]

# 写入表头
for i in range(0,len(row)):
    sheet.write(0,i,row[i])

i=1
range_max = math.floor(issueCount/30+1) if issueCount%30==0 else math.floor(issueCount/30+2)
#如果issueCount是30的整数倍则range_max=math.floor(issueCount/30+1)，否则range_max=math.floor(issueCount/30+2)
for pageNum_i in range(1,range_max):#页数必须正好，多了就会返回重复数据，431/30=14.3
    tony_dict=requests_data(pageNum_i)
    for j in tony_dict:
        if j != '{':
            tony_dict=tony_dict[-(len(tony_dict)-1):]
        else :
            break
    if tony_dict[len(tony_dict)-1]==')':
        tony_dict=tony_dict[:len(tony_dict)-1]#删除最后一个右括号)
    content = json.loads(tony_dict)
    content_data=content['data']


    for item in content_data:
        sheet.write(i, 0, item['issue'])
        sheet.write(i, 1, item['openTime'])
        sheet.write(i, 2, item['week'])
        sheet.write(i, 3, item['frontWinningNum'])
        sheet.write(i, 4, item['backWinningNum'])
        sheet.write(i, 5, item['saleMoney'])
        sheet.write(i, 6, item['prizePoolMoney'])

        winner_details=item.get('winnerDetails',[])
        winner_details = item.get('winnerDetails', [])

        # 写入中奖详情
        for award in winner_details:
            award_etc = award.get('awardEtc', '')  # 处理 awardEtc 缺失的情况
            base_bet_winner = award.get('baseBetWinner', {})  # 处理 baseBetWinner 缺失的情况

            # 根据 award_etc 写入不同等级的奖项信息
            try:  # 防止 award_etc 不是数字类型
                award_level = int(award_etc)  # 将 award_etc 转换为整数，表示奖项等级
                if 1 <= award_level <= 6:  # 只处理一到六等奖
                    col_index = 7 + (award_level - 1) * 2  # 计算列索引
                    sheet.write(i, col_index, base_bet_winner.get('awardNum', ''))  # 写入注数
                    sheet.write(i, col_index + 1, base_bet_winner.get('awardMoney', ''))  # 写入奖金
            except ValueError:
                print(f"awardEtc: {award_etc} 不是有效的数字")
                continue

        i += 1
# 保存
wb.save("双色球开奖情况.xls")



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


# --- System Data Retrieval ---


# --- Comparison ---
latest_issue_in_system = get_latest_issue_from_system()

if latest_issue_in_system is None:
    print("Failed to get latest issue from system.")
else:
    if last_issue_in_excel == int(latest_issue_in_system):  # Convert to int for comparison
        print(f"2.Excel and system data are synchronized. Latest issue: {last_issue_in_excel}")
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

# 读取 Excel 文件
df = pd.read_excel('双色球开奖情况.xls')

# 将“后区号码”列转换为字符串类型
df['后区号码'] = df['后区号码'].astype(str)

# 提取后区号码数据
back_numbers = df['后区号码'].str.split(' ', expand=True).stack()

# 统计数字出现频率
number_counts = back_numbers.value_counts()

# 打印数字出现频率
print(number_counts)



#统计每年的开奖次数
# 将“开奖日期”列转换为 datetime 类型
df['开奖日期'] = pd.to_datetime(df['开奖日期'])

# 提取年份
df['年份'] = df['开奖日期'].dt.year

# 按年份分组并统计期数
yearly_counts = df.groupby('年份')['期号'].count()

# 打印结果
print(yearly_counts)

# 可视化（可选）
import matplotlib.pyplot as plt

yearly_counts.plot(kind='bar')
plt.xlabel('年份')
plt.ylabel('开奖期数')
plt.title('每年双色球开奖期数')
plt.show()



# 获取前 10 行数据的 "前区号码" 和 "后区号码" 列，并按索引倒序排列
last_rows_reversed = df.head(50)[["前区号码", "后区号码"]].sort_index(ascending=False)

# 打印结果
print(last_rows_reversed)
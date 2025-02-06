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

current_2025_times=11# Lukcy 8 open times in 2024
#times_2003=89
#time_2004=122
#time_2005=153

issueCount = 3246+current_2025_times
# 2002年 -2024年 总共发行了3246期， 可以在运行本代码时根据实际日期修改本变量。
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

# 将“后区号码”列转换为字符串类型
df['后区号码'] = df['后区号码'].astype(str)

# 提取后区号码数据
back_numbers = df['后区号码'].str.split(' ', expand=True).stack()

# 统计数字出现频率
number_counts = back_numbers.value_counts()

# 打印数字出现频率
print(number_counts)

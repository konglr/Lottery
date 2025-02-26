import json
import math as math
import numpy as np
import pandas as pd
import requests
import xlwt
import xlrdtimizers, utils, datasets
#from utils import display

latest_issue = get_latest_issue_from_system()
if latest_issue is None:
    print("无法获取最新期号，程序终止。")
    exit()

KL8_current_2025_times = latest_issue - 2025000 # Lukcy 8 open times in 2024
KL8_total_issueCount = 65+351+351+351+351+KL8_current_2025_times
#快乐8在20201028发行第一期，在当年发行了65期，在2021年总共发行了351期， 2022 = 351 ,2023  - 351 截止20220115共发行了65+351+351+351+21=1139，大家可以在运行本代码时根据实际日期修改本变量。
def requests_data(index,issueCount):
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
        ('lotteryId', '6'),#快乐8的ID为6; 7乐彩 ID为3；双色球 ID为1，
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
sheet= wb.add_sheet('快乐8')
# 存储表头文件
row=["期号","开奖日期","前区号码","后区号码","总销售额(元)","奖池金额(元)","一等奖注数","一等奖奖金",
     "二等奖注数","二等奖奖金","三等奖注数","三等奖奖金","四等奖注数","四等奖奖金",
     "五等奖注数","五等奖奖金","六等奖注数","六等奖奖金"]
# 写入表头
for i in range(0,len(row)):
    sheet.write(0,i,row[i])

i=1
range_max = math.floor(KL8_total_issueCount/30+1) if KL8_total_issueCount%30==0 else math.floor(KL8_total_issueCount/30+2)
#如果issueCount是30的整数倍则range_max=math.floor(issueCount/30+1)，否则range_max=math.floor(issueCount/30+2)
for pageNum_i in range(1,range_max):#页数必须正好，多了就会返回重复数据，431/30=14.3
    tony_dict=requests_data(pageNum_i,KL8_total_issueCount)
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
        sheet.write(i, 2, item['frontWinningNum'])
        sheet.write(i, 3, item['backWinningNum'])
        sheet.write(i, 4, item['saleMoney'])
        sheet.write(i, 5, item['prizePoolMoney'])
        i=i+1
# 保存
wb.save("快乐8开奖情况.xls")



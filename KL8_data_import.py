import json
import math as math
import numpy as np
import pandas as pd
import requests
import xlwt
import re
import logging
import os
import time
from tqdm import tqdm
from funcs.requestsdata import requests_data,get_latest_issue_from_system

Lottry_ID = 6 #快乐8的ID为6; 7乐彩 ID为3；双色球 ID为1，

try:
    df_existing = pd.read_excel('快乐8开奖情况.xls')
    last_issue_in_excel = int(df_existing['期号'].max())
except FileNotFoundError:
    last_issue_in_excel = 0

latest_issue_in_system = get_latest_issue_from_system(Lottry_ID)
if latest_issue_in_system is None:
    print("无法获取最新期号，程序终止。")
    exit()

KL8_current_2025_times = latest_issue_in_system - 2025000 # Lukcy 8 open times in 2025
KL8_total_issueCount = 65+351+351+351+352+KL8_current_2025_times
#快乐8在20201028发行第一期，在当年发行了65期，在2021年总共发行了351期， 2022 = 351 ,2023  - 351 截止20220115共发行了65+351+351+351+21=1139，大家可以在运行本代码时根据实际日期修改本变量。

if last_issue_in_excel == latest_issue_in_system:
    print(f"本地数据已是最新，跳过下载。最新期号: {latest_issue_in_system}")
else:
        wb = xlwt.Workbook()
        sheet = wb.add_sheet('快乐8')
        # 存储表头文件
        row = ["期号", "开奖日期", "前区号码", "后区号码", "总销售额(元)", "奖池金额(元)",
               "选10中10注数", "选10中10奖金", "选10中9注数", "选10中9奖金", "选10中8注数", "选10中8奖金",
               "选10中7注数", "选10中7奖金", "选10中6注数", "选10中6奖金", "选10中5注数", "选10中5奖金",
               "选10中0注数", "选10中0奖金",
               "选9中9注数", "选9中9奖金", "选9中8注数", "选9中8奖金", "选9中7注数", "选9中7奖金",
               "选9中6注数", "选9中6奖金", "选9中5注数", "选9中5奖金", "选9中4注数", "选9中4奖金",
               "选9中0注数", "选9中0奖金",
               "选8中8注数", "选8中8奖金", "选8中7注数", "选8中7奖金", "选8中6注数", "选8中6奖金",
               "选8中5注数", "选8中5奖金", "选8中4注数", "选8中4奖金", "选8中0注数", "选8中0奖金",
               "选7中7注数", "选7中7奖金", "选7中6注数", "选7中6奖金", "选7中5注数", "选7中5奖金",
               "选7中4注数", "选7中4奖金", "选7中0注数", "选7中0奖金",
               "选6中6注数", "选6中6奖金", "选6中5注数", "选6中5奖金", "选6中4注数", "选6中4奖金",
               "选6中3注数", "选6中3奖金",
               "选5中5注数", "选5中5奖金", "选5中4注数", "选5中4奖金", "选5中3注数", "选5中3奖金",
               "选4中4注数", "选4中4奖金", "选4中3注数", "选4中3奖金", "选4中2注数", "选4中2奖金",
               "选3中3注数", "选3中3奖金", "选3中2注数", "选3中2奖金",
               "选2中2注数", "选2中2奖金",
               "选1中1注数", "选1中1奖金"
               ]

        # 写入表头
        for i in range(0, len(row)):
            sheet.write(0, i, row[i])

        i = 1
        range_max = math.floor(KL8_total_issueCount / 30 + 1) if KL8_total_issueCount % 30 == 0 else math.floor(
            KL8_total_issueCount / 30 + 2)
        # 如果issueCount是30的整数倍则range_max=math.floor(issueCount/30+1)，否则range_max=math.floor(issueCount/30+2)
        for pageNum_i in range(1, range_max):  # 页数必须正好，多了就会返回重复数据，431/30=14.3
            tony_dict = requests_data(pageNum_i, KL8_total_issueCount, Lottry_ID)
            for j in tony_dict:
                if j != '{':
                    tony_dict = tony_dict[-(len(tony_dict) - 1):]
                else:
                    break
            if tony_dict[len(tony_dict) - 1] == ')':
                tony_dict = tony_dict[:len(tony_dict) - 1]  # 删除最后一个右括号)
            content = json.loads(tony_dict)
            content_data = content['data']

            for item in content_data:
                sheet.write(i, 0, item['issue'])
                sheet.write(i, 1, item['openTime'])
                sheet.write(i, 2, item['frontWinningNum'])
                sheet.write(i, 3, item['backWinningNum'])
                sheet.write(i, 4, item['saleMoney'])
                sheet.write(i, 5, item['prizePoolMoney'])

                # 处理中奖详情
                winner_details = item['winnerDetails']
                winner_data = {}
                for detail in winner_details:
                    award_etc = detail['awardEtc']
                    base_winner = detail['baseBetWinner']
                    winner_data[award_etc] = {
                        'awardNum': base_winner['awardNum'],
                        'awardMoney': base_winner['awardMoney']
                    }

                # 写入中奖信息
                sheet.write(i, 6, winner_data.get('x10z10', {}).get('awardNum', ''))
                sheet.write(i, 7, winner_data.get('x10z10', {}).get('awardMoney', ''))
                sheet.write(i, 8, winner_data.get('x10z9', {}).get('awardNum', ''))
                sheet.write(i, 9, winner_data.get('x10z9', {}).get('awardMoney', ''))
                sheet.write(i, 10, winner_data.get('x10z8', {}).get('awardNum', ''))
                sheet.write(i, 11, winner_data.get('x10z8', {}).get('awardMoney', ''))
                sheet.write(i, 12, winner_data.get('x10z7', {}).get('awardNum', ''))
                sheet.write(i, 13, winner_data.get('x10z7', {}).get('awardMoney', ''))
                sheet.write(i, 14, winner_data.get('x10z6', {}).get('awardNum', ''))
                sheet.write(i, 15, winner_data.get('x10z6', {}).get('awardMoney', ''))
                sheet.write(i, 16, winner_data.get('x10z5', {}).get('awardNum', ''))
                sheet.write(i, 17, winner_data.get('x10z5', {}).get('awardMoney', ''))
                sheet.write(i, 18, winner_data.get('x10z0', {}).get('awardNum', ''))
                sheet.write(i, 19, winner_data.get('x10z0', {}).get('awardMoney', ''))
                sheet.write(i, 20, winner_data.get('x9z9', {}).get('awardNum', ''))
                sheet.write(i, 21, winner_data.get('x9z9', {}).get('awardMoney', ''))
                sheet.write(i, 22, winner_data.get('x9z8', {}).get('awardNum', ''))
                sheet.write(i, 23, winner_data.get('x9z8', {}).get('awardMoney', ''))
                sheet.write(i, 24, winner_data.get('x9z7', {}).get('awardNum', ''))
                sheet.write(i, 25, winner_data.get('x9z7', {}).get('awardMoney', ''))
                sheet.write(i, 26, winner_data.get('x9z6', {}).get('awardNum', ''))
                sheet.write(i, 27, winner_data.get('x9z6', {}).get('awardMoney', ''))
                sheet.write(i, 28, winner_data.get('x9z5', {}).get('awardNum', ''))
                sheet.write(i, 29, winner_data.get('x9z5', {}).get('awardMoney', ''))
                sheet.write(i, 30, winner_data.get('x9z4', {}).get('awardNum', ''))
                sheet.write(i, 31, winner_data.get('x9z4', {}).get('awardMoney', ''))
                sheet.write(i, 32, winner_data.get('x9z0', {}).get('awardNum', ''))
                sheet.write(i, 33, winner_data.get('x9z0', {}).get('awardMoney', ''))
                sheet.write(i, 34, winner_data.get('x8z8', {}).get('awardNum', ''))
                sheet.write(i, 35, winner_data.get('x8z8', {}).get('awardMoney', ''))
                sheet.write(i, 36, winner_data.get('x8z7', {}).get('awardNum', ''))
                sheet.write(i, 37, winner_data.get('x8z7', {}).get('awardMoney', ''))
                sheet.write(i, 38, winner_data.get('x8z6', {}).get('awardNum', ''))
                sheet.write(i, 39, winner_data.get('x8z6', {}).get('awardMoney', ''))
                sheet.write(i, 40, winner_data.get('x8z5', {}).get('awardNum', ''))
                sheet.write(i, 41, winner_data.get('x8z5', {}).get('awardMoney', ''))
                sheet.write(i, 42, winner_data.get('x8z4', {}).get('awardNum', ''))
                sheet.write(i, 43, winner_data.get('x8z4', {}).get('awardMoney', ''))
                sheet.write(i, 44, winner_data.get('x8z0', {}).get('awardNum', ''))
                sheet.write(i, 45, winner_data.get('x8z0', {}).get('awardMoney', ''))
                sheet.write(i, 46, winner_data.get('x7z7', {}).get('awardNum', ''))
                sheet.write(i, 47, winner_data.get('x7z7', {}).get('awardMoney', ''))
                sheet.write(i, 48, winner_data.get('x7z6', {}).get('awardNum', ''))
                sheet.write(i, 49, winner_data.get('x7z6', {}).get('awardMoney', ''))
                sheet.write(i, 50, winner_data.get('x7z5', {}).get('awardNum', ''))
                sheet.write(i, 51, winner_data.get('x7z5', {}).get('awardMoney', ''))
                sheet.write(i, 52, winner_data.get('x7z4', {}).get('awardNum', ''))
                sheet.write(i, 53, winner_data.get('x7z4', {}).get('awardMoney', ''))
                sheet.write(i, 54, winner_data.get('x7z0', {}).get('awardNum', ''))
                sheet.write(i, 55, winner_data.get('x7z0', {}).get('awardMoney', ''))
                sheet.write(i, 56, winner_data.get('x6z6', {}).get('awardNum', ''))
                sheet.write(i, 57, winner_data.get('x6z6', {}).get('awardMoney', ''))
                sheet.write(i, 58, winner_data.get('x6z5', {}).get('awardNum', ''))
                sheet.write(i, 59, winner_data.get('x6z5', {}).get('awardMoney', ''))
                sheet.write(i, 60, winner_data.get('x6z4', {}).get('awardNum', ''))
                sheet.write(i, 61, winner_data.get('x6z4', {}).get('awardMoney', ''))
                sheet.write(i, 62, winner_data.get('x6z3', {}).get('awardNum', ''))
                sheet.write(i, 63, winner_data.get('x6z3', {}).get('awardMoney', ''))
                sheet.write(i, 64, winner_data.get('x5z5', {}).get('awardNum', ''))
                sheet.write(i, 65, winner_data.get('x5z5', {}).get('awardMoney', ''))
                sheet.write(i, 66, winner_data.get('x5z4', {}).get('awardNum', ''))
                sheet.write(i, 67, winner_data.get('x5z4', {}).get('awardMoney', ''))
                sheet.write(i, 68, winner_data.get('x5z3', {}).get('awardNum', ''))
                sheet.write(i, 69, winner_data.get('x5z3', {}).get('awardMoney', ''))
                sheet.write(i, 70, winner_data.get('x4z4', {}).get('awardNum', ''))
                sheet.write(i, 71, winner_data.get('x4z4', {}).get('awardMoney', ''))
                sheet.write(i, 72, winner_data.get('x4z3', {}).get('awardNum', ''))
                sheet.write(i, 73, winner_data.get('x4z3', {}).get('awardMoney', ''))
                sheet.write(i, 74, winner_data.get('x4z2', {}).get('awardNum', ''))
                sheet.write(i, 75, winner_data.get('x4z2', {}).get('awardMoney', ''))
                sheet.write(i, 76, winner_data.get('x3z3', {}).get('awardNum', ''))
                sheet.write(i, 77, winner_data.get('x3z3', {}).get('awardMoney', ''))
                sheet.write(i, 78, winner_data.get('x3z2', {}).get('awardNum', ''))
                sheet.write(i, 79, winner_data.get('x3z2', {}).get('awardMoney', ''))
                sheet.write(i, 80, winner_data.get('x2z2', {}).get('awardNum', ''))
                sheet.write(i, 81, winner_data.get('x2z2', {}).get('awardMoney', ''))
                sheet.write(i, 82, winner_data.get('x1z1', {}).get('awardNum', ''))
                sheet.write(i, 83, winner_data.get('x1z1', {}).get('awardMoney', ''))

                i = i + 1

            # 保存
            wb.save("快乐8开奖情况.xls")



#统计每年的开奖次数
# 读取 Excel 文件
df = pd.read_excel('快乐8开奖情况.xls', header=0)

# 将“开奖日期”列转换为 datetime 类型
df['开奖日期'] = pd.to_datetime(df['开奖日期'])

# 提取年份
df['年份'] = df['开奖日期'].dt.year

# 按年份分组并统计期数
yearly_counts = df.groupby('年份')['期号'].count()

# 打印结果
print(yearly_counts)

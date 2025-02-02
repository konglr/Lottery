import json
import math as math
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import pandas as pd
import requests
import xlwt
import xlrd
#from tensorflow.keras import layers, models, optimizers, utils, datasets
#from utils import display

current_2025_times=26# Lukcy 8 open times in 2024
issueCount = 65+351+351+351+351+current_2025_times
#快乐8在20201028发行第一期，在当年发行了65期，在2021年总共发行了351期， 2022 = 351 ,2023  - 351 截止20220115共发行了65+351+351+351+21=1139，大家可以在运行本代码时根据实际日期修改本变量。
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
row=["期号","开奖日期","前区号码","后区号码","总销售额(元)","奖池金额(元)"]
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
        sheet.write(i, 2, item['frontWinningNum'])
        sheet.write(i, 3, item['backWinningNum'])
        sheet.write(i, 4, item['saleMoney'])
        sheet.write(i, 5, item['prizePoolMoney'])
        i=i+1
# 保存
wb.save("快乐8开奖情况.xls")


#接下来读取.xls数据
data= xlrd.open_workbook('./快乐8开奖情况.xls')
table = data.sheets()[0]
data_lstm=[]
for i in range(issueCount,0,-1):#在excel中最新的数据在最上面因此要倒序读excel
    x=table.row(i)[2].value
    for j in range(20):
        data_lstm=np.append(data_lstm,float(x[3*j])*10+float(x[3*j+1]))
print(data_lstm)
data_np=data_lstm


# Read the Excel file into a pandas DataFrame
data = pd.read_excel('./快乐8开奖情况.xls')

# Split the numbers in the "开奖号码" column into separate columns
balls_df = data['开奖号码'].str.split(' ', expand=True)

# Rename the columns
balls_df.columns = [f'ball_{i+1}' for i in range(balls_df.shape[1])]

# Concatenate the DataFrame with the issue and open time columns
result_df = pd.concat([data[['期号', '开奖日期', '总销售额(元)']], balls_df], axis=1)

# change the sort of date
result_df = result_df.iloc[::-1]
result_df.reset_index(drop=True, inplace=True)

# change column data Types
result_df['开奖日期'] = pd.to_datetime(result_df['开奖日期'])
result_df[['ball_1', 'ball_2', 'ball_3', 'ball_4', 'ball_5', 'ball_6', 'ball_7', 'ball_8', 'ball_9', 'ball_10',
           'ball_11', 'ball_12', 'ball_13', 'ball_14', 'ball_15', 'ball_16', 'ball_17', 'ball_18', 'ball_19', 'ball_20']] = \
    result_df[['ball_1', 'ball_2', 'ball_3', 'ball_4', 'ball_5', 'ball_6', 'ball_7', 'ball_8', 'ball_9', 'ball_10',
                'ball_11', 'ball_12', 'ball_13', 'ball_14', 'ball_15', 'ball_16', 'ball_17', 'ball_18', 'ball_19', 'ball_20']].astype('int64')


# Display the resulting DataFrame
print(result_df)



# Step 1: Calculate the occurring rate of each number as a percentage
total_draws = len(data_np) // 20  # Total number of draws
number_counts = {}  # Dictionary to store occurrence counts for each number

for number in range(1, 81):
    occurrences = np.count_nonzero(data_np == number)
    occurring_rate = (occurrences / total_draws) * 100  # Convert to percentage
    number_counts[number] = (occurrences, occurring_rate)

# Step 2: Plot the bar chart for occurrence times
plt.figure(figsize=(12, 12))

# Plot the bar chart for occurrence times
plt.subplot(2, 1, 1)
bars = plt.bar(number_counts.keys(), [count[0] for count in number_counts.values()], color='skyblue')
plt.xlabel('Number')
plt.ylabel('Occurrence Times')
plt.title('History Lucky 8 Draw Numbers Occurrence Times')
plt.xticks(np.arange(1, 81, 1))  # Show ticks every numbers
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add annotations for occurrence times to each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom', ha='center')

# Plot the line chart for occurring rate
plt.subplot(2, 1, 2)
plt.plot(number_counts.keys(), [count[1] for count in number_counts.values()], color='red', marker='o', linestyle='-')
plt.xlabel('Number')
plt.ylabel('Occurring Rate (%)')
plt.title('History Lucky 8 Draw Numbers Occurring Rate')
plt.xticks(np.arange(1, 81, 1))  # Show ticks every numbers
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add annotations for occurring rate as percentage
for i, rate in enumerate([count[1] for count in number_counts.values()]):
    plt.text(i+1, rate, f'{rate:.2f}%', va='top', ha='center', color='red')

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

# Assuming number_counts is a dictionary
number_counts_df = pd.DataFrame.from_dict(number_counts)

# Print the summary statistics
number_counts_df.iloc[0].describe()
number_counts_df.iloc[1].describe()




#绘制快乐8的一维开奖数据
fig_size = plt.rcParams['figure.figsize']
fig_size[0] = 15
fig_size[1] = 5
plt.rcParams['figure.figsize'] = fig_size
plt.title("happy8 results")
plt.ylabel("Draw results")
plt.xlabel("Data")
plt.grid(True)
plt.autoscale(axis='x',tight=True)
plt.plot(data_np)
plt.show()


// 这个文件是原始的双色球数据导入脚本，目前已经不再使用，从网上下载而来。

import pandas as pd
import csv
import linecache

def get_one_page(page):
    url = 'http://kaijiang.zhcw.com/zhcw/html/ssq/list_%s.html' % (str(page))
    tb = pd.read_html(url, skiprows=[0, 1])[0]  # 跳过前两行 （去除前两行开奖日期	期号	中奖号码	销售额(元)	等信息，后面自己定义标题）
    return tb.drop([len(tb)-1])  # len(tb)是抓取的网页行数，去掉最后一行（去掉最后一行共116 页 /2318 条记录 首页 上一页 下一页 末页 当前第 1 页等信息）

with open(r'F:\PythonFiles\PycharmFile\ssq.csv', 'w', encoding='utf-8-sig', newline='') as csvFile:  #此行注意缩进，不是def定义中的代码#打开文件的方法
    csv.writer(csvFile).writerow(['开奖日期', '期号', '红1',  '红2', '红3', '红4', '红5', '红6', '蓝球','销售额(元)', '中奖注数一等奖', '中奖注数二等奖'])  #给csv文件中插入一行

    '''
for i in range(1,2):  # range（其实编号，总共）目前116页数据
    #第一种方法，整体写入数据，但不能把同单元格内各球数据分开
    get_one_page(i).to_csv(r'F:\PythonFiles\PycharmFile\ssq.csv', mode='a', encoding='utf_8_sig', header=0, index=0)
    print('第'+str(i)+'页抓取完成')
    '''
    #第二种方法，逐个写入数据
for i in range(1,2):  # range（其实编号，总共）目前116页数据
    reader=get_one_page(i)  #接收到第i页所有数据
    #print(reader[2][1])   #第二列第一行
    length=len(reader)  #获取该页数据的行数
    for j in range(0,length):
        col1=reader[0][j] #每一行第一列开奖日期给数据col1，后面存放到新的数据表中
        col2=reader[1][j]  #该行第二列开奖期号数据给col2
        col=list(filter(None,reader[2][j].split(" ")))  #将第三个存有号码的单元格进行拆分
        col3=col[0]  #第一个红球
        col4=col[1]  #第二个红球
        col5 = col[2]
        col6 = col[3]
        col7 = col[4]
        col8 = col[5]  #蓝球
        col9 = col[6]  # 蓝球
        col10 = reader[3][j]  #销售额
        col11 = reader[4][j]  #一等奖中奖注数
        col12 = reader[5][j]  #二等奖中奖注数
        with open(r'F:\PythonFiles\PycharmFile\ssq.csv', 'a', encoding='utf-8-sig', newline='') as csvFile:
            csv.writer(csvFile).writerow([col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11])  # 给csv文件中插入一行
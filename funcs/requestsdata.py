import time
import requests
import logging
import re
import json

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


def get_latest_issue_from_system(lottry_id):
    try:
        response = requests_data(1, 1, lottry_id)
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
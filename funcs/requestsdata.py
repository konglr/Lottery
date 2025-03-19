import time
import requests
import logging
import re
import json
import os  # 添加 os 模块的导入
import csv # 添加csv 模块的导入

# 获取脚本根目录
root_dir = os.getcwd()  # 或者使用 os.path.dirname(os.path.abspath(__file__))
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(root_dir, 'my_log_file.log'))  # 修改这行
    ]
)

def requests_data(pages, issue_count, ID, start_issue='', end_issue=''):
    headers = {
        'Connection': 'keep-alive',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.117 Safari/537.36',
        'Accept': '*/*',
        'Sec-Fetch-Site': 'same-site',
        'Sec-Fetch-Mode': 'no-cors',
        'Referer': 'https://www.zhcw.com/kjxx/',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'zh-CN,zh;q=0.9',
    }

    timestamp = int(time.time() * 1000)
    callback = f'jQuery1122035713028555611515_{timestamp}'
    tt = '0.123456789'
    _ = timestamp + 10
    params = {
        'callback': callback,
        'transactionType': '10001001',
        'lotteryId': ID,
        'issueCount': issue_count,
        'startDate': '',
        'endDate': '',
        'type': '0',
        'pageNum': pages,
        'pageSize': '30',
        'tt': tt,
        '_': _,
    }
    if start_issue and end_issue :
         params['startIssue'] = start_issue
         params['endIssue'] = end_issue

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

def parse_lottery_data(json_data):
    """解析 JSONP 响应并提取 data 字段"""
    try:
        # 使用正则匹配 JSONP 回调中的 JSON 内容
        match = re.search(r'\((.*)\);?$', json_data)
        if not match:
            logging.error("无法提取 JSON 内容")
            return None
        json_str = match.group(1)
        data = json.loads(json_str)
        if data.get('resCode') != '000000':
            logging.error(f"接口返回错误: {data.get('resMsg')}")
            return None
        return data.get('data', [])
    except json.JSONDecodeError as e:
        logging.error(f"JSON 解析错误: {e}, 原始数据片段: {json_data[:200]}...")
        return None
    except Exception as e:
        logging.error(f"解析出错: {e}")
        return None

def save_to_csv(data, filename):
    """将数据保存到 CSV 文件，自动创建 data 目录"""
    try:
        if not data:
            logging.warning("没有数据可以保存到 CSV 文件")
            return
        # 获取脚本根目录
        root_dir = os.getcwd()
        # 构建完整的文件路径
        filepath = os.path.join(root_dir, "data", filename)
        # 确保 data 目录存在
        os.makedirs(os.path.join(root_dir, "data"), exist_ok=True)

        with open(filepath, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile)
            header = data[0].keys()
            writer.writerow(header)
            for row in data:
                writer.writerow(row.values())
        logging.info(f"数据已成功保存到 {filepath}")
    except Exception as e:
        logging.error(f"保存 CSV 文件出错: {e}")

def get_lottery_data(lottery_id, lottery_name, pages=1, issue_count=10):
    """获取并保存指定彩票的数据"""
    filename = f'{lottery_name}_lottery_data.csv'
    json_data = requests_data(pages, issue_count, lottery_id)
    if json_data:
        lottery_data = parse_lottery_data(json_data)
        if lottery_data:
            save_to_csv(lottery_data, filename)
        else:
            logging.warning(f"未能解析到 {lottery_name} 的有效数据")
    else:
        logging.warning(f"未能获取到 {lottery_name} 的接口数据")

if __name__ == "__main__":
    lotteries = {
        "ssq": {"id": "1", "jc": "双色球"},
        "d3": {"id": "2", "jc": "福彩3D"},
        "qlc": {"id": "3", "jc": "七乐彩"},
        "kl8": {"id": "6", "jc": "快乐8"},
        "dlt": {"id": "281", "jc": "超级大乐透"},
        "pl3": {"id": "283", "jc": "排列三"},
        "pl5": {"id": "284", "jc": "排列五"},
        "xqxc": {"id": "287", "jc": "七星彩"},
    }

    for key, value in lotteries.items():
        get_lottery_data(value["id"], value["jc"])
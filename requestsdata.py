import time
import requests
import logging
import re
import json
import os  # 添加 os 模块的导入
import csv # 添加csv 模块的导入
import math
from tqdm import tqdm


# 获取脚本根目录
root_dir = os.getcwd()  # 或者使用 os.path.dirname(os.path.abspath(__file__))

# 获取上层目录
parent_dir = os.path.dirname(root_dir)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(root_dir, 'my_log_file.log'))  # 修改这行
    ]
)

logging.info(f"日志文件保存在：{os.path.join(root_dir, 'my_log_file.log')}")

def get_total_issue_count(lottery_id, before_issues):
    """
    获取系统最新期号，并计算总期数
    """
    latest_issue_in_system = get_latest_issue_from_system(lottery_id)
    if latest_issue_in_system is None:
        logging.error("❌ 无法获取最新期号，程序终止。")
        exit()

    # 特殊彩票（dlt, pl3, pl5, xqxc）计算方式不同
    special_lotteries = {"281", "283", "284", "287"}  # dlt, pl3, pl5, xqxc

    if lottery_id in special_lotteries:
        current_2025_times = latest_issue_in_system - 25000
    else:
        current_2025_times = latest_issue_in_system - 2025000

    # 计算总期数
    total_count = before_issues + current_2025_times
    logging.info(f"📌 {lottery_id} 最新期号: {latest_issue_in_system}, 总期数: {total_count}")

    return total_count


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
        'pageSize': '100',
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
    """解析 JSONP 响应，并提取 data 字段，转换成标准字段格式"""
    try:
        # 解析 JSONP 结构
        match = re.search(r"\((.*)\);?$", json_data)
        if not match:
            logging.error("❌ 无法提取 JSON 内容")
            return None

        json_str = match.group(1)
        data = json.loads(json_str)

        if data.get("resCode") != "000000":
            logging.error(f"❌ 接口返回错误: {data.get('resMsg')}")
            return None

        raw_records = data.get("data", [])
        if not raw_records:
            logging.warning("⚠️ 未提取到有效数据")
            return None

        structured_data = []
        for record in raw_records:
            if isinstance(record, dict):
                # 解析红球、蓝球
                record = extract_ball_numbers(record)

                # **✅ 直接保留 `winnerDetails`，不解析**
                structured_data.append(record)

        return structured_data

    except json.JSONDecodeError as e:
        logging.error(f"❌ JSON 解析错误: {e}, 原始数据片段: {json_data[:200]}...")
        return None
    except Exception as e:
        logging.error(f"❌ 解析出错: {e}")
        return None


def extract_ball_numbers(record):
    """
    解析 frontWinningNum 和 backWinningNum，动态生成红球和蓝球列
    :param record: 字典，包含 'frontWinningNum' 和 'backWinningNum'
    :return: 解析后的新字典，包含 '红球1'、'红球2'... 和 '篮球'/'蓝球1', '蓝球2'...
    """
    new_record = record.copy()  # 复制原始数据，避免修改原数据

    # 解析 frontWinningNum（红球）
    front_numbers = record.get("frontWinningNum", "").split()
    for i, num in enumerate(front_numbers, start=1):
        new_record[f"红球{i}"] = int(num)  # 动态创建列

    # 解析 backWinningNum（蓝球）
    back_numbers = record.get("backWinningNum", "").split()
    if len(back_numbers) == 1:
        new_record["篮球"] = int(back_numbers[0])  # 只有一个时叫 "篮球"
    else:
        for i, num in enumerate(back_numbers, start=1):
            new_record[f"蓝球{i}"] = int(num)  # 多个时叫 "蓝球1", "蓝球2"...

    # 删除原始字段，保持最终数据干净
    del new_record["frontWinningNum"]
    del new_record["backWinningNum"]

    return new_record



def save_to_csv(data, filename):
    """将数据保存到 CSV 文件，自动创建 data 目录"""
    try:
        if not data:
            logging.warning("没有数据可以保存到 CSV 文件")
            return
        # 获取脚本根目录
        root_dir = os.getcwd()  # 或者使用 os.path.dirname(os.path.abspath(__file__))

        # 构建完整的文件路径
        filepath= os.path.join(root_dir, "data", filename)
        # 确保 data 目录存在
        # os.makedirs(os.path.join(root_dir, "data"), exist_ok=True)

        with open(filepath, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile)
            header = data[0].keys()
            writer.writerow(header)
            for row in data:
                writer.writerow(row.values())
        logging.info(f"数据已成功保存到 {filepath}")
    except Exception as e:
        logging.error(f"保存 CSV 文件出错: {e}")


def get_lottery_data(lottery_id, lottery_name, before_issues):
    """获取彩票数据，计算 `total_count` 和 `pages`"""
    filename = f"{lottery_name}_lottery_data.csv"

    # 计算总期数
    total_count = get_total_issue_count(lottery_id, before_issues)

    # 计算总页数
    total_pages = math.floor(total_count / 100 + 1) if total_count % 100 == 0 else math.floor(total_count / 100 + 2)
    logging.info(f"📄 {lottery_name} 计算总页数: {total_pages}")

    all_data = []

    for page in tqdm(range(1, total_pages), desc=f"📥 下载 {lottery_name} 数据"):
        json_data = requests_data(page, total_count, lottery_id)
        if json_data:
            lottery_data = parse_lottery_data(json_data)
            if lottery_data:
                all_data.extend(lottery_data)

    # 保存数据
    save_to_csv(all_data, filename)

# =========== 主程序 =========== #
if __name__ == "__main__":
    lotteries = {
        "ssq": {"id": "1", "jc": "双色球", "before_issues": 3246},
        "d3": {"id": "2", "jc": "福彩3D", "before_issues": 7157},# 最早一期是2004001
        "qlc": {"id": "3", "jc": "七乐彩", "before_issues": 2500},
        "kl8": {"id": "6", "jc": "快乐8", "before_issues": 1470},
        "dlt": {"id": "281", "jc": "超级大乐透", "before_issues": 3800}, # 组早一期是08149
        "pl3": {"id": "283", "jc": "排列三", "before_issues": 5700}, #找到第一期 08355
        "pl5": {"id": "284", "jc": "排列五", "before_issues": 5657},#找到第一期 08355
        "xqxc": {"id": "287", "jc": "七星彩", "before_issues": 1828},#20100为第一期
    }

    for key, value in lotteries.items():
        get_lottery_data(value["id"], value["jc"], value["before_issues"])
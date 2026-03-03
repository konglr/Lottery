import time
import requests
import logging
import re
import json
import os  # 添加 os 模块的导入
import csv # 添加csv 模块的导入
import math
import pandas as pd
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


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

# 创建一个全局 Session 并配置自动重试
def create_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=5,  # 最大重试次数
        backoff_factor=1,  # 指数退避因子 (1s, 2s, 4s, 8s, 16s)
        status_forcelist=[429, 500, 502, 503, 504],  # 需要重试的状态码
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

session = create_session()

def get_total_issue_count(lottery_id):
    """
    通过接口自动获取该彩种在系统中的总记录数，无需每年手动维护。
    使用一个较大（假设大于历史总期数）的 issueCount 来触发 API 返回真实的 'total'。
    """
    try:
        # 请求较大的期数，pageSize=1 即可，目的是获取返回 JSON 中的 'total' 字段
        # 100000 已经远超目前所有彩种的总期数（如双色球约 3400 期）
        response = requests_data(1, 100000, lottery_id)
        if response is None:
            return 0

        match = re.search(r"\((.*)\)", response)
        if match:
            response = match.group(1)

        content = json.loads(response)
        total_count = int(content.get('total', 0))
        
        if total_count > 0:
            logging.info(f"📌 彩种 ID {lottery_id} 自动检测到系统总期数: {total_count}")
            return total_count
        else:
            # 如果获取失败，尝试通过获取一期数据来通过 latest_issue 估算（保留原逻辑作为 backup？）
            # 但经过测试，total 字段是非常可靠的，这里直接返回 0 并在调用处处理
            logging.warning(f"⚠️ 彩种 ID {lottery_id} 接口返回的总期数为 0 或无效")
            return 0
    except Exception as e:
        logging.error(f"❌ 自动获取系统总期数出错: {e}")
        return 0


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
        # 使用全局 session 发送请求，增加超时时间并禁用 SSL 验证（更鲁棒）
        response = session.get(
            'https://jc.zhcw.com/port/client_json.php', 
            headers=headers, 
            params=params, 
            timeout=15, 
            verify=False
        ).content.decode('utf-8')
        return response
    except requests.exceptions.RequestException as e:
        logging.error(f"网络请求错误: {e}")
        return None

def get_latest_issue_from_system(lottry_id):
    """获取系统中最新的期号"""
    try:
        response = requests_data(1, 1, lottry_id)
        if response is None:
            return None

        match = re.search(r"\((.*)\)", response)
        if match:
            response = match.group(1)

        content = json.loads(response)
        latest_issue = content['data'][0]['issue']
        return latest_issue
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


def get_lottery_data(lottery_id, lottery_name):
    """获取彩票数据，动态计算 `total_count` 并自动分页下载"""
    filename = f"{lottery_name}_lottery_data.csv"

    # 自动获取当前系统中该彩种的总期数
    total_count = get_total_issue_count(lottery_id)
    
    if total_count == 0:
        logging.error(f"❌ 无法确定 {lottery_name} 的总期数，跳过下载。")
        return

    # 计算总页数
    total_pages = math.floor(total_count / 100 + 1) if total_count % 100 == 0 else math.floor(total_count / 100 + 2)
    logging.info(f"📄 {lottery_name} 计算总页数: {total_pages}")

    all_data = []

    for page in tqdm(range(1, total_pages), desc=f"📥 下载 {lottery_name} 数据"):
        json_data = requests_data(page, total_count, lottery_id)
        if json_data:
            lottery_data = parse_lottery_data(json_data) # 分解红球和蓝球数据到单独列
            if lottery_data:
                all_data.extend(lottery_data)
        
        # 为了更小的连接冲击逻辑，每页抓取后增加一个微小随机延迟
        time.sleep(1)

    # 保存数据
    save_to_csv(all_data, filename)

def process_ssq_data(input_csv="data/双色球_lottery_data.csv", output_csv="data/双色球_processed_data.csv"):
    """
    自给自足的数据处理函数：
    1. 从原始 CSV 读取下载好的数据
    2. 解析中奖等级和奖金
    3. 计算奇偶、大小、三区、连号、跳号等指标
    4. 结果保存为 app.py 使用的格式
    """
    if not os.path.exists(input_csv):
        logging.error(f"找不到输入文件: {input_csv}")
        return

    logging.info(f"开始后期处理 {input_csv} ...")
    df = pd.read_csv(input_csv)

    # 字段重命名映射
    mapping = {
        'issue': '期号',
        'openTime': '开奖日期',
        'week': 'WeekDay',
        'saleMoney': '总销售额(元)',
        'prizePoolMoney': '奖池金额(元)',
        '篮球': '蓝球'
    }
    df = df.rename(columns=mapping)

    # 解析中奖信息
    awards = ['一等奖', '二等奖', '三等奖', '四等奖', '五等奖', '六等奖']
    for award in awards:
        df[f'{award}注数'] = 0
        df[f'{award}奖金'] = 0

    def parse_awards(row):
        details_str = row['winnerDetails']
        if pd.isna(details_str) or not details_str:
            return row
        try:
            # 处理单引号字符串，eval 在已知数据源下是快捷方式
            details = eval(details_str)
            for item in details:
                award_etc = item.get('awardEtc')
                base = item.get('baseBetWinner', {})
                try:
                    level = int(award_etc)
                    if 1 <= level <= 6:
                        row[f'{awards[level-1]}注数'] = base.get('awardNum', 0)
                        row[f'{awards[level-1]}奖金'] = base.get('awardMoney', 0)
                except: continue
        except: pass
        return row

    df = df.apply(parse_awards, axis=1)

    # 数字类型转换
    red_ball_columns = ['红球1', '红球2', '红球3', '红球4', '红球5', '红球6']
    for col in red_ball_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    def count_consecutive(nums):
        nums.sort()
        n, counts = len(nums), {'二连': 0, '三连': 0, '四连': 0, '五连': 0, '六连': 0}
        i = 0
        while i < n - 1:
            if nums[i] + 1 == nums[i + 1]:
                length = 2
                while i + length < n and nums[i + length - 1] + 1 == nums[i + length]: length += 1
                key = f"{['', '', '二', '三', '四', '五', '六'][length]}连"
                if 2 <= length <= 6: counts[key] += 1
                i += length
            else: i += 1
        return counts

    def count_jumps(nums):
        nums.sort()
        n, counts = len(nums), {'二跳': 0, '三跳': 0, '四跳': 0, '五跳': 0, '六跳': 0}
        i = 0
        while i < n - 1:
            diff = nums[i + 1] - nums[i]
            if diff >= 2:
                length = 1
                while i + length < n - 1 and nums[i + length + 1] - nums[i + length] == diff: length += 1
                jump_key = None
                if diff == 2: jump_key = {1: '二跳', 2: '三跳', 3: '四跳', 4: '五跳', 5: '六跳'}.get(length)
                elif diff in [3,4,5,6] and (length == diff-1 or length == diff):
                    jump_key = {2: '三跳', 3: '四跳', 4: '五跳', 5: '六跳'}.get(length)
                if jump_key:
                    counts[jump_key] += 1
                    i += (length + 1)
                else: i += 1
            else: i += 1
        return counts

    df = df.sort_values('期号', ascending=False).reset_index(drop=True)
    stats = []
    for i in range(len(df)):
        nums = sorted(df.loc[i, red_ball_columns].tolist())
        odd = sum(1 for n in nums if n % 2 == 1)
        small = sum(1 for n in nums if n <= 16)
        z1 = sum(1 for n in nums if 1 <= n <= 11)
        z2 = sum(1 for n in nums if 12 <= n <= 22)
        z3 = sum(1 for n in nums if 23 <= n <= 33)
        rep, adj = 0, 0
        if i < len(df) - 1:
            prev = set(df.loc[i+1, red_ball_columns].tolist())
            rep = len(set(nums) & prev)
            adj = sum(1 for n in nums if (n-1 in prev or n+1 in prev))
        
        sum_val = sum(nums)
        ac = len(set(abs(a-b) for a in nums for b in nums if a > b)) - 5
        
        res = {
            '奇数': odd, '偶数': 6 - odd, '小号': small, '大号': 6 - small,
            '一区': z1, '二区': z2, '三区': z3, '重号': rep, '邻号': adj, '孤号': 6-rep-adj,
            '和值': sum_val, 'AC': ac, '跨度': max(nums) - min(nums)
        }
        res.update(count_consecutive(nums))
        res.update(count_jumps(nums))
        stats.append(res)

    final_df = pd.concat([df, pd.DataFrame(stats)], axis=1)
    final_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    logging.info(f"✅ 后期处理完成: {output_csv}")

# =========== 主程序 =========== #
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

    # 下载完成后，自动触发双色球数据的后期加工处理
    process_ssq_data()
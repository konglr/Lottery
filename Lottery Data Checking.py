import os
import pandas as pd
import logging
from datetime import datetime

# 彩票信息字典
lotteries = {
    "ssq": {"id": "1", "jc": "双色球", "before_issues": 3246, "date_format": "%Y-%m-%d"},
    "d3": {"id": "2", "jc": "福彩3D", "before_issues": 7100, "date_format": "%Y-%m-%d"},
    "qlc": {"id": "3", "jc": "七乐彩", "before_issues": 2500, "date_format": "%Y-%m-%d"},
    "kl8": {"id": "6", "jc": "快乐8", "before_issues": 1470, "date_format": "%Y-%m-%d"},
    "dlt": {"id": "281", "jc": "超级大乐透", "before_issues": 2430, "date_format": "%Y-%m-%d"},
    "pl3": {"id": "283", "jc": "排列三", "before_issues": 5672, "date_format": "%Y-%m-%d"},
    "pl5": {"id": "284", "jc": "排列五", "before_issues": 5672, "date_format": "%Y-%m-%d"},
    "xqxc": {"id": "287", "jc": "七星彩", "before_issues": 5000, "date_format": "%Y-%m-%d"},
}

def check_lottery_data(lottery_name):
    """检验彩票数据"""

    if lottery_name not in lotteries:
        print(f"错误：不支持的彩票类型：{lottery_name}")
        return

    lottery = lotteries[lottery_name]
    filename = f"{lottery['jc']}_lottery_data.csv"

    # 获取脚本根目录
    root_dir = os.getcwd()

    filepath = os.path.join(root_dir, "data", filename)

    try:
        # 读取 CSV 文件
        df = pd.read_csv(filepath, header=0)
    except FileNotFoundError:
        print(f"错误：找不到文件：{filepath}")
        return
    except Exception as e:
        print(f"错误：读取文件失败：{e}")
        return

    # 1. 检查最早的数据是否符合预期
    expected_earliest_issue = df['issue'].min()  # 使用 'issue' 列
    earliest_issue = df['issue'].min()  # 使用 'issue' 列
    if earliest_issue != expected_earliest_issue:
        print(f"1.警告：最早的数据不是 {expected_earliest_issue}，而是 {earliest_issue}。")
    else:
        print("1.最早的数据符合预期。")

    # 2. 检查最新一期的数据是否与系统里的相同
    logging.basicConfig(filename='funcs/my_log_file.log', level=logging.INFO)

    last_issue_in_excel = df['issue'].max()  # 使用 'issue' 列
    last_issue_in_excel = int(last_issue_in_excel)

    # --- 比较 ---
    latest_issue_in_system = get_latest_issue_from_system(lottery["id"])

    if latest_issue_in_system is None:
        print("Failed to get latest issue from system.")
    else:
        if last_issue_in_excel == int(latest_issue_in_system):
            print(f"2.CSV与系统数据同步. 最新期数为: {last_issue_in_excel}")
            logging.info(f"Data synchronized: {last_issue_in_excel}")
        else:
            print(f"警告: CSV和系统数据不同步!")
            print(f"CSV: {last_issue_in_excel}, System: {latest_issue_in_system}")
            logging.warning(f"Data mismatch: CSV: {last_issue_in_excel}, System: {latest_issue_in_system}")

    # 3. 检查是否有重复数据
    duplicate_issues = df[df.duplicated(['issue'], keep=False)]  # 使用 'issue' 列
    if not duplicate_issues.empty:
        print("3.警告：发现重复的期号:")
        print(duplicate_issues[['issue']])  # 使用 'issue' 列
        print(f"   重复记录数量: {len(duplicate_issues)}") # 输出重复记录数量
    else:
        print("3.未发现重复数据。")

    # 4. 统计每年的开奖次数（如果数据包含开奖日期）
    if "openTime" in df.columns:  # 使用 'openTime' 列
        try:
            df['openTime'] = pd.to_datetime(df['openTime'], format=lottery["date_format"])  # 使用 'openTime' 列
            df['年份'] = df['openTime'].dt.year  # 使用 'openTime' 列
            yearly_counts = df.groupby('年份')['issue'].count()  # 使用 'issue' 列
            print(yearly_counts)
        except ValueError:
            print("警告：开奖日期格式不正确。")
    else:
        print("警告：数据中不包含开奖日期列。")

# 示例用法
check_lottery_data("d3")
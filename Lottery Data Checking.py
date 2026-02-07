import os
import pandas as pd
import logging
from datetime import datetime
from requestsdata import get_latest_issue_from_system

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
        print(f"\n❌ 错误：不支持的彩票类型：{lottery_name}")
        return

    lottery = lotteries[lottery_name]
    filename = f"{lottery['jc']}_lottery_data.csv"
    print(f"\n🔍 正在检查 {lottery['jc']} ({lottery_name})...")

    # 获取脚本根目录
    root_dir = os.getcwd()

    filepath = os.path.join(root_dir, "data", filename)

    try:
        # 读取 CSV 文件
        df = pd.read_csv(filepath, header=0)
    except FileNotFoundError:
        print(f"❌ 错误：找不到文件：{filepath}")
        return
    except Exception as e:
        print(f"❌ 错误：读取文件失败：{e}")
        return

    # 1. 检查数据行数及最早记录
    print(f"1. 数据总行数：{len(df)}")
    if not df.empty:
        # 找到期号最小的那一行（可能不是日期最小，如果是跨年份期号重置的情况）
        # 但通常我们关心的是历史上的第一期
        min_idx = df['issue'].idxmin()
        earliest_issue = df.loc[min_idx, 'issue']
        earliest_date = df.loc[min_idx, 'openTime'] if 'openTime' in df.columns else "未知"
        # 如果 openTime 是 datetime 对象，转换回字符串显示
        if isinstance(earliest_date, pd.Timestamp):
            earliest_date = earliest_date.strftime('%Y-%m-%d')
        print(f"   📅 最早一期：期号 [{earliest_issue}]，开奖日期 [{earliest_date}]")

    # 2. 检查最新一期的数据是否与系统里的相同
    # 配置日志
    log_path = os.path.join(root_dir, 'funcs', 'my_log_file.log')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(filename=log_path, level=logging.INFO)

    last_issue_in_csv = df['issue'].max()
    last_issue_in_csv = str(last_issue_in_csv)

    # --- 比较 ---
    latest_issue_in_system = get_latest_issue_from_system(lottery["id"])

    if latest_issue_in_system is None:
        print("2. ⚠️ 无法从系统获取最新期号进行同步检查。")
    else:
        latest_issue_in_system = str(latest_issue_in_system)
        if last_issue_in_csv == latest_issue_in_system:
            print(f"2. ✅ CSV与系统数据同步。最新期号为: {last_issue_in_csv}")
            logging.info(f"{lottery_name} synchronized: {last_issue_in_csv}")
        else:
            print(f"2. ⚠️ 警告: CSV和系统数据不同步!")
            print(f"   CSV 最新: {last_issue_in_csv}, 系统最新: {latest_issue_in_system}")
            logging.warning(f"{lottery_name} mismatch: CSV: {last_issue_in_csv}, System: {latest_issue_in_system}")

    # 3. 检查是否有重复数据 (基于 'issue' 期号)
    duplicate_mask = df.duplicated(['issue'], keep=False)
    if duplicate_mask.any():
        duplicate_issues = df[duplicate_mask].sort_values('issue')
        print(f"3. ⚠️ 警告：发现 [{len(duplicate_issues)}] 条记录存在期号重复问题:")
        # 列出重复的详细信息，方便用户对比
        cols_to_show = ['issue', 'openTime'] if 'openTime' in df.columns else ['issue']
        # 尝试增加红球列（如果存在）
        ball_cols = [c for c in df.columns if '球' in c]
        print(duplicate_issues[cols_to_show + ball_cols].to_string(index=False))
    else:
        print("3. ✅ 未发现重复期号。")

    # 4. 统计每年的开奖次数
    if "openTime" in df.columns:
        try:
            # 转换日期并处理可能的异常格式
            df['openTime'] = pd.to_datetime(df['openTime'], format=lottery["date_format"], errors='coerce')
            if df['openTime'].isnull().any():
                print("   ⚠️ 部分日期格式不正确，已设为 NaT")
            
            df['年份'] = df['openTime'].dt.year
            yearly_counts = df.groupby('年份')['issue'].count()
            print("4. 📅 每年开奖次数统计:")
            # 将 Series 转换为更易读的格式
            for year, count in yearly_counts.items():
                if pd.notnull(year):
                    print(f"   {int(year)}年: {count}期")
        except Exception as e:
            print(f"4. ❌ 统计年份出错: {e}")
    else:
        print("4. ⚠️ 警告：数据中不包含开奖日期列。")

if __name__ == "__main__":
    # 检查所有配置好的彩票
    for key in lotteries.keys():
        check_lottery_data(key)
        print("-" * 50)
import os
import pandas as pd
import logging
from request_data_all import requests_data, parse_lottery_data, get_latest_issue_from_system, process_ssq_data

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_local_latest_issue(filepath):
    """从本地 CSV 文件中获取最新的期号"""
    if not os.path.exists(filepath):
        return None
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            return None
        # 假设期号是第一列且已排序或我们可以取最大值
        return df['issue'].max()
    except Exception as e:
        logging.error(f"读取本地文件 {filepath} 出错: {e}")
        return None

def update_lottery_incremental(lottery_id, lottery_name):
    """增量更新指定彩种的数据"""
    filename = f"{lottery_name}_lottery_data.csv"
    filepath = os.path.join("data", filename)
    
    local_latest = get_local_latest_issue(filepath)
    system_latest = get_latest_issue_from_system(lottery_id)
    
    if system_latest is None:
        logging.warning(f"⚠️ 无法获取 {lottery_name} 的系统最新期号，跳过。")
        return

    # 统一转换为整数进行比较
    try:
        system_latest = int(system_latest)
        if local_latest is not None:
            local_latest = int(local_latest)
    except ValueError:
        logging.error(f"期号转换格式错误: local={local_latest}, system={system_latest}")
        return

    if local_latest is not None and local_latest >= system_latest:
        logging.info(f"✅ {lottery_name} 数据已是最新 (最新期号: {local_latest})。")
        return

    logging.info(f"🔄 {lottery_name} 发现新数据: 本地 {local_latest} -> 系统 {system_latest}")
    
    # 计算需要获取的数量 (这里简单起见，如果相差不大，直接请求前 N 条)
    # 也可以使用 startIssue 和 endIssue 过滤，但 API 的 issueCount 更直接
    if local_latest is None:
        # 如果没有本地数据，则退回到全量下载逻辑（或者报错提示用户先全量下载一次）
        logging.warning(f"⚠️ 未找到 {lottery_name} 本地数据，建议先运行 request_data_all.py 进行全量同步。")
        return

    # 简单计算一个需要下载的 issueCount 范围
    # 注意：期号并不一定是连续的整数差，但这种差值通常能覆盖缺少的数量
    # 为了保险，我们取 (系统期号 - 本地期号) 的绝对差，并加上一小段缓冲
    # 实际上，如果只缺几期，直接请求最近的 100 条 (pageSize) 过滤即可
    
    # 步骤：
    # 1. 请求最新的一页 (100条)
    # 2. 筛选出 issue > local_latest 的记录
    # 3. 将新记录合并到旧数据开头
    
    json_response = requests_data(1, 100, lottery_id)
    if not json_response:
        logging.error(f"❌ 抓取 {lottery_name} 增量数据失败")
        return
        
    new_records = parse_lottery_data(json_response)
    if not new_records:
        logging.warning(f"⚠️ 解析 {lottery_name} 增量数据为空")
        return
        
    df_new = pd.DataFrame(new_records)
    # 过滤掉已经存在的期号
    df_new['issue'] = df_new['issue'].astype(int)
    df_incremental = df_new[df_new['issue'] > local_latest]
    
    if df_incremental.empty:
        logging.info(f"ℹ️ {lottery_name} 虽然最新期号不同，但未在最近 100 条中发现更高期号，可能已同步。")
        return
        
    logging.info(f"➕ {lottery_name} 新增 {len(df_incremental)} 条记录")
    
    # 加载旧数据
    df_old = pd.read_csv(filepath)
    
    # 合并 (新数据在前，假设原始 CSV 是按降序排列的)
    df_final = pd.concat([df_incremental, df_old], ignore_index=True)
    
    # 按照期号降序排列以保证一致性
    df_final = df_final.sort_values('issue', ascending=False)
    
    # 保存覆盖
    df_final.to_csv(filepath, index=False, encoding='utf-8-sig')
    logging.info(f"💾 {lottery_name} 增量更新完成，已保存至 {filepath}")

def main():
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
        update_lottery_incremental(value["id"], value["jc"])

    # 专门为双色球触发后期处理
    #process_ssq_data()

if __name__ == "__main__":
    main()

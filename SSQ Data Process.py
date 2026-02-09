import pandas as pd
import os
import json
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_ssq_data(input_csv="data/双色球_lottery_data.csv", output_csv="data/双色球_processed_data.csv"):
    """
    1. 从双色球_lottery_data.csv读取数据
    2. 解析中奖细节 (winnerDetails)
    3. 计算各种特征 (奇偶, 大小, 三区, 重邻孤, 连号, 跳号, 和值, AC, 跨度)
    4. 生成并保存结果到数据/双色球_processed_data.csv
    """
    if not os.path.exists(input_csv):
        logging.error(f"找不到输入文件: {input_csv}")
        return

    logging.info(f"正在从 {input_csv} 导入数据...")
    df = pd.read_csv(input_csv)

    # 1. 基础字段映射与解析
    # Mapping CSV columns to the expected format
    # issue,openTime,seqFrontWinningNum,seqBackWinningNum,saleMoney,r9SaleMoney,prizePoolMoney,fixPoolMoney,week,tryCode,djEndTime,awardEndDesc,specialNotes,fixUBound,floatBound,winnerDetails,红球1,红球2,红球3,红球4,红球5,红球6,篮球
    
    mapping = {
        'issue': '期号',
        'openTime': '开奖日期',
        'week': 'WeekDay',
        'saleMoney': '总销售额(元)',
        'prizePoolMoney': '奖池金额(元)',
        '篮球': '蓝球'
    }
    df = df.rename(columns=mapping)

    # 解析 winnerDetails
    # 示例结构: [{'awardEtc': '1', 'baseBetWinner': {'remark': '一等奖', 'awardNum': '8', 'awardMoney': '5687274', 'totalMoney': ''}, ...}]
    awards = ['一等奖', '二等奖', '三等奖', '四等奖', '五等奖', '六等奖']
    for award in awards:
        df[f'{award}注数'] = 0
        df[f'{award}奖金'] = 0

    def parse_awards(row):
        details_str = row['winnerDetails']
        if pd.isna(details_str) or not details_str:
            return row
        try:
            # CSV 中可能是字符串化的 Python 列表 (使用了 repr 或 str)
            # 或者是标准 JSON。request_data_all.py 存的是 repr(list)
            # pd.read_csv 读进来如果是单引号，可能需要 ast.literal_eval 或者先处理成 json
            details = eval(details_str) # 安全起见通常用 ast.literal_eval
            for item in details:
                award_etc = item.get('awardEtc')
                base = item.get('baseBetWinner', {})
                try:
                    level = int(award_etc)
                    if 1 <= level <= 6:
                        row[f'{awards[level-1]}注数'] = base.get('awardNum', 0)
                        row[f'{awards[level-1]}奖金'] = base.get('awardMoney', 0)
                except:
                    continue
        except Exception as e:
            pass
        return row

    logging.info("正在解析中奖细节...")
    df = df.apply(parse_awards, axis=1)

    # 2. 数据计算部分
    red_ball_columns = ['红球1', '红球2', '红球3', '红球4', '红球5', '红球6']
    # 确保红球列为整数
    for col in red_ball_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # 计算连号函数
    def count_consecutive(nums):
        nums.sort()
        n = len(nums)
        counts = {'二连': 0, '三连': 0, '四连': 0, '五连': 0, '六连': 0}
        i = 0
        while i < n - 1:
            if nums[i] + 1 == nums[i + 1]:
                length = 2
                while i + length < n and nums[i + length - 1] + 1 == nums[i + length]:
                    length += 1
                key = f"{['', '', '二', '三', '四', '五', '六'][length]}连"
                if length >= 2 and length <= 6:
                    counts[key] += 1
                i += length
            else:
                i += 1
        return counts

    # 计算跳号函数
    def count_jumps(nums):
        nums.sort()
        n = len(nums)
        counts = {'二跳': 0, '三跳': 0, '四跳': 0, '五跳': 0, '六跳': 0}
        i = 0
        while i < n - 1:
            diff = nums[i + 1] - nums[i]
            if diff >= 2:
                length = 1
                while i + length < n - 1 and nums[i + length + 1] - nums[i + length] == diff:
                    length += 1
                
                # 这里保留原逻辑，但进行了重构
                jump_key = None
                if diff == 2:
                    mapping = {1: '二跳', 2: '三跳', 3: '四跳', 4: '五跳', 5: '六跳'}
                    jump_key = mapping.get(length)
                elif diff in [3, 4, 5, 6]:
                    mapping = {2: '三跳', 3: '四跳', 4: '五跳', 5: '六跳'}
                    # diff=3, len=2 -> 三跳; diff=4, len=3 -> 四跳 ...
                    if length == diff - 1: jump_key = mapping.get(length)
                    elif length == diff: jump_key = mapping.get(length) # 这里原逻辑稍微有点乱，为了兼容我通过 diff 和 length 判定
                
                if jump_key:
                    counts[jump_key] += 1
                    i += (length + 1)
                else:
                    i += 1
            else:
                i += 1
        return counts

    logging.info("开始计算各项特征指标...")
    # 结果容器
    results = []

    # 排序：因为计算重号邻号需要用到“上一期”，假设 CSV 是按期号倒序排的（最新的在前）
    # 检查顺序
    df = df.sort_values('期号', ascending=False).reset_index(drop=True)

    for i in range(len(df)):
        nums = sorted(df.loc[i, red_ball_columns].tolist())
        
        # 基础分析
        odd_count = sum(1 for num in nums if num % 2 == 1)
        small_count = sum(1 for num in nums if num <= 16)
        
        # 三区
        z1 = sum(1 for num in nums if 1 <= num <= 11)
        z2 = sum(1 for num in nums if 12 <= num <= 22)
        z3 = sum(1 for num in nums if 23 <= num <= 33)

        # 重邻孤 (对比上一期，即索引 i+1)
        repeat_count = 0
        adjacent_count = 0
        if i < len(df) - 1:
            prev_nums = set(df.loc[i+1, red_ball_columns].tolist())
            repeat_count = len(set(nums) & prev_nums)
            adjacent_count = sum(1 for num in nums if (num-1 in prev_nums or num+1 in prev_nums))
        
        consecutive = count_consecutive(nums)
        jumps = count_jumps(nums)
        
        sum_val = sum(nums)
        # AC 值
        diffs = sorted(set(abs(a-b) for a in nums for b in nums if a > b))
        ac = len(diffs) - 5
        span = max(nums) - min(nums)

        res = {
            '奇数': odd_count, '偶数': 6 - odd_count,
            '小号': small_count, '大号': 6 - small_count,
            '一区': z1, '二区': z2, '三区': z3,
            '重号': repeat_count, '邻号': adjacent_count, '孤号': 6 - repeat_count - adjacent_count,
            '和值': sum_val, 'AC': ac, '跨度': span
        }
        res.update(consecutive)
        res.update(jumps)
        results.append(res)

    res_df = pd.DataFrame(results)
    final_df = pd.concat([df, res_df], axis=1)

    # 保存
    final_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    logging.info(f"数据处理完成，结果已保存至 {output_csv}")

if __name__ == "__main__":
    process_ssq_data()
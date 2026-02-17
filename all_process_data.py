import pandas as pd
import ast
import os
import glob
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TIER_MAP = {
    '1': '一等奖', '2': '二等奖', '3': '三等奖',
    '4': '四等奖', '5': '五等奖', '6': '六等奖',
    '7': '七等奖', '8': '八等奖', '9': '九等奖'
}

def parse_awards(row, lottery_type):
    details_str = row['winnerDetails']
    if pd.isna(details_str) or not details_str:
        return row
    
    try:
        if isinstance(details_str, str):
            try:
                details = ast.literal_eval(details_str)
            except:
                details = eval(details_str)
        else:
            details = details_str
            
        if not isinstance(details, list):
            return row

        for item in details:
            award_etc = str(item.get('awardEtc', ''))
            base = item.get('baseBetWinner', {})
            remark = base.get('remark', '').strip()
            num = base.get('awardNum', 0)
            money = base.get('awardMoney', 0)

            # Determine Tier Name based on lottery type
            tier_name = ""
            
            # Normalization for PL3/PL5/3D
            if remark in ["组选3", "组选三"]: remark = "组选三"
            if remark in ["组选6", "组选六"]: remark = "组选六"
            if remark in ["组三"]: remark = "组三"
            if remark in ["组六"]: remark = "组六"
            
            if lottery_type in ['福彩3D', '排列三', '排列五']:
                # These must use remark for naming
                tier_name = remark
            elif remark and remark in ["一等奖", "二等奖", "三等奖", "四等奖", "五等奖", "六等奖", "七等奖"]:
                tier_name = remark
            elif award_etc in TIER_MAP:
                tier_name = TIER_MAP[award_etc]
            else:
                tier_name = remark if remark else f"奖项{award_etc}"
            
            if tier_name:
                def clean_val(v):
                    if isinstance(v, str):
                        v = v.replace(',', '').strip()
                        return float(v) if v else 0.0
                    return float(v) if v is not None else 0.0

                row[f'{tier_name}注数'] = clean_val(num)
                row[f'{tier_name}奖金'] = clean_val(money)

                # Handle DLT 追加
                add_winner = item.get('addToBetWinner')
                if add_winner and isinstance(add_winner, dict) and add_winner.get('awardNum'):
                    row[f'{tier_name}追加注数'] = clean_val(add_winner.get('awardNum', 0))
                    row[f'{tier_name}追加奖金'] = clean_val(add_winner.get('awardMoney', 0))
                
    except Exception as e:
        # Error in one row shouldn't stop others
        pass
    return row

def process_all_files():
    data_dir = 'data'
    files = glob.glob(os.path.join(data_dir, '*_lottery_data.csv'))
    
    for file_path in files:
        lottery_name = os.path.basename(file_path).replace('_lottery_data.csv', '')
        logging.info(f"Processing {file_path} (Type: {lottery_name})...")
        
        try:
            df = pd.read_csv(file_path)
            if 'winnerDetails' not in df.columns:
                logging.warning(f"No winnerDetails in {file_path}, skipping.")
                continue

            # 1. Define Standard Columns to Initialize
            unique_tiers = set()
            if lottery_name in ['双色球', '七乐彩', '七星彩']:
                unique_tiers = {"一等奖", "二等奖", "三等奖", "四等奖", "五等奖", "六等奖", "福运奖"}
            elif lottery_name == '超级大乐透':
                unique_tiers = {"一等奖", "二等奖", "三等奖", "四等奖", "五等奖", "六等奖", "七等奖"}
            elif lottery_name == '福彩3D':
                unique_tiers = {"单选", "组三", "组六"}
            elif lottery_name == '排列三':
                unique_tiers = {"直选", "组选三", "组选六"}
            elif lottery_name == '排列五':
                unique_tiers = {"直选"}
            elif lottery_name == '快乐8':
                CN_NUMS = {1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七", 8: "八", 9: "九", 10: "十"}
                kl8_rules = {
                    "选十": [10, 9, 8, 7, 6, 5, 0], "选九": [9, 8, 7, 6, 5, 4, 0],
                    "选八": [8, 7, 6, 5, 4, 0], "选七": [7, 6, 5, 4, 0],
                    "选六": [6, 5, 4, 3], "选五": [5, 4, 3],
                    "选四": [4, 3, 2], "选三": [3, 2],
                    "选二": [2], "选一": [1]
                }
                for mode, k_list in kl8_rules.items():
                    for k in k_list:
                        suffix = "全不中" if k == 0 else f"中{CN_NUMS[k]}"
                        unique_tiers.add(f"{mode}{suffix}")

            # 2. Aggressive Cleanup for PL3, PL5, 3D
            # These should NOT have "一等奖", "二等奖", etc.
            if lottery_name in ['排列三', '排列五', '福彩3D']:
                drop_cols = [c for c in df.columns if any(p in c for p in ["一等奖", "二等奖", "三等奖", "奖项"])]
                if drop_cols:
                    logging.info(f"Dropping incorrect columns for {lottery_name}: {drop_cols}")
                    df.drop(columns=drop_cols, inplace=True)

            # 3. Initialize Columns to 0
            for tier in unique_tiers:
                df[f'{tier}注数'] = 0.0
                df[f'{tier}奖金'] = 0.0
                if lottery_name == '超级大乐透':
                    df[f'{tier}追加注数'] = 0.0
                    df[f'{tier}追加奖金'] = 0.0

            # 4. Fill Data
            df = df.apply(lambda r: parse_awards(r, lottery_name), axis=1)
            
            # 5. Calculate Odd/Even Counts for Red Balls
            #    AND Big/Small Counts (New)
            red_cols = [c for c in df.columns if c.startswith('红球')]
            if red_cols:
                # Define Max Number for each lottery to determine midpoint
                # Default to SSQ (33) if unknown, but better to be precise
                max_num_map = {
                    '双色球': 33, '七乐彩': 30, '七星彩': 9,
                    '超级大乐透': 35, '福彩3D': 9, '排列三': 9, '排列五': 9,
                    '快乐8': 80
                }
                max_n = max_num_map.get(lottery_name, 33)
                midpoint = max_n / 2
                
                # Zone Boundaries (Trisection)
                # Zone 1: [Min, Max/3]
                # Zone 2: (Max/3, 2*Max/3]
                # Zone 3: (2*Max/3, Max]
                # For 3D (0-9): Max=9. z1=3, z2=6. Z1:0-3, Z2:4-6, Z3:7-9.
                # For SSQ (33): Max=33. z1=11, z2=22. Z1:1-11, Z2:12-22, Z3:23-33.
                z1_limit = max_n / 3.0
                z2_limit = (max_n * 2) / 3.0

                def count_stats(row):
                    odds = 0; evens = 0
                    bigs = 0; smalls = 0
                    z1 = 0; z2 = 0; z3 = 0
                    
                    for col in red_cols:
                        try:
                            val = int(row[col])
                            # Parity
                            if val % 2 != 0: odds += 1
                            else: evens += 1
                            
                            # Size
                            if val <= midpoint: smalls += 1
                            else: bigs += 1
                            
                            # Zones
                            # "Equal to intermediate value goes to downward zone"
                            # strictly strictly: if limit is 11.0, 11 <= 11.0 -> Zone 1
                            if val <= z1_limit:
                                z1 += 1
                            elif val <= z2_limit:
                                z2 += 1
                            else:
                                z3 += 1
                        except:
                            pass
                    return pd.Series([odds, evens, bigs, smalls, z1, z2, z3])

                df[['奇数', '偶数', '大号', '小号', '一区', '二区', '三区']] = df.apply(count_stats, axis=1)
            
            # 6. Calculate Consecutive Counts (New)
            if red_cols:
                def count_consecutive(row):
                    # Extract and sort numbers
                    try:
                        nums = sorted([int(row[c]) for c in red_cols])
                    except:
                        return pd.Series(dtype='int')
                        
                    n = len(nums)
                    # Initialize counters for 2-consecutive up to n-consecutive
                    # Uses Chinese keys
                    CN_KEYS = ["", "", "二", "三", "四", "五", "六", "七", "八", "九", "十", 
                               "十一", "十二", "十三", "十四", "十五", "十六", "十七", "十八", "十九", "二十"]
                    
                    counts = {f"{CN_KEYS[k]}连": 0 for k in range(2, min(n + 1, 21))}
                    
                    i = 0
                    while i < n - 1:
                        if nums[i] + 1 == nums[i + 1]:
                            length = 2
                            while i + length < n and nums[i + length - 1] + 1 == nums[i + length]:
                                length += 1
                            
                            if length <= 20: # Cap at 20 just in case
                                key = f"{CN_KEYS[length]}连"
                                if key in counts:
                                    counts[key] += 1
                            
                            i += length
                        else:
                            i += 1
                    return pd.Series(counts)

                # Initialize columns first to ensure they exist with 0
                max_consecutive = min(len(red_cols), 20)
                CN_KEYS_LIST = ["", "", "二", "三", "四", "五", "六", "七", "八", "九", "十", 
                                "十一", "十二", "十三", "十四", "十五", "十六", "十七", "十八", "十九", "二十"]
                for k in range(2, max_consecutive + 1):
                    col_name = f"{CN_KEYS_LIST[k]}连"
                    if col_name not in df.columns:
                        df[col_name] = 0

                consecutive_df = df.apply(count_consecutive, axis=1).fillna(0).astype(int)
                df.update(consecutive_df)
                for col in consecutive_df.columns:
                    df[col] = consecutive_df[col]

            # 7. Calculate Jump Counts (New)
            if red_cols:
                CN_KEYS_LIST = ["", "", "二", "三", "四", "五", "六", "七", "八", "九", "十", 
                                "十一", "十二", "十三", "十四", "十五", "十六", "十七", "十八", "十九", "二十"]
                
                def count_jumps(row):
                    try:
                        nums = sorted([int(row[c]) for c in red_cols if pd.notna(row[c])])
                    except:
                        return pd.Series(dtype='int')
                    
                    n = len(nums)
                    if n < 2:
                        return pd.Series(dtype='int')
                    
                    # Initialize counters for jumps
                    counts = {f"{CN_KEYS_LIST[k]}跳": 0 for k in range(2, min(n + 1, 21))}
                    
                    i = 0
                    while i < n - 1:
                        diff = nums[i + 1] - nums[i]
                        
                        if diff >= 2:
                            length = 1
                            while i + length < n - 1 and nums[i + length + 1] - nums[i + length] == diff:
                                length += 1
                            
                            jump_key = None
                            # 严格执行用户提供的参考逻辑
                            if diff == 2:
                                # length 1 (2 balls) -> 二跳, length 2 (3 balls) -> 三跳...
                                jump_key = {1: '二跳', 2: '三跳', 3: '四跳', 4: '五跳', 5: '六跳', 6: '七跳', 7: '八跳'}.get(length)
                                if not jump_key and length >= 2: # 扩展到更长的情况
                                    jump_key = f"{CN_KEYS_LIST[length + 1]}跳"
                            elif diff in [3, 4, 5, 6] and (length == diff - 1 or length == diff):
                                # 只有当长度与间距匹配时才记录
                                jump_key = {2: '三跳', 3: '四跳', 4: '五跳', 5: '六跳', 6: '七跳'}.get(length)
                            
                            if jump_key and jump_key in counts:
                                counts[jump_key] += 1
                                i += (length + 1)
                            else:
                                i += 1
                        else:
                            i += 1
                    
                    return pd.Series(counts)
                
                # Initialize jump columns
                max_jumps = min(len(red_cols), 20)
                for k in range(2, max_jumps + 1):
                    col_name = f"{CN_KEYS_LIST[k]}跳"
                    if col_name not in df.columns:
                        df[col_name] = 0
                
                jump_df = df.apply(count_jumps, axis=1).fillna(0).astype(int)
                for col in jump_df.columns:
                    df[col] = jump_df[col]

            # 8. Calculate Statistical Metrics (重号, 邻号, 孤号, 和值, AC, 跨度)
            if red_cols:
                # Initialize columns
                for col_name in ['重号', '邻号', '孤号', '和值', 'AC', '跨度']:
                    if col_name not in df.columns:
                        df[col_name] = 0
                
                # Calculate for each row
                for idx in df.index:
                    try:
                        # Get current row's red balls
                        nums = sorted([int(df.loc[idx, c]) for c in red_cols if pd.notna(df.loc[idx, c])])
                        
                        if not nums:
                            continue
                        
                        # 重号 and 邻号 (need previous draw)
                        rep, adj = 0, 0
                        if idx < len(df) - 1:  # Check if there's a previous row
                            try:
                                prev_nums = set([int(df.loc[idx + 1, c]) for c in red_cols if pd.notna(df.loc[idx + 1, c])])
                                current_set = set(nums)
                                
                                # 重号: numbers that appear in both current and previous draw
                                rep = len(current_set & prev_nums)
                                
                                # 邻号: numbers in current draw that are adjacent (+1 or -1) to previous draw
                                adj = sum(1 for n in nums if (n - 1 in prev_nums or n + 1 in prev_nums))
                            except:
                                pass
                        
                        # 孤号: numbers that are neither repeats nor adjacent
                        iso = len(nums) - rep - adj
                        
                        # 和值: sum of all red balls
                        sum_val = sum(nums)
                        
                        # AC: Arithmetic Complexity
                        # AC = number of unique differences - (red_count - 1)
                        if len(nums) >= 2:
                            diffs = set(abs(a - b) for a in nums for b in nums if a > b)
                            ac = len(diffs) - (len(nums) - 1)
                        else:
                            ac = 0
                        
                        # 跨度: range (max - min)
                        span = max(nums) - min(nums) if len(nums) >= 2 else 0
                        
                        # Assign values
                        df.loc[idx, '重号'] = rep
                        df.loc[idx, '邻号'] = adj
                        df.loc[idx, '孤号'] = iso
                        df.loc[idx, '和值'] = sum_val
                        df.loc[idx, 'AC'] = ac
                        df.loc[idx, '跨度'] = span
                        
                    except Exception as e:
                        logging.warning(f"Error calculating stats for row {idx}: {e}")
                        continue

            # Save
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
            logging.info(f"Successfully updated {file_path}")

        except Exception as e:
            logging.exception(f"Failed to process {file_path}: {e}")

if __name__ == "__main__":
    process_all_files()

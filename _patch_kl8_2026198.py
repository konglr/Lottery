"""
_one-off: append KL8 2026198 to 快乐8_lottery_data.csv

Fetch from upstream (jc.zhcw.com, lotteryId=6), build a row matching the
project's 172-column schema, derive stats using the same formulas as
request_process_all_data.py, insert into CSV (kept sorted by issue desc).

Usage: python3 _patch_kl8_2026198.py
"""
from __future__ import annotations
import csv, json, re, time, shutil, os, sys, warnings
from pathlib import Path
warnings.filterwarnings('ignore')

import requests

ROOT = Path(os.path.expanduser(
    '~/Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/Lottery'
))
CSV_PATH = ROOT / 'data' / '快乐8_lottery_data.csv'
BACKUP = ROOT / 'data' / '快乐8_lottery_data.csv.bak.2026198'
TARGET_ISSUE = '2026198'

# ---------- 1. 拉取 upstream ----------
def fetch_issue(issue_count=1):
    ts = int(time.time() * 1000)
    params = {
        'callback': 'jQuery1122_x',
        'transactionType': '10001001',
        'lotteryId': '6',                 # KL8
        'issueCount': issue_count,
        'type': '0',
        'pageNum': 1,
        'pageSize': '100',
        'tt': '0.123',
        '_': ts + 10,
    }
    headers = {
        'User-Agent': ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_2) '
                       'AppleWebKit/537.36 Chrome/79.0.3945.117 Safari/537.36'),
        'Referer': 'https://www.zhcw.com/kjxx/',
    }
    r = requests.Session().get(
        'https://jc.zhcw.com/port/client_json.php',
        headers=headers, params=params, timeout=15, verify=False
    ).content.decode('utf-8')
    m = re.search(r'jQuery1122_x\((.*)\)\s*$', r.strip())
    return json.loads(m.group(1))['data']

# ---------- 2. 红球列 & 衍生统计 ----------
RED_COLS = [f'红球{i}' for i in range(1, 21)]     # 快乐8 = 20 红球
CN_KEYS = ["", "", "二", "三", "四", "五", "六", "七",
           "八", "九", "十", "十一", "十二", "十三", "十四",
           "十五", "十六", "十七", "十八", "十九", "二十"]

def split_front(front_str: str) -> list[int]:
    nums = sorted(int(x) for x in front_str.split())
    if len(nums) != 20:
        raise ValueError(f'expected 20 reds, got {len(nums)}')
    return nums

def count_stats(nums: list[int], max_n=80) -> dict:
    """奇数/偶数/大号/小号/一区/二区/三区 — match request_process_all_data."""
    midpoint = max_n / 2
    z1_limit, z2_limit = max_n / 3.0, (max_n * 2) / 3.0
    odds = evens = bigs = smalls = z1 = z2 = z3 = 0
    for v in nums:
        if v % 2:        odds += 1
        else:            evens += 1
        if v <= midpoint: smalls += 1
        else:            bigs += 1
        if   v <= z1_limit: z1 += 1
        elif v <= z2_limit: z2 += 1
        else:               z3 += 1
    return {'奇数': odds, '偶数': evens,
            '大号': bigs, '小号': smalls,
            '一区': z1,   '二区': z2,   '三区': z3}

def count_consecutive(nums: list[int]) -> dict:
    n = len(nums)
    counts = {f'{CN_KEYS[k]}连': 0 for k in range(2, min(n + 1, 21))}
    i = 0
    while i < n - 1:
        if nums[i] + 1 == nums[i + 1]:
            length = 2
            while (i + length < n
                   and nums[i + length - 1] + 1 == nums[i + length]):
                length += 1
            if 2 <= length <= 20:
                key = f'{CN_KEYS[length]}连'
                if key in counts:
                    counts[key] += 1
            i += length
        else:
            i += 1
    return counts

def count_jumps(nums: list[int]) -> dict:
    """'k 跳' = run of k numbers with gap 0 between, then gap >=1 at end."""
    n = len(nums)
    counts = {f'{CN_KEYS[k]}跳': 0 for k in range(2, min(n + 1, 21))}
    i = 0
    while i < n - 1:
        if nums[i + 1] - nums[i] == 1:
            length = 2
            while (i + length < n
                   and nums[i + length] - nums[i + length - 1] == 1):
                length += 1
            if length + 1 <= n and length >= 2 and length <= 20:
                key = f'{CN_KEYS[length]}跳'
                if key in counts:
                    counts[key] += 1
            i += length
        else:
            i += 1
    return counts

def ac_value(nums: list[int]) -> int:
    if len(nums) < 2:
        return 0
    diffs = {abs(a - b) for a in nums for b in nums if a > b}
    return len(diffs) - (len(nums) - 1)

# ---------- 3. 主流程 ----------
def main():
    print(f'>>> CSV: {CSV_PATH}')

    # 备份
    if not BACKUP.exists():
        shutil.copy2(CSV_PATH, BACKUP)
        print(f'>>> backup -> {BACKUP.name}')

    # 读 CSV
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        rd = csv.reader(f)
        header, rows = next(rd), list(rd)
    print(f'>>> header={len(header)} rows={len(rows)}')

    # 拉取 upstream
    api = fetch_issue(1)
    new_rec = next(r for r in api if str(r['issue']) == TARGET_ISSUE)
    print(f'>>> upstream issue={new_rec["issue"]} '
          f'openTime={new_rec["openTime"]} '
          f'front={new_rec["frontWinningNum"]}')

    # 防重
    if any(r[header.index('issue')] == TARGET_ISSUE for r in rows):
        print(f'!!!  {TARGET_ISSUE} 已在 CSV 内，无需插入')
        return

    # 当前 CSV 最新期号（同列号 = 上一期，用于重号/邻号/孤号）
    idx_issue = header.index('issue')
    existing_issues = sorted(
        (int(r[idx_issue]) for r in rows if r[idx_issue].isdigit()),
        reverse=True
    )
    latest_local = str(existing_issues[0])
    print(f'>>> local latest issue = {latest_local}')

    # 计算本行 red 列 + 衍生
    nums = split_front(new_rec['frontWinningNum'])
    print(f'>>> reds (sorted): {nums}')
    print(f'>>> sum={sum(nums)} max-min={max(nums)-min(nums)} '
          f'ac={ac_value(nums)}')

    stats = count_stats(nums)
    cons = count_consecutive(nums)
    jmps = count_jumps(nums)

    # 前一期用来算 重号/邻号/孤号
    prev_row = next(r for r in rows if r[idx_issue] == latest_local)
    prev_nums = set()
    for col in RED_COLS:
        v = prev_row[header.index(col)]
        try:
            prev_nums.add(int(v))
        except (ValueError, TypeError):
            pass
    current_set = set(nums)
    rep = len(current_set & prev_nums)
    adj = sum(1 for n in nums if (n - 1 in prev_nums or n + 1 in prev_nums))
    iso = len(nums) - rep - adj
    sum_val = sum(nums)
    ac = ac_value(nums)
    span = max(nums) - min(nums)
    print(f'>>> 重号={rep} 邻号={adj} 孤号={iso} '
          f'和值={sum_val} AC={ac} 跨度={span}')

    # ---------- 拼新行：按 header 顺序填空 ----------
    new_row = [''] * len(header)
    # 直接来自 upstream
    DIRECT_MAP = {
        'AC': 'AC',           # placeholder, computed below
        'issue': 'issue',
        'openTime': 'openTime',
        'frontWinningNum': 'frontWinningNum',
        'backWinningNum': 'backWinningNum',
        'seqFrontWinningNum': 'seqFrontWinningNum',
        'seqBackWinningNum': 'seqBackWinningNum',
        'saleMoney': 'saleMoney',
        'r9SaleMoney': 'r9SaleMoney',
        'prizePoolMoney': 'prizePoolMoney',
        'fixPoolMoney': 'fixPoolMoney',
        'week': 'week',
        'tryCode': 'tryCode',
        'djEndTime': 'djEndTime',
        'awardEndDesc': 'awardEndDesc',
        'fixUBound': 'fixUBound',
        'floatBound': 'floatBound',
        'specialNotes': 'specialNotes',
    }
    for col_idx, col_name in enumerate(header):
        if col_name in DIRECT_MAP:
            v = new_rec.get(DIRECT_MAP[col_name], '')
            new_row[col_idx] = '' if v is None else str(v)
    # 红球 1..20
    for i in range(20):
        new_row[header.index(f'红球{i+1}')] = str(nums[i])
    # 蓝球 (KL8 = -1)
    new_row[header.index('蓝球')] = '-1'
    def w(col, val):
        """write if column exists in header; ignore otherwise."""
        if col in header:
            new_row[header.index(col)] = str(val)

    # 统计
    for k, v in stats.items(): w(k, v)
    for k, v in cons.items():  w(k, v)
    for k, v in jmps.items():  w(k, v)
    # 重/邻/孤/和/AC/跨度
    w('重号', rep); w('邻号', adj); w('孤号', iso)
    w('和值', sum_val); w('AC', ac); w('跨度', span)

    # winnerDetails — upstream 已经给的是 list-of-dict 的字符串化版本；
    # 跟现有 197 行的格式完全一致 (Python repr of list of dicts with single quotes)
    new_row[header.index('winnerDetails')] = str(new_rec['winnerDetails'])

    # 计算 column AC 之外的 AC 列（如果 header 里有其他列叫 "AC" ）
    # 已经覆盖。

    # 选X中X 注数/奖金（不在 upstream payload → 新行留空，但 197 行有填写）
    # 这些其实是 当天全部到位 后的 4-6 小时汇总，KL8 统计有时隔天；
    # 跨夜/隔几天再来才补 —— 跟历史 197 那行入档情况保持一致，初始留空也无所谓
    # 我们仅填派生列，红球列，上游直接列；其他列保持空。

    # ---------- 排序并写回 ----------
    # 现 CSV 按 issue desc（PKL8 spec）；把新行放在 rows 最前
    rows.insert(0, new_row)
    with open(CSV_PATH, 'w', encoding='utf-8', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(header)
        wr.writerows(rows)

    print(f'>>> 已写入 {CSV_PATH}')
    print(f'>>> 新行数: {len(rows)}, '
          f'头 3 期: {[rows[i][idx_issue] for i in range(3)]}')

    # 简单 sanity
    import pandas as pd
    df = pd.read_csv(CSV_PATH)
    kl8 = df[df['issue'].astype(str).str.startswith('2026')]
    print(f'>>> 2026 年累计期数: {len(kl8)}')
    print('>>> 2026 年首末三期:')
    for _, r in kl8.head(1).iterrows():
        print(f'    {r["issue"]} {r["openTime"]} {r["frontWinningNum"]}')
    for _, r in kl8.tail(2).iterrows():
        print(f'    {r["issue"]} {r["openTime"]} {r["frontWinningNum"]}')

if __name__ == '__main__':
    main()

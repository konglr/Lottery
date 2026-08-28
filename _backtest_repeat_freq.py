"""
_one-off 回测 — 快乐8 "前三期出现次数" 选号策略

逻辑:
  for each T in [3, 4, ..., N]:
      window = last_3[T-3:T]   # T-3, T-2, T-1 开奖 20 红球
      freq = Counter()         # 1..80 每个号码在 window 里出现次数
      bucket = {0:[],1:[],2:[],3:[]}
      for n in 1..80:
          f = freq.get(n, 0)
          bucket[f].append(n)
      actual = set(红球 in T 期)
      hit = actual & set(bucket[k])
      record:
        bucket_size[k]        # 池子大小
        hit_size[k]           # 池子命中 T 期号码的个数
        hit_probability[k]    # hit_size / bucket_size

输出:
  1. 四池大小 / 平均命中数 / 平均命中率
  2. 策略 = 选 freq=1 池，T 期实际命中均值 + 标准差
  3. 命中率分布 (10-bin 直方图)
  4. 跨年汇总

并把 T 期实际红球 / 上一周期 freq=1 池都写进 reports/, 出 csv 报告
"""
from __future__ import annotations
import csv, json, statistics
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime

ROOT = Path(os.path.expanduser(
    '~/Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/Lottery'
)) if False else None
import os
ROOT = Path(os.path.expanduser(
    '~/Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/Lottery'
))
CSV = ROOT / 'data' / '快乐8_lottery_data.csv'
OUT_DIR = ROOT / 'reports'
OUT_DIR.mkdir(exist_ok=True)

# ---------- 读 CSV，按 issue asc ----------
def load():
    rows = []
    with open(CSV, 'r', encoding='utf-8') as f:
        rd = csv.reader(f)
        header = next(rd)
        i_issue = header.index('issue')
        i_open  = header.index('openTime')
        i_front = header.index('frontWinningNum')
        for r in rd:
            reds = tuple(int(x) for x in r[i_front].split())
            if len(reds) != 20:
                continue
            rows.append({
                'issue': r[i_issue],
                'openTime': r[i_open],
                'reds': set(reds),
                'reds_list': reds,
            })
    rows.sort(key=lambda x: int(x['issue']))
    return rows

def backtest(rows, window=3):
    """
    Returns list of records, one per T (where T >= window+1).
    Each record dict:
      issue, openTime, bucket_size{0,1,2,3}, hit{0,1,2,3},
      hit_probability{0,1,2,3}, freq1_pool, freq1_hits_list
    """
    out = []
    for T in range(window, len(rows)):
        prev3 = rows[T-window : T]            # T-3,T-2,T-1
        cur   = rows[T]
        # 统计 1..80 在前 3 期出现次数
        freq = Counter()
        for p in prev3:
            freq.update(p['reds'])
        bucket = defaultdict(list)
        for n in range(1, 81):
            f = freq.get(n, 0)
            bucket[f].append(n)
        actual = cur['reds']
        hit = {k: len(set(bucket[k]) & actual) for k in (0, 1, 2, 3)}
        size = {k: len(bucket[k]) for k in (0, 1, 2, 3)}
        # 注意: 80 - sum(size) 可能 > 0 if window 长度 < 80, but window=3所以总和必然 == 60 (= 3*20)
        # 所以 size 总和 = 60; 0池最大, 1池中等, 2,3 池小
        prob  = {k: hit[k] / size[k] if size[k] else 0.0 for k in (0, 1, 2, 3)}
        out.append({
            'issue': cur['issue'],
            'openTime': cur['openTime'],
            'bucket_size': size,
            'hit': hit,
            'hit_probability': prob,
            'freq1_pool': bucket[1],
            'freq1_hits': sorted(actual & set(bucket[1])),
        })
    return out

def main():
    rows = load()
    print(f'>>> total rows: {len(rows)}')
    print(f'>>> first: {rows[0]["issue"]}  last: {rows[-1]["issue"]}')

    records = backtest(rows, window=3)
    print(f'>>> backtest records (need T>=4): {len(records)}')

    # ----- 总体聚合 -----
    # size 应该是常数: 0池=(80-60)=20, 1池=?
    # 数学上: 60 个球位放在 80 球中, 单球出现 0/1/2/3 次的期望是
    # E[freq=k] = C(3,k) * (20/80)^k * (60/80)^(3-k); 但实测会有波动
    bs = [r['bucket_size'] for r in records]
    bs0 = statistics.mean([b[0] for b in bs])
    bs1 = statistics.mean([b[1] for b in bs])
    bs2 = statistics.mean([b[2] for b in bs])
    bs3 = statistics.mean([b[3] for b in bs])
    print(f'>>> 平均池子大小:  0→{bs0:.1f}  1→{bs1:.1f}  2→{bs2:.1f}  3→{bs3:.1f}')

    # 各池命中率
    hp = [r['hit_probability'] for r in records]
    print('\n>> 各池命中率 (平均 / 最大 / 最小 / std)')
    for k in (0, 1, 2, 3):
        ms = [r['hit_probability'][k] for r in records]
        print(f'   k={k}: mean={statistics.mean(ms):.4f}  '
              f'max={max(ms):.4f}  min={min(ms):.4f}  '
              f'std={statistics.pstdev(ms):.4f}  '
              f'median={statistics.median(ms):.4f}')

    # 频次绝对命中数: 各池 T 期实际开出的号码数
    print('\n>> 实际命中数 (分布 T=每期)')
    for k in (0, 1, 2, 3):
        hits = [r['hit'][k] for r in records]
        cnt = Counter(hits)
        dist = '  '.join(f'{n}次:{c}' for n, c in sorted(cnt.items()))
        print(f'   k={k}: avg={statistics.mean(hits):.2f}  '
              f'median={statistics.median(hits)}  '
              f'max={max(hits)}  dist=({dist})')

    # ----- 策略: 选 freq=1 池 -----
    print('\n=========================')
    print('策略: 选 freq=1 池作为预测')
    print('=========================')
    f1_hits_per_t = [r['hit'][1] for r in records]
    f1_hits_size  = [r['bucket_size'][1] for r in records]
    cnt = Counter(f1_hits_per_t)
    print(f'观察期数: {len(records)}')
    print(f'freq=1 池平均大小: {statistics.mean(f1_hits_size):.2f} '
          f'(单次波动: 最小 {min(f1_hits_size)}, 最大 {max(f1_hits_size)})')
    print(f'每期命中数 (T=近一期 20 红球 ∩ freq1 池):')
    print('  命中数 : 期数 : 频率')
    for n in sorted(cnt.keys()):
        c = cnt[n]
        bar = '█' * int(round(c / len(records) * 80))
        print(f'    {n:>2}     : {c:>5}  ({c/len(records)*100:5.2f}%)  {bar}')
    avg = statistics.mean(f1_hits_per_t)
    sd  = statistics.pstdev(f1_hits_per_t)
    print(f'\n均值={avg:.3f}  标准差={sd:.3f}  占比={avg/20*100:.2f}%')

    # 基准: 纯随机 20/80 = 0.25
    expected_random = 20  # size 平均值/4 * 0.25 ...
    # 严格 E[|freq=1 池 ∩ T 期|] = |freq=1| * (20/80)
    pool1_avg = statistics.mean(f1_hits_size)
    expected_random = pool1_avg * 20 / 80
    print(f'纯随机基线 = |freq=1池| × 20/80 = {pool1_avg:.2f} × 0.25 = {expected_random:.3f}')
    print(f'策略均值 / 随机基线 = {avg / expected_random:.3f}')

    # ----- 跨窗口大小做敏感性: window = 1..6 -----
    print('\n=========================')
    print('敏感性: 窗口 window = 1..6')
    print('=========================')
    print(f'  win | pool1 size | hit mean | baseline | ratio')
    for win in (1, 2, 3, 4, 5, 6):
        rb = backtest(rows, window=win)
        if not rb:
            continue
        ps = [r['bucket_size'][1] for r in rb if 1 in r['bucket_size']]
        hm = [r['hit'][1] for r in rb]
        if not hm:
            continue
        pool1 = statistics.mean(ps)
        base = pool1 * 20 / 80
        mean_h = statistics.mean(hm)
        print(f'   {win}  |   {pool1:6.2f}   |   {mean_h:.3f}  |  {base:.3f}  |'
              f'  {mean_h/base:.3f}')

    # ----- 写 csv 报告 -----
    csv_out = OUT_DIR / 'kl8_freq_window3_backtest_detail.csv'
    with open(csv_out, 'w', encoding='utf-8', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['issue', 'openTime',
                     'sz0','sz1','sz2','sz3',
                     'hit0','hit1','hit2','hit3',
                     'p0','p1','p2','p3',
                     'freq1_pool', 'freq1_hits'])
        for r in records:
            wr.writerow([r['issue'], r['openTime'],
                         r['bucket_size'][0],r['bucket_size'][1],
                         r['bucket_size'][2],r['bucket_size'][3],
                         r['hit'][0],r['hit'][1],r['hit'][2],r['hit'][3],
                         f'{r["hit_probability"][0]:.4f}',
                         f'{r["hit_probability"][1]:.4f}',
                         f'{r["hit_probability"][2]:.4f}',
                         f'{r["hit_probability"][3]:.4f}',
                         ' '.join(map(str, sorted(r['freq1_pool']))),
                         ' '.join(map(str, r['freq1_hits']))])
    print(f'\n>>> 详细数据已写入 {csv_out}')

    # ----- md 报告 -----
    md = []
    md.append('# 快乐8 "前三期出现次数" 策略回测')
    md.append('')
    md.append(f'> 回测时间: {datetime.now().strftime("%Y-%m-%d %H:%M")}  ')
    md.append(f'> 数据源: `data/快乐8_lottery_data.csv`  ')
    md.append(f'> 总期数: **{len(rows)}** (issue {rows[0]["issue"]} ~ {rows[-1]["issue"]})  ')
    md.append(f'> 回测期数: **{len(records)}** (前 3 期必须有数据, T≥4)  ')
    md.append(f'> 详细CSV: `{csv_out.relative_to(ROOT)}`  ')
    md.append('')
    md.append('## 1. 各池基础统计 (T-3,T-2,T-1 出现次数)')
    md.append('')
    md.append('| 出现次数 | 平均池大小 | 平均 T 期命中数 | 平均命中率 | 中位命中率 | std |')
    md.append('|---------|-----------|---------------|----------|----------|-----|')
    for k in (0, 1, 2, 3):
        ms = [r['hit_probability'][k] for r in records]
        hs = [r['hit'][k] for r in records]
        sz = [r['bucket_size'][k] for r in records]
        md.append(f'| **{k}** 次 | {statistics.mean(sz):.2f} | {statistics.mean(hs):.3f} '
                  f'| {statistics.mean(ms)*100:.2f}% | {statistics.median(ms)*100:.2f}% '
                  f'| {statistics.pstdev(ms)*100:.2f}% |')
    md.append('')
    md.append('**说明**: 窗口=3 期 = 60 个号码位. 总球池 80 个, 所以:  ')
    md.append('- 0次 池平均最大 (约 20-40 个号码)  ')
    md.append('- 3次 池最小 (期望 20×(20/80)^3 ≈ 0.156 个, 实际常有 0/1/2)  ')
    md.append('- 1次 池是大部分 "中等热度" 号码  ')
    md.append('')
    md.append('## 2. 策略: 选"前 3 期开过 1 次"作为下一期预测')
    md.append('')
    f1_hits = [r['hit'][1] for r in records]
    md.append(f'- 观察期数: **{len(records)}**')
    md.append(f'- 池平均大小: **{statistics.mean(f1_hits_size):.2f}** 个号码')
    md.append(f'- 池大小波动: 最小 **{min(f1_hits_size)}** | 最大 **{max(f1_hits_size)}**')
    md.append('')
    md.append('| 命中数 | 期数 | 频率 |')
    md.append('|--------|------|------|')
    for n in sorted(cnt.keys()):
        c = cnt[n]
        md.append(f'| {n} | {c} | {c/len(records)*100:.2f}% |')
    md.append(f'| **均值** | — | **{avg:.3f} / 20 = {avg/20*100:.2f}%** |')
    md.append('')
    md.append(f'**纯随机基线**: |freq=1池| × 20/80 = {pool1_avg:.2f} × 0.25 = **{expected_random:.3f}**  ')
    md.append(f'**策略 / 基线 = {avg / expected_random:.3f}** (1.0 = 与随机无差别, >1 = 略优, <1 = 略差)  ')
    md.append('')
    md.append('## 3. 敏感性: 不同窗口期数对比')
    md.append('')
    md.append('| 窗口 | freq=1 平均池 | 平均命中数 | 随机基线 | 倍率 |')
    md.append('|------|--------------|-----------|---------|------|')
    for win in (1, 2, 3, 4, 5, 6):
        rb = backtest(rows, window=win)
        if not rb: continue
        ps = [r['bucket_size'][1] for r in rb]
        hm = [r['hit'][1] for r in rb]
        if not hm: continue
        pool1 = statistics.mean(ps)
        base = pool1 * 20 / 80
        mean_h = statistics.mean(hm)
        md.append(f'| {win} 期 | {pool1:.2f} | {mean_h:.3f} | {base:.3f} | '
                  f'**{mean_h/base:.3f}** |')
    md.append('')
    md.append('## 4. 结论')
    md.append('')
    if avg / expected_random > 1.05:
        md.append('⚠️ **该策略略优于纯随机**, 但需要关注:')
    elif avg / expected_random < 0.95:
        md.append('❌ **该策略略劣于纯随机**.')
    else:
        md.append('📊 **该策略与纯随机几乎无差别**.')
    md.append('')
    md.append('**Lucky 提醒**:  ')
    md.append('快乐8 是独立随机事件, 任何"按过往选择下一期号码"的策略命中率期望值 = 20/80 = 0.25.  ')
    md.append('真正能提高"命中率"的方式是**扩大池子**(例如把 freq=0,1,2 都纳入), '
              '但这同时增加了覆盖号码数, 回报期望值不变 (投注金额也变大).  ')
    md.append('真正有用的是**覆盖率**：freq=1 池平均只有约 20+ 个号码, 你实际是 "20 选 20 个球的命中率" — 详见下表.')

    md_path = OUT_DIR / 'kl8_freq_window3_strategy_backtest.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))
    print(f'>>> 报告已写入 {md_path}')

if __name__ == '__main__':
    main()

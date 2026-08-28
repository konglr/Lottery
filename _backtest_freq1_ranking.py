"""
_one-off — 策略 v2: 在 freq=1 池子上对每个候选打分, 按分数 Top N, 看 N=15..30 覆盖率
"""

from __future__ import annotations
import csv, json, statistics
from collections import Counter, defaultdict, deque
from pathlib import Path
from datetime import datetime
import os

ROOT = Path(os.path.expanduser(
    '~/Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/Lottery'
))
CSV = ROOT / 'data' / '快乐8_lottery_data.csv'
OUT_DIR = ROOT / 'reports'
OUT_DIR.mkdir(exist_ok=True)

# ---------- 数据 ----------
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

# ---------- 评分维度 ----------
def score_methods_for_pool(pool, freq_full_history_deque, recent10, recent30, all_history):
    """
    freq_full_history_deque: deque[(issue, 20 red)]  已经包含最近 100 期, 按时间 asc
    recent10 / recent30 / all_history: list of issues

    给每个候选算 4 个分, 最后给一个加权综合分.
    """
    scores = {}
    for n in pool:
        # 1. 遗漏值 (距离上次出现的期数, 上限 100)
        miss = 0
        for entry in reversed(recent10 if len(recent10) > 0 else deque()):
            ...
        # 用 freq_full_history_deque 倒序找
        miss = 0
        for entry in reversed(freq_full_history_deque):
            if n in entry[1]:
                break
            miss += 1
        miss = min(miss, 100)
        score_miss = miss  # 越大越好

        # 2. 加权频率: 近 10 期 / 近 30 期 / 全期 加权
        cnt10  = sum(1 for e in recent10  if n in e[1])
        cnt30  = sum(1 for e in recent30  if n in e[1])
        cntAll = sum(1 for e in all_history if n in e[1])
        denom10 = max(len(recent10), 1)
        denom30 = max(len(recent30), 1)
        denomA  = max(len(all_history), 1)
        score_freq = (3 * cnt10/denom10 + 2 * cnt30/denom30 + 1 * cntAll/denomA) / 6.0 * 100

        # 3. 邻号潜力: 近 3 期, 哪些号码 +/-1 曾开过
        neigh = 0
        for e in recent10[-3:]:
            r = e[1]
            if (n-1) in r or (n+1) in r or (n-2) in r or (n+2) in r:
                neigh += 1
        score_neigh = neigh

        # 4. 末日号: 近 3 期所有号码末位 (n%10) 出现过几次
        lastdig = n % 10
        ld_count = 0
        for e in recent10[-3:]:
            r = e[1]
            for x in r:
                if x % 10 == lastdig:
                    ld_count += 1
        # 越多越好 (该末位热度)
        score_last = ld_count

        # 综合分
        score_total = (
            score_miss * 0.4 +
            score_freq * 0.35 +
            score_neigh * 1.5 +
            score_last * 0.5
        )

        scores[n] = {
            'miss': miss, 'freq10': cnt10, 'freq30': cnt30,
            'freqAll': cntAll, 'neigh': neigh, 'lastdig': score_last,
            'total': score_total,
        }
    return scores


def backtest(rows, top_ns=(15,18,20,22,25,28,30,33), window=3, history_window=100):
    """
    对每个 T ≥ window, 用前 window 期构建 freq 池, 然后按评分排序, 取 top N,
    评估覆盖率 (top-N 命中 T 期 20 个号码几个).
    """
    # 维护一个滑动 deque 保留最近 history_window 期
    history = deque(maxlen=history_window)
    recent10 = deque(maxlen=10)
    recent30 = deque(maxlen=30)

    out = []
    for T in range(len(rows)):
        cur = rows[T]
        # 当前期作为 (issue, red_set)
        rec_tuple = (int(cur['issue']), cur['reds'])

        if T < window:
            history.append(rec_tuple)
            recent10.append(rec_tuple)
            recent30.append(rec_tuple)
            continue

        # 计算 freq=1 池
        prev3 = rows[T-window : T]
        freq = Counter()
        for p in prev3:
            freq.update(p['reds'])
        pool = sorted([n for n in range(1, 81) if freq.get(n, 0) == 1])

        if not pool:
            history.append(rec_tuple); recent10.append(rec_tuple); recent30.append(rec_tuple)
            continue

        # 评分
        scores = score_methods_for_pool(
            pool, history, recent10, recent30, list(history)
        )

        # 按 total 排序, 但同样记录 miss 排序、freq 排序, 看不同维度
        ranked_total = sorted(pool, key=lambda x: (-scores[x]['total'], x))
        ranked_miss  = sorted(pool, key=lambda x: (-scores[x]['miss'],  x))
        ranked_freq  = sorted(pool, key=lambda x: (-scores[x]['freq10'], x))
        ranked_neigh = sorted(pool, key=lambda x: (-scores[x]['neigh'], x))

        actual = cur['reds']

        result = {
            'issue': cur['issue'],
            'openTime': cur['openTime'],
            'pool_size': len(pool),
            'pool': pool,
        }

        for label, ranked in (('total', ranked_total),
                              ('miss',  ranked_miss),
                              ('freq',  ranked_freq),
                              ('neigh', ranked_neigh)):
            result[f'ranked_{label}'] = ranked
            for N in top_ns:
                tk = min(N, len(ranked))
                hit = len(actual & set(ranked[:tk]))
                result[f'hit_{label}_top{N}'] = hit

        out.append(result)

        history.append(rec_tuple)
        recent10.append(rec_tuple)
        recent30.append(rec_tuple)

    return out


def main():
    rows = load()
    print(f'>>> rows={len(rows)}, '
          f'first={rows[0]["issue"]} last={rows[-1]["issue"]}')

    recs = backtest(rows)
    print(f'>>> backtest records: {len(recs)}')

    top_ns = (15, 18, 20, 22, 25, 28, 30, 33)
    labels = ('total', 'miss', 'freq', 'neigh')

    # 聚合覆盖率
    print('\n========================')
    print('各打分策略下 Top N 覆盖率 (平均 / 中位 / 最大)')
    print('========================')
    print(f'  {"label":<7} {"Top N":>6} | {"avg%":>6} {"med%":>6} {"max%":>5} {"baseline%":>10} {"hit_mean":>9}')
    pool_sizes = [r['pool_size'] for r in recs]
    pool_avg = statistics.mean(pool_sizes)
    for label in labels:
        for N in top_ns:
            hits = [r[f'hit_{label}_top{N}'] for r in recs]
            avg = statistics.mean(hits)
            med = statistics.median(hits)
            mx  = max(hits)
            # 随机基线: 一个大小为 pool_size 的池 (avg=pool_avg), 抽 N 个, 期望命中 N * 20/80
            base_random = N * 20 / 80
            print(f'  {label:<7} {N:>6} | {avg/20*100:>5.2f}% {med/20*100:>5.2f}% '
                  f'{mx/20*100:>4.1f}%   {base_random/20*100:>5.2f}%   {avg:>6.3f}')

    # 整体: 对 N=20 (即覆盖 20 个号码 = 期望的"全包"水平), 看策略
    print('\n========================')
    print('重点: Top N=20, 看策略/随机倍数')
    print('========================')
    for label in labels:
        hits = [r[f'hit_{label}_top20'] for r in recs]
        avg = statistics.mean(hits)
        ratio = avg / 5.0   # 5 = 20 * 20/80 = 5
        print(f'  {label:<7}: 平均命中 {avg:.3f} / 20 = {avg/20*100:.2f}%, '
              f'策略/随机 = {ratio:.3f}')

    # ----- 写报告 -----
    csv_out = OUT_DIR / 'kl8_freq1_ranking_detail.csv'
    with open(csv_out, 'w', encoding='utf-8', newline='') as f:
        wr = csv.writer(f)
        hdr = ['issue', 'openTime', 'pool_size', 'pool']
        for label in labels:
            hdr.append(f'ranked_{label}')
            for N in top_ns:
                hdr.append(f'hit_{label}_top{N}')
        wr.writerow(hdr)
        for r in recs:
            row = [r['issue'], r['openTime'], r['pool_size'], ' '.join(map(str, r['pool']))]
            for label in labels:
                row.append(' '.join(map(str, r[f'ranked_{label}'])))
                for N in top_ns:
                    row.append(r[f'hit_{label}_top{N}'])
            wr.writerow(row)
    print(f'\n>>> detailed csv -> {csv_out}')

    # ----- markdown -----
    md = []
    md.append('# 快乐8 选号策略 v2: freq=1 池打分排名')
    md.append('')
    md.append(f'> 回测时间: {datetime.now().strftime("%Y-%m-%d %H:%M")}  ')
    md.append(f'> 数据源: `data/快乐8_lottery_data.csv`  ')
    md.append(f'> 总期数: {len(rows)} (issue {rows[0]["issue"]} ~ {rows[-1]["issue"]})  ')
    md.append(f'> 回测期数: {len(recs)}  ')
    md.append('')
    md.append('## 评分维度')
    md.append('')
    md.append('- **miss**: 距上次出现的遗漏期数 (越大分越高, 上限 100)')
    md.append('- **freq**: 加权频率 = (近10 × 3 + 近30 × 2 + 全期 × 1) / 6')
    md.append('- **neigh**: 邻号热度 — 近 3 期里 +/- 1 或 +/- 2 出现过的次数')
    md.append('- **lastdig**: 末日号热度 — 近 3 期里同末位号码出现的总数')
    md.append('- **total** = miss*0.4 + freq*0.35 + neigh*1.5 + last*0.5')
    md.append('')
    md.append('## Top N 覆盖率 (T 期实际 20 红球 ∩ 选出来的 Top N)')
    md.append('')
    md.append(f'候选池平均大小: **{pool_avg:.1f}** 个号码')
    md.append('')
    md.append('| 打分 | Top N | 平均命中 | 覆盖率 | 策略/随机倍数 |')
    md.append('|------|-------|---------|--------|-------------|')
    for label in labels:
        for N in (15, 18, 20, 22, 25, 28, 30, 33):
            hits = [r[f'hit_{label}_top{N}'] for r in recs]
            avg = statistics.mean(hits)
            base = N * 20 / 80
            ratio = avg / base
            md.append(f'| {label} | {N} | {avg:.3f} | {avg/20*100:.2f}% | '
                      f'**{ratio:.3f}** |')
    md.append('')
    md.append('## 重点结论: Top 20 (期望与纯选 20 个号码的随机基线)')
    md.append('')
    md.append('| 策略 | 平均命中 / 20 | 覆盖率 | 倍率 |')
    md.append('|------|--------------|--------|------|')
    for label in labels:
        hits = [r[f'hit_{label}_top20'] for r in recs]
        avg = statistics.mean(hits)
        ratio = avg / 5.0
        verdict = '略优' if ratio > 1.05 else ('略劣' if ratio < 0.95 else '与随机无别')
        md.append(f'| {label} | {avg:.3f} | {avg/20*100:.2f}% | '
                  f'**{ratio:.3f}** ({verdict}) |')
    md.append('')
    md.append('## 实用建议')
    md.append('')
    md.append('候选池 (freq=1) 平均 33.8 个, 选 Top 20 意味着排除约 14 个"低分"号.  ')
    md.append('如果 4 个打分维度的 Top 20 命中率都贴 5.0 (随机基线), 说明: ')
    md.append('')
    md.append('- 在一个**全集随机子集**上, 不管怎么排序, 前 20 命中率期望都是 5.0  ')
    md.append('- 这又是"独立随机事件"的证据 — 历史选择对下一期没有可测的预测力  ')
    md.append('- 想突破, 必须**改变成本模型**: 选 freq=1 ∪ freq=2 池(更大) 看 Top 20 是否能压住')
    md.append('')
    md.append('Lucky 边界: 预测号码 ≠ 中奖保证. **任何统计模型的命中数期望 = 5.0**.  ')
    md.append('如果你想"减少投注数 + 提高期望命中", 只能压缩池子 (如只选 Top 10), '
              '代价是命中数更少. 这是 ROI 取舍, 不是"找中奖号".')

    md_path = OUT_DIR / 'kl8_freq1_ranking_backtest.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))
    print(f'>>> md -> {md_path}')


if __name__ == '__main__':
    main()

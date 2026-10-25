"""
调用 Gemini 3.1 Flash Lite 预测 SSQ 下一期 (2026113)
==============================================

时间: 2026-09-28 22:48 GMT+8
作者: Lucky / MiniMax-M3
用户偏好: 默认模型 gemini-3.1-flash-lite (2026-07-05 设定)

输入: 最近 30 期 SSQ 开奖数据
输出: Gemini 3.1 预测的 Top 4 / Top 6 / Top 12 红球 + Top 3 蓝球 + 推理依据
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

import pandas as pd

ROOT = Path.home() / 'Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/Lottery'

# 加载 API Key
env_file = ROOT / '.Renviron'
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ[k.strip()] = v.strip().strip('"').strip("'")

api_key = os.environ.get('GEMINI_API_KEY')
if not api_key:
    print("❌ GEMINI_API_KEY 未加载!")
    sys.exit(1)
print(f"✅ GEMINI_API_KEY 已加载 (长度: {len(api_key)})")

from google import genai
client = genai.Client(api_key=api_key)


# ============================================================
# 准备数据
# ============================================================
df = pd.read_csv(ROOT / 'data/双色球_lottery_data.csv')
period_col = 'issue' if 'issue' in df.columns else '期号'
red_cols = [f'红球{i}' for i in range(1, 7)]
blue_col = '蓝球'

# 最近 30 期数据
recent_data = []
for i in range(30):
    period = df[period_col].iloc[i]
    reds = df[red_cols].iloc[i].astype(int).tolist()
    blue = int(df[blue_col].iloc[i])
    recent_data.append({
        'period': period,
        'reds': sorted(reds),
        'blue': blue
    })

# 计算近期冷热号 (近 30 期)
hot_count = {}
for n in range(1, 34):
    hot_count[n] = sum(1 for d in recent_data if n in d['reds'])

# 蓝球冷热 (近 30 期)
blue_count = {}
for n in range(1, 17):
    blue_count[n] = sum(1 for d in recent_data if d['blue'] == n)

# 和值 / 跨度统计
sums = [sum(d['reds']) for d in recent_data]
spans = [max(d['reds']) - min(d['reds']) for d in recent_data]

# 找近期热号 (c30 >= 3) 和冷号 (c30 = 0)
hot_nums = sorted([n for n, c in hot_count.items() if c >= 3])
cold_nums = sorted([n for n, c in hot_count.items() if c == 0])
blue_hot = sorted([n for n, c in blue_count.items() if c >= 3], key=lambda x: -blue_count[x])
blue_cold = sorted([n for n, c in blue_count.items() if c <= 1], key=lambda x: blue_count[x])

# 跟上一期的重号 (跟 2026112 重号)
last_reds = set(recent_data[0]['reds'])


# ============================================================
# 构造 prompt
# ============================================================
prompt = f"""你是一个专业的彩票数据分析模型,需要为双色球(SSQ)第 2026113 期提供预测。

# 任务
基于以下最近 30 期开奖数据,结合你看到的统计规律,预测 2026113 期(今晚开奖)。
- 红球: 6 个号, 范围 1-33
- 蓝球: 1 个号, 范围 1-16

# 最近 30 期开奖数据 (从最新到最老)
```
"""
for d in recent_data:
    reds_str = ' '.join(f'{n:02d}' for n in d['reds'])
    prompt += f"{d['period']}  红球 {reds_str}  蓝球 {d['blue']:02d}\n"

prompt += f"""
# 已知统计规律 (基于近 30 期)

## 1. 红球冷热 (近 30 期出现次数, c30)
- 热号 (c30 ≥ 3): {hot_nums} (共 {len(hot_nums)} 个)
- 冷号 (c30 = 0): {cold_nums} (共 {len(cold_nums)} 个)
- 注意: 几乎所有号都是"热号", 这是信号摊薄的特征

## 2. 蓝球冷热 (近 30 期)
- 热号 (≥3 次): {[(n, blue_count[n]) for n in blue_hot]}
- 冷号 (≤1 次): {[(n, blue_count[n]) for n in blue_cold[:8]]}

## 3. 上一期 (2026112) 开奖: {sorted(last_reds)}
- 跟下一期的重号参考: 通常下一期会跟上一期有 1-2 个重号

## 4. 形态近期均值
- 和值均值: {sum(sums)/len(sums):.1f} (历史均值 101)
- 跨度均值: {sum(spans)/len(spans):.1f}
- 上一期和值 74 (偏低, 4 区 3/2/1 偏小号), 下一期可能反弹到正常区间

# 预测要求 (严格按 JSON 格式输出)

请按以下 JSON 格式给出预测, 并给出 100 字以内的推理理由:

```json
{{
  "target_period": "2026113",
  "predicted_at": "{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
  "model": "gemini-3.1-flash-lite",
  "top_4_red": [4 个最看好的红球, 升序],
  "top_6_red": [6 个红球, 升序],
  "top_12_red": [12 个红球, 升序],
  "top_3_blue": [3 个最看好的蓝球, 升序],
  "reasoning": "100 字以内的推理: 你看好的号, 你避开的号, 以及为什么"
}}
```

# 提示
1. 不要追求"全中", 你的目标是"高于随机基线 18.18% 的命中率"
2. 优先选号: 形态均衡 (和值 90-110), 4 区位 1/2/3 或 2/2/2, 跨度 22-30
3. 可以考虑: 上一期的热号延续 + 蓝球极冷号 (10, 13) 反转候选
4. 给出你**真的有信心**的预测, 不要为了凑数乱选
"""


# ============================================================
# 调用 Gemini 3.1 Flash Lite
# ============================================================
print("\n" + "=" * 60)
print("调用 Gemini 3.1 Flash Lite 预测 2026113 期")
print("=" * 60)

response = client.models.generate_content(
    model="gemini-3.1-flash-lite",
    contents=prompt,
    config={
        "response_mime_type": "application/json",
        "temperature": 0.7,
    }
)

result_text = response.text
print(f"\n📥 Gemini 原始回复 ({len(result_text)} 字符):")
print("-" * 60)
print(result_text)
print("-" * 60)


# ============================================================
# 解析 + 校验 JSON
# ============================================================
try:
    prediction = json.loads(result_text)
except Exception as e:
    print(f"\n❌ JSON 解析失败: {e}")
    print("尝试从文本中提取 JSON ...")
    import re
    m = re.search(r'```json\s*(.+?)\s*```', result_text, re.DOTALL)
    if m:
        try:
            prediction = json.loads(m.group(1))
            print("✅ 从 markdown 块中提取成功")
        except Exception as e2:
            print(f"❌ 仍然失败: {e2}")
            sys.exit(1)
    else:
        sys.exit(1)


# 校验字段
def check_red_list(lst, expected_n, name):
    if not isinstance(lst, list):
        print(f"❌ {name} 不是 list: {lst}")
        return False
    if len(lst) != expected_n:
        print(f"❌ {name} 长度 {len(lst)} (预期 {expected_n}): {lst}")
        return False
    if not all(isinstance(n, int) and 1 <= n <= 33 for n in lst):
        print(f"❌ {name} 包含非法号: {lst}")
        return False
    if len(set(lst)) != len(lst):
        print(f"❌ {name} 有重复: {lst}")
        return False
    print(f"✅ {name} 校验通过: {sorted(lst)}")
    return True


print(f"\n{'='*60}")
print("JSON 字段校验")
print("=" * 60)

ok = True
ok &= check_red_list(prediction.get('top_4_red', []), 4, 'top_4_red')
ok &= check_red_list(prediction.get('top_6_red', []), 6, 'top_6_red')
ok &= check_red_list(prediction.get('top_12_red', []), 12, 'top_12_red')

blue_list = prediction.get('top_3_blue', [])
if isinstance(blue_list, list) and len(blue_list) == 3 and all(isinstance(n, int) and 1 <= n <= 16 for n in blue_list):
    print(f"✅ top_3_blue 校验通过: {sorted(blue_list)}")
else:
    print(f"❌ top_3_blue 校验失败: {blue_list}")
    ok = False

if not ok:
    print("\n⚠️ 预测数据有问题, 但仍会保存原始回复")


# ============================================================
# 输出最终预测
# ============================================================
print(f"\n{'='*60}")
print("🎯 Gemini 3.1 Flash Lite 预测结果 (2026113 期)")
print("=" * 60)
print(f"Top 4 红球 (主推 4 胆):  {prediction.get('top_4_red', [])}")
print(f"Top 6 红球 (小复式):     {prediction.get('top_6_red', [])}")
print(f"Top 12 红球 (大复式):    {prediction.get('top_12_red', [])}")
print(f"Top 3 蓝球:              {prediction.get('top_3_blue', [])}")
print(f"\n📝 推理: {prediction.get('reasoning', '(无)')}")


# ============================================================
# 保存预测 JSON
# ============================================================
output = {
    'meta': {
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S GMT+8'),
        'target_period': '2026113',
        'lottery': '双色球',
        'model': 'gemini-3.1-flash-lite',
        'caller': 'Lucky / MiniMax-M3',
        'data_periods': 3508,
        'last_period': '2026112',
        'prompt_strategy': '30 期数据 + 形态统计 + 冷热号特征',
        'response_format': 'json',
    },
    'recent_30_periods': recent_data,
    'prediction': prediction,
}

output_path = ROOT / 'data/backtest/2026113_predictions_gemini31.json'
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\n💾 已保存到: {output_path}")
print(f"   文件大小: {output_path.stat().st_size} 字节")
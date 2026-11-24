#!/usr/bin/env python3
"""
Generate SEO Assets for Lottery Analysis App:
1. sitemap.xml
2. robots.txt
3. static_seo/index.html (Pre-rendered static landing page for Google/Baidu crawlers)
"""

import os
import sys
import datetime
import csv

# Add parent directory to path so config can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LOTTERY_CONFIG
from views.seo_helper import LOTTERY_SEO_INFO

BASE_URL = "https://lottery-analysis.streamlit.app"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static_seo")

def get_latest_lottery_summary():
    """Reads the latest draw result for each lottery from CSV files."""
    summaries = []
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    for name, conf in LOTTERY_CONFIG.items():
        code = conf["code"]
        file_path = os.path.join(base_dir, conf["data_file"])
        summary = {
            "name": name,
            "code": code,
            "issue": "-",
            "date": "-",
            "red_balls": [],
            "blue_balls": [],
            "red_count": conf["red_count"],
            "url": f"{BASE_URL}/?lottery={code}"
        }
        
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8-sig", errors="ignore") as f:
                    reader = csv.DictReader(f)
                    first_row = None
                    for row in reader:
                        first_row = row
                        break
                        
                    if first_row:
                        # Normalize issue & date
                        issue = first_row.get("期号") or first_row.get("issue") or first_row.get("period") or "-"
                        summary["issue"] = str(issue).replace(".0", "").strip()
                        raw_date = str(first_row.get("开奖日期") or first_row.get("openTime") or "-")
                        summary["date"] = raw_date.split(" ")[0].split("T")[0].strip()
                        
                        # Red balls
                        for i in range(1, conf["red_count"] + 1):
                            c = f"{conf['red_col_prefix']}{i}"
                            val = first_row.get(c)
                            if val and val.strip():
                                try:
                                    summary["red_balls"].append(f"{int(float(val)):02d}")
                                except Exception:
                                    summary["red_balls"].append(val.strip())
                        
                        # Blue balls
                        if conf.get("has_blue"):
                            prefix = conf.get("blue_col_name", "蓝球")
                            if conf["blue_count"] == 1:
                                for p in [prefix, "蓝球", "篮球"]:
                                    val = first_row.get(p)
                                    if val and val.strip():
                                        try:
                                            summary["blue_balls"].append(f"{int(float(val)):02d}")
                                        except Exception:
                                            summary["blue_balls"].append(val.strip())
                                        break
                            else:
                                for i in range(1, conf["blue_count"] + 1):
                                    for p in [prefix, "蓝球", "篮球"]:
                                        cname = f"{p}{i}"
                                        val = first_row.get(cname)
                                        if val and val.strip():
                                            try:
                                                summary["blue_balls"].append(f"{int(float(val)):02d}")
                                            except Exception:
                                                summary["blue_balls"].append(val.strip())
                                            break
            except Exception as e:
                print(f"Warning: Failed reading {file_path}: {e}")
                
        summaries.append(summary)
    return summaries

def generate_sitemap(summaries):
    """Generates sitemap.xml."""
    today = datetime.date.today().isoformat()
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
        '  <url>',
        f'    <loc>{BASE_URL}/</loc>',
        f'    <lastmod>{today}</lastmod>',
        '    <changefreq>daily</changefreq>',
        '    <priority>1.0</priority>',
        '  </url>'
    ]
    
    for item in summaries:
        code = item["code"]
        lines.extend([
            '  <url>',
            f'    <loc>{BASE_URL}/?lottery={code}</loc>',
            f'    <lastmod>{today}</lastmod>',
            '    <changefreq>daily</changefreq>',
            '    <priority>0.9</priority>',
            '  </url>'
        ])
        
    lines.append('</urlset>')
    return "\n".join(lines)

def generate_robots():
    """Generates robots.txt."""
    return f"""# robots.txt for {BASE_URL}
User-agent: *
Allow: /
Allow: /?lottery=*

# Specific search engine crawlers
User-agent: Googlebot
Allow: /

User-agent: Baiduspider
Allow: /

User-agent: Bingbot
Allow: /

User-agent: Sogou web spider
Allow: /

User-agent: 360Spider
Allow: /

User-agent: Bytespider
Allow: /

User-agent: YisouSpider
Allow: /

Sitemap: {BASE_URL}/sitemap.xml
"""

def generate_static_landing_html(summaries):
    """Generates an SEO-optimized pre-rendered static HTML landing page."""
    today_str = datetime.date.today().strftime("%Y年%m月%d日")
    
    cards_html = ""
    for item in summaries:
        name = item["name"]
        code = item["code"]
        issue = item["issue"]
        date = item["date"]
        red_html = "".join([f'<span class="ball red">{b}</span>' for b in item["red_balls"]])
        blue_html = "".join([f'<span class="ball blue">{b}</span>' for b in item["blue_balls"]])
        seo_meta = LOTTERY_SEO_INFO.get(name, {})
        desc = seo_meta.get("description", "大数据走势图分析与AI预测推荐。")
        
        cards_html += f"""
        <div class="lottery-card">
            <div class="card-header">
                <h3>{name}</h3>
                <span class="badge">第 {issue} 期</span>
            </div>
            <p class="date">开奖日期：{date}</p>
            <div class="balls-container">
                {red_html}
                {blue_html}
            </div>
            <p class="desc">{desc}</p>
            <div class="card-actions">
                <a href="{BASE_URL}/?lottery={code}" class="btn-primary" target="_blank" rel="noopener">
                    📊 查看{name}走势图与AI预测 &rarr;
                </a>
            </div>
        </div>
        """
        
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>双色球/大乐透/快乐8开奖走势图与AI智能选号预测系统 - 彩票大数据平台</title>
    <meta name="description" content="专业彩票大数据分析与AI预测系统。提供双色球、超级大乐透、快乐8、福彩3D、排列三、排列五、七星彩等官方开奖数据查询、冷热号走势、奇偶连号统计、形态学优选与大模型AI预测。">
    <meta name="keywords" content="双色球走势图, 大乐透走势图, 快乐8开奖结果, 彩票AI预测, 双色球预测, 彩票冷热号, 福彩3D, 排列三, 七星彩, 彩票大数据">
    <meta name="robots" content="index, follow">
    <link rel="canonical" href="{BASE_URL}/">
    
    <!-- Open Graph -->
    <meta property="og:title" content="双色球/大乐透/快乐8开奖走势图与AI智能选号预测系统">
    <meta property="og:description" content="基于大数据与前沿AI大模型的中国福利彩票与体育彩票数据分析平台，涵盖20+项专业走势指标与模型回测。">
    <meta property="og:type" content="website">
    <meta property="og:url" content="{BASE_URL}/">
    
    <!-- JSON-LD Structured Data -->
    <script type="application/ld+json">
    {{
        "@context": "https://schema.org",
        "@type": "WebApplication",
        "name": "彩票大数据分析与AI智能预测系统",
        "url": "{BASE_URL}/",
        "description": "基于大数据与AI大模型的中国福利彩票与体育彩票综合分析平台，涵盖双色球、大乐透、快乐8等主流彩种走势图分析与智能预测。",
        "applicationCategory": "UtilitiesApplication",
        "operatingSystem": "All",
        "offers": {{
            "@type": "Offer",
            "price": "0",
            "priceCurrency": "CNY"
        }}
    }}
    </script>
    
    <style>
        :root {{
            --primary: #1e88e5;
            --primary-dark: #1565c0;
            --red: #e53935;
            --blue: #1e88e5;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --text-main: #1e293b;
            --text-sub: #64748b;
            --border: #e2e8f0;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
            background-color: var(--bg);
            color: var(--text-main);
            line-height: 1.6;
        }}
        .header {{
            background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
            color: #ffffff;
            padding: 50px 20px;
            text-align: center;
        }}
        .header h1 {{ font-size: 2.2rem; margin-bottom: 12px; font-weight: 700; }}
        .header p {{ font-size: 1.1rem; opacity: 0.9; max-width: 800px; margin: 0 auto 20px auto; }}
        .header-btn {{
            display: inline-block;
            background: #22c55e;
            color: #ffffff;
            text-decoration: none;
            padding: 12px 28px;
            font-size: 1.1rem;
            font-weight: 600;
            border-radius: 8px;
            box-shadow: 0 4px 14px rgba(34, 197, 94, 0.4);
            transition: transform 0.2s, background 0.2s;
        }}
        .header-btn:hover {{ background: #16a34a; transform: translateY(-2px); }}
        
        .container {{ max-width: 1200px; margin: 40px auto; padding: 0 20px; }}
        .section-title {{
            font-size: 1.6rem;
            margin-bottom: 24px;
            text-align: center;
            position: relative;
        }}
        .section-title::after {{
            content: '';
            display: block;
            width: 50px;
            height: 4px;
            background: var(--primary);
            margin: 8px auto 0 auto;
            border-radius: 2px;
        }}
        
        .lottery-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 24px;
            margin-bottom: 50px;
        }}
        .lottery-card {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
            border: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .lottery-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        }}
        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }}
        .card-header h3 {{ font-size: 1.3rem; color: #0f172a; }}
        .badge {{
            background: #e0f2fe;
            color: #0369a1;
            padding: 4px 8px;
            border-radius: 6px;
            font-size: 0.85rem;
            font-weight: 600;
        }}
        .date {{ font-size: 0.9rem; color: var(--text-sub); margin-bottom: 14px; }}
        .balls-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-bottom: 16px;
        }}
        .ball {{
            width: 34px;
            height: 34px;
            border-radius: 50%;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 0.95rem;
            color: #ffffff;
        }}
        .ball.red {{ background: var(--red); box-shadow: 0 2px 4px rgba(229, 57, 53, 0.3); }}
        .ball.blue {{ background: var(--blue); box-shadow: 0 2px 4px rgba(30, 136, 229, 0.3); }}
        .desc {{ font-size: 0.92rem; color: var(--text-sub); margin-bottom: 20px; flex-grow: 1; }}
        .card-actions {{ margin-top: auto; }}
        .btn-primary {{
            display: block;
            text-align: center;
            background: var(--primary);
            color: #ffffff;
            text-decoration: none;
            padding: 10px 16px;
            border-radius: 6px;
            font-weight: 600;
            transition: background 0.2s;
        }}
        .btn-primary:hover {{ background: var(--primary-dark); }}
        
        .features-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 20px;
            margin-bottom: 50px;
        }}
        .feature-box {{
            background: #ffffff;
            border: 1px solid var(--border);
            padding: 20px;
            border-radius: 8px;
        }}
        .feature-box h4 {{ margin-bottom: 8px; color: var(--primary-dark); font-size: 1.1rem; }}
        .feature-box p {{ font-size: 0.92rem; color: var(--text-sub); }}
        
        .faq-section {{
            background: #ffffff;
            border-radius: 12px;
            padding: 30px;
            border: 1px solid var(--border);
            margin-bottom: 50px;
        }}
        .faq-item {{ margin-bottom: 20px; }}
        .faq-item h4 {{ font-size: 1.05rem; margin-bottom: 6px; color: #0f172a; }}
        .faq-item p {{ font-size: 0.95rem; color: #475569; }}
        
        .footer {{
            background: #0f172a;
            color: #94a3b8;
            padding: 40px 20px;
            text-align: center;
            font-size: 0.9rem;
        }}
        .footer p {{ margin-bottom: 8px; }}
    </style>
</head>
<body>

    <header class="header">
        <h1>📊 彩票大数据分析与AI智能预测系统</h1>
        <p>基于官方真实开奖数据，提供双色球、超级大乐透、快乐8、福彩3D等主流彩种的深度走势图表、冷热规律统计与多大模型辅助选号。</p>
        <a href="{BASE_URL}/" class="header-btn" target="_blank" rel="noopener">🚀 立即打开分析系统 (Streamlit App)</a>
    </header>

    <main class="container">
        <h2 class="section-title">主流彩种最新开奖与趋势直达 ({today_str})</h2>
        <div class="lottery-grid">
            {cards_html}
        </div>

        <h2 class="section-title">核心功能与技术特色</h2>
        <div class="features-grid">
            <div class="feature-box">
                <h4>📈 20+ 项专业技术指标</h4>
                <p>涵盖红蓝球冷热频次、奇偶比、大小比、连号跳号、尾数分布、区间分布、和值与跨度趋势分析。</p>
            </div>
            <div class="feature-box">
                <h4>🤖 多大模型 AI 预测</h4>
                <p>集成 Google Gemini 2.0、阿里通义千问 Qwen、MiniMax 等前沿 LLM，多期历史数据驱动概率推演。</p>
            </div>
            <div class="feature-box">
                <h4>🎯 统计形态学约束</h4>
                <p>根据和值分布、跨度极值、重号与连号规则，智能排除极端不合理号码组合，科学缩水。</p>
            </div>
            <div class="feature-box">
                <h4>🔙 全流程历史回测</h4>
                <p>每期 AI 预测均自动归档入库，与真实开奖结果自动化比对，准确率与命中情况真实可见。</p>
            </div>
        </div>

        <div class="faq-section">
            <h2 class="section-title" style="margin-bottom: 24px;">常见问题解答 (FAQ)</h2>
            <div class="faq-item">
                <h4>Q: 走势图中的“冷号”和“热号”该如何参考？</h4>
                <p>A: 热号代表近期出现频率高、活跃度强的号码；冷号代表长期遗漏未出的号码。根据大数定律与均值回归，极端大遗漏号码存在回补概率，但单期开奖仍为独立事件，建议结合区间均衡与奇偶形态综合判断。</p>
            </div>
            <div class="faq-item">
                <h4>Q: AI 大模型是如何预测彩票的？</h4>
                <p>A: 平台将多期历史开奖特征、冷热走势、形态学约束作为结构化 Prompt 输入大语言模型，利用模型强大的逻辑推理与模式识别能力生成候选推荐，再经过形态学规则过滤生成最终选号。</p>
            </div>
            <div class="faq-item">
                <h4>Q: 预测结果保证中奖吗？</h4>
                <p>A: 不保证。彩票开奖是典型的物理随机过程，任何算法、数学模型或 AI 工具都无法做到 100% 预测。平台旨在提供科学严谨的数据统计与规律分析，请彩民朋友务必理性购彩、量力而行。</p>
            </div>
        </div>
    </main>

    <footer class="footer">
        <p><strong>彩票大数据分析与AI预测系统</strong> | 开奖数据实时同步自官方公开信息</p>
        <p>免责声明：本站所有数据及预测结果仅供数学建模、计算机编程及数据挖掘爱好者研究交流使用，不构成任何购彩建议。</p>
        <p>© {datetime.date.today().year} Lottery BigData Analysis. All rights reserved.</p>
    </footer>

</body>
</html>
"""
    return html

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summaries = get_latest_lottery_summary()
    
    # 1. Generate sitemap.xml
    sitemap_content = generate_sitemap(summaries)
    sitemap_path = os.path.join(OUTPUT_DIR, "sitemap.xml")
    with open(sitemap_path, "w", encoding="utf-8") as f:
        f.write(sitemap_content)
    print(f"[OK] Generated: {sitemap_path}")
    
    # Also save a copy in root if desired
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root_dir, "sitemap.xml"), "w", encoding="utf-8") as f:
        f.write(sitemap_content)
    print(f"[OK] Generated: {os.path.join(root_dir, 'sitemap.xml')}")
        
    # 2. Generate robots.txt
    robots_content = generate_robots()
    robots_path = os.path.join(OUTPUT_DIR, "robots.txt")
    with open(robots_path, "w", encoding="utf-8") as f:
        f.write(robots_content)
    print(f"[OK] Generated: {robots_path}")
    
    with open(os.path.join(root_dir, "robots.txt"), "w", encoding="utf-8") as f:
        f.write(robots_content)
    print(f"[OK] Generated: {os.path.join(root_dir, 'robots.txt')}")
        
    # 3. Generate static index.html
    html_content = generate_static_landing_html(summaries)
    html_path = os.path.join(OUTPUT_DIR, "index.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[OK] Generated: {html_path}")

if __name__ == "__main__":
    main()

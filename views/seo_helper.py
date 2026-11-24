import json
from typing import Dict, Any

# SEO Information configurations for Chinese Lotteries
LOTTERY_SEO_INFO = {
    "双色球": {
        "title": "双色球走势图_双色球AI智能预测与开奖结果分析 - 彩票大数据系统",
        "keywords": "双色球走势图, 双色球预测, 双色球开奖结果, 双色球AI选号, 双色球冷热号, 双色球奇偶比, 中国福利彩票",
        "description": "专业双色球大数据分析与AI预测平台，提供最新双色球开奖结果查询、历史走势图表（红球冷热、蓝球遗漏、奇偶比、连号跳号、和值跨度）、形态学优选及多大模型AI智能预测。",
        "faq": [
            ("双色球走势图有哪些核心指标？", "主要包含红球出现频次、冷热遗漏值、奇偶比、大小比、质合比、三区分布、连号与跳号、和值走势与AC值等，帮助彩民多维度捕捉历史号码分布规律。"),
            ("平台AI预测双色球的原理是什么？", "本系统融合了统计学模型、形态学多维约束过滤规则（和值/跨度/重号/同尾过滤），并结合前沿大语言模型（Gemini、通义千问、MiniMax等）对高频与遗漏特征进行综合推理与打分。")
        ]
    },
    "超级大乐透": {
        "title": "超级大乐透走势图_大乐透AI预测与中奖查询 - 彩票大数据系统",
        "keywords": "超级大乐透走势图, 大乐透预测, 大乐透开奖结果, 大乐透AI选号, 大乐透冷热号, 前区后区走势, 体彩大乐透",
        "description": "提供体彩超级大乐透最新开奖数据查询与历史走势图分析，包括前区35选5冷热号、后区12选2分布、和值跨度趋势、AI大模型预测推荐与多期回测报告。",
        "faq": [
            ("超级大乐透前区与后区的号码特征？", "前区从01-35中选择5个号码，后区从01-12中选择2个号码。走势分析中重点关注前区和值区间（通常65-115）及后区奇偶搭配。"),
            ("大乐透AI预测如何进行历史验证？", "平台内置自动回测系统，记录每期AI推荐号码与实际开奖的命中情况（红球/蓝球命中数），保证预测过程透明、有据可查。")
        ]
    },
    "快乐8": {
        "title": "快乐8走势图_快乐8开奖结果查询与AI选十选九预测 - 彩票大数据系统",
        "keywords": "快乐8走势图, 快乐8开奖结果, 快乐8预测, 快乐8选十推荐, 快乐8选五, 快乐8冷热号, 快乐8和值趋势",
        "description": "快乐8大数据分析平台，支持全80码开奖走势图分析、选一至选十中奖详情、区间出号分布、连号聚集性统计及AI大模型智能选号推荐。",
        "faq": [
            ("快乐8的常见中奖玩法有哪些？", "快乐8包含选一至选十共10种玩法。其中选十、选九奖金丰厚，选五、选四返奖概率高，走势图重点关注出号密度与区间均衡性。"),
            ("如何利用大数据筛选快乐8号码？", "通过80个号码的冷热出号周期、连号聚集度、同尾数分布及和值均值（通常在680-940之间）进行多重形态过滤。")
        ]
    },
    "福彩3D": {
        "title": "福彩3D走势图_3D开奖结果查询与直选组选预测 - 彩票大数据系统",
        "keywords": "福彩3D走势图, 福彩3D预测, 3D开奖结果, 3D直选组选, 3D百十个位走势, 3D和值跨度",
        "description": "福利彩票福彩3D专业走势分析，百十个位数字走势、0-9冷热分布、和值振幅、奇偶大小形态、直选与组选AI预测推荐。",
        "faq": [
            ("福彩3D如何判断组三与组六形态？", "3个开奖号码中有2个相同为组三，3个各不相同为组六。可通过历史遗漏期数与连出周期图表辅助判断。")
        ]
    },
    "排列三": {
        "title": "体彩排列三走势图_排列3开奖结果与AI预测 - 彩票大数据系统",
        "keywords": "排列三走势图, 排列3预测, 排列3开奖结果, 体彩排列三, 排列3和值走势, 直选组选选号",
        "description": "体育彩票排列三（排列3）大数据走势图，涵盖百十个位定位走势、奇偶大小比、和值跨度走势图、冷热号与AI大模型预测推荐。",
        "faq": [
            ("排列三和值走势对选号有什么参考价值？", "排列三和值范围为0-27，高频黄金和值通常集中在10-18之间，结合跨度指标可大幅缩减候选注数。")
        ]
    },
    "排列五": {
        "title": "体彩排列五走势图_排列5开奖结果与号码走势分析 - 彩票大数据系统",
        "keywords": "排列五走势图, 排列5预测, 排列5开奖结果, 体彩排列五, 万千百十个位定位走势",
        "description": "排列五（排列5）专业开奖走势分析，提供万位、千位、百位、十位、个位五位数字全量历史走势、冷热频次统计、奇偶大小与AI号码推荐。",
        "faq": [
            ("排列五选号的核心技巧是什么？", "重点分析五位定位出号的冷热振幅、重号延续率与和值跨度形态，运用形态学过滤不合理的极端组合。")
        ]
    },
    "七乐彩": {
        "title": "七乐彩走势图_七乐彩开奖结果与中奖号码预测 - 彩票大数据系统",
        "keywords": "七乐彩走势图, 七乐彩开奖结果, 七乐彩预测, 七乐彩冷热号, 30选7走势",
        "description": "中国福利彩票七乐彩数据分析系统，30选7号码走势图、特别号码分布、和值跨度、连号跳号统计与AI智能选号助手。",
        "faq": [
            ("七乐彩的选号区间与号码特征？", "从01-30共30个号码中选出7个基本号码和1个特别号码，重点分析连号形态（通常包含1-2组连号）及三区比例。")
        ]
    },
    "七星彩": {
        "title": "体彩七星彩走势图_新七星彩开奖结果与预测 - 彩票大数据系统",
        "keywords": "七星彩走势图, 七星彩开奖结果, 七星彩预测, 新七星彩后区走势, 前六位定位",
        "description": "体育彩票七星彩大数据系统，前六位0-9数字定位走势、第七位0-14特码分布、和值跨度趋势、AI大模型预测与开奖结果查询。",
        "faq": [
            ("新七星彩的规则特点是什么？", "前六位从000000-999999中产生，第七位从0-14中产生，前区重点看奇偶大小比，后区重点看冷温热出号周期。")
        ]
    }
}

DEFAULT_SEO = {
    "title": "彩票大数据分析与AI智能预测系统 - 双色球/大乐透/快乐8开奖走势图",
    "keywords": "彩票走势图, 双色球走势图, 大乐透走势图, 快乐8开奖结果, 彩票AI预测, 福利彩票, 体育彩票, 彩票冷热号, 大模型选号",
    "description": "基于大数据与AI大模型的中国福利彩票与体育彩票综合分析平台。涵盖双色球、大乐透、快乐8、福彩3D、排列三/五、七乐彩、七星彩等核心彩种走势图分析、冷热号挖掘、形态学过滤与多模型预测推荐。",
    "faq": []
}

def get_seo_meta(lottery_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieve SEO metadata for the current lottery."""
    info = LOTTERY_SEO_INFO.get(lottery_name, DEFAULT_SEO)
    code = config.get("code", "ssq")
    canonical_url = f"https://lottery-analysis.streamlit.app/?lottery={code}"
    
    schema_org = {
        "@context": "https://schema.org",
        "@type": "WebApplication",
        "name": info["title"],
        "url": canonical_url,
        "applicationCategory": "UtilitiesApplication",
        "operatingSystem": "All",
        "description": info["description"],
        "keywords": info["keywords"],
        "offers": {
            "@type": "Offer",
            "price": "0",
            "priceCurrency": "CNY"
        },
        "featureList": [
            "实时开奖数据自动采集与校验",
            "20+项核心技术指标趋势图表分析",
            "大语言模型 (Gemini, Qwen, MiniMax) 智能辅助选号",
            "历史预测自动回测与命中率统计",
            "多维形态学（和值/跨度/同尾/连号）组合过滤"
        ]
    }
    
    faq_schema = None
    if info.get("faq"):
        faq_schema = {
            "@context": "https://schema.org",
            "@type": "FAQPage",
            "mainEntity": [
                {
                    "@type": "Question",
                    "name": q,
                    "acceptedAnswer": {
                        "@type": "Answer",
                        "text": a
                    }
                }
                for q, a in info["faq"]
            ]
        }

    return {
        "title": info["title"],
        "keywords": info["keywords"],
        "description": info["description"],
        "canonical_url": canonical_url,
        "faq": info.get("faq", []),
        "schema_org": schema_org,
        "faq_schema": faq_schema
    }

def inject_seo_meta(lottery_name: str, config: Dict[str, Any]):
    """Injects meta tags, title, OpenGraph tags and JSON-LD structured data into the host head."""
    import streamlit as st
    meta = get_seo_meta(lottery_name, config)
    title_escaped = meta["title"].replace('"', '\\"')
    desc_escaped = meta["description"].replace('"', '\\"')
    keywords_escaped = meta["keywords"].replace('"', '\\"')
    canonical = meta["canonical_url"]
    schema_json = json.dumps(meta["schema_org"], ensure_ascii=False)
    faq_json = json.dumps(meta["faq_schema"], ensure_ascii=False) if meta.get("faq_schema") else "{}"

    injection_code = f"""
    <script>
    (function() {{
        try {{
            var doc = window.parent.document;
            if (!doc) doc = document;
            
            // 1. Update Title
            doc.title = "{title_escaped}";
            
            // Helper to set or create meta tag
            function setMeta(nameAttr, nameVal, content) {{
                var el = doc.querySelector('meta[' + nameAttr + '="' + nameVal + '"]');
                if (!el) {{
                    el = doc.createElement('meta');
                    el.setAttribute(nameAttr, nameVal);
                    doc.head.appendChild(el);
                }}
                el.setAttribute('content', content);
            }}
            
            // 2. Standard SEO Meta
            setMeta('name', 'description', "{desc_escaped}");
            setMeta('name', 'keywords', "{keywords_escaped}");
            setMeta('name', 'author', 'Lottery BigData AI Platform');
            setMeta('name', 'robots', 'index, follow, max-snippet:-1, max-image-preview:large');
            
            // 3. Open Graph (for Social Sharing & WeChat/Preview cards)
            setMeta('property', 'og:title', "{title_escaped}");
            setMeta('property', 'og:description', "{desc_escaped}");
            setMeta('property', 'og:type', 'website');
            setMeta('property', 'og:url', "{canonical}");
            setMeta('property', 'og:site_name', '彩票大数据分析系统');
            setMeta('name', 'twitter:card', 'summary_large_image');
            setMeta('name', 'twitter:title', "{title_escaped}");
            setMeta('name', 'twitter:description', "{desc_escaped}");
            
            // 4. Canonical link
            var canonicalLink = doc.querySelector('link[rel="canonical"]');
            if (!canonicalLink) {{
                canonicalLink = doc.createElement('link');
                canonicalLink.setAttribute('rel', 'canonical');
                doc.head.appendChild(canonicalLink);
            }}
            canonicalLink.setAttribute('href', "{canonical}");
            
            // 5. Schema.org JSON-LD structured data
            var existingSchema = doc.getElementById('lottery-schema-jsonld');
            if (!existingSchema) {{
                existingSchema = doc.createElement('script');
                existingSchema.id = 'lottery-schema-jsonld';
                existingSchema.type = 'application/ld+json';
                doc.head.appendChild(existingSchema);
            }}
            existingSchema.textContent = JSON.stringify({schema_json});
            
            // 6. FAQ JSON-LD
            var faqData = {faq_json};
            if (faqData && faqData['@type']) {{
                var existingFaq = doc.getElementById('lottery-faq-jsonld');
                if (!existingFaq) {{
                    existingFaq = doc.createElement('script');
                    existingFaq.id = 'lottery-faq-jsonld';
                    existingFaq.type = 'application/ld+json';
                    doc.head.appendChild(existingFaq);
                }}
                existingFaq.textContent = JSON.stringify(faqData);
            }}

            // 7. Google Analytics 4 (GA4) Tracking
            var GA4_ID = "G-VDRN56C948";
            var win = window.parent || window;
            if (!doc.getElementById('google-ga4-script')) {{
                var gaScript = doc.createElement('script');
                gaScript.id = 'google-ga4-script';
                gaScript.async = true;
                gaScript.src = 'https://www.googletagmanager.com/gtag/js?id=' + GA4_ID;
                doc.head.appendChild(gaScript);

                win.dataLayer = win.dataLayer || [];
                win.gtag = win.gtag || function() {{ win.dataLayer.push(arguments); }};
                win.gtag('js', new Date());
                win.gtag('config', GA4_ID, {{
                    'page_title': "{title_escaped}",
                    'page_location': "{canonical}"
                }});
            }} else {{
                // When switching lottery tabs/pages, report page_view event to GA4
                if (win.gtag) {{
                    win.gtag('event', 'page_view', {{
                        'page_title': "{title_escaped}",
                        'page_location': "{canonical}"
                    }});
                }}
            }}
        }} catch (e) {{
            console.warn('SEO Meta injection error: ', e);
        }}
    }})();
    </script>
    """
    st.components.v1.html(injection_code, height=0, width=0)

def render_share_and_stats(df, config: Dict[str, Any]):
    """Renders a crawlable semantic summary bar and 1-click sharing tool."""
    import streamlit as st
    import pandas as pd
    if df.empty:
        return
    
    code = config.get("code", "ssq")
    name = config.get("name", "彩票")
    share_url = f"https://lottery-analysis.streamlit.app/?lottery={code}"
    
    row = df.iloc[0]
    issue = str(row.get('期号', ''))
    date_str = str(row.get('开奖日期', '')).split(' ')[0].split('T')[0]
    
    # Extract red balls
    red_balls = []
    for i in range(1, config['red_count'] + 1):
        col = f"{config['red_col_prefix']}{i}"
        if col in df.columns and pd.notna(row[col]):
            red_balls.append(f"{int(row[col]):02d}")
    
    # Extract blue balls
    blue_balls = []
    if config.get('has_blue'):
        prefix = config.get('blue_col_name', '蓝球')
        if config['blue_count'] == 1:
            for p in [prefix, '蓝球', '篮球']:
                if p in df.columns and pd.notna(row[p]):
                    blue_balls.append(f"{int(row[p]):02d}")
                    break
        else:
            for i in range(1, config['blue_count'] + 1):
                for p in [prefix, '蓝球', '篮球']:
                    cname = f"{p}{i}"
                    if cname in df.columns and pd.notna(row[cname]):
                        blue_balls.append(f"{int(row[cname]):02d}")
                        break

    red_str = " ".join(red_balls)
    blue_str = f" + {' '.join(blue_balls)}" if blue_balls else ""
    summary_text = f"【{name}】第 {issue} 期（{date_str}）开奖号码：{red_str}{blue_str}"

    col_info, col_action = st.columns([3, 1])
    with col_info:
        st.caption(f"📌 **网页专属链接**：`{share_url}` | 欢迎收藏与分享本页走势图")
    with col_action:
        # Copy link helper
        copy_script = f"""
        <button onclick="navigator.clipboard.writeText('{share_url}').then(() => alert('已成功复制【{name}】专属直达链接！\\n{share_url}')).catch(() => prompt('请复制链接：', '{share_url}'))"
            style="background-color: #f0f2f6; border: 1px solid #dcdfe6; border-radius: 6px; padding: 4px 12px; font-size: 13px; cursor: pointer; color: #333; float: right;">
            📋 复制本页链接
        </button>
        """
        st.components.v1.html(copy_script, height=35)

def render_seo_guide_and_faq(config: Dict[str, Any]):
    """Renders semantic, keyword-dense guide & FAQ section for search crawlers and users."""
    import streamlit as st
    name = config.get("name", "彩票")
    meta = LOTTERY_SEO_INFO.get(name, DEFAULT_SEO)
    faqs = meta.get("faq", [])
    
    with st.expander(f"📚 {name}数据分析指南、核心选号指标与常见问题 (FAQ)", expanded=False):
        st.markdown(f"""
        ### 🔍 为什么需要借助大数据与趋势图表分析【{name}】？
        彩票本质属于独立随机事件，但长期海量开奖数据在统计学上呈现出明显的**均值回归**、**区间分布规律**与**极值收敛特性**。
        本平台依托中国福利彩票与体育彩票真实历史数据库，提供超过 20 项专业走势指标，帮助研究人员与数据爱好者告别盲目选号：
        
        1. **冷温热号码分布**：实时监控各球号的历史出现频次与遗漏期数，捕捉号码反弹与连出周期。
        2. **形态学多维约束**：计算和值区间、跨度范围、奇偶比（如3:3或2:4）、大小比、同尾号规律，过滤违背统计常理的极端组合。
        3. **关联伴随矩阵**：挖掘高频伴随红球对（Hot Pairs）与三连组合（Hot Triples），辅助组合优化。
        4. **AI 智能大模型预测**：集成 Gemini 2.0、阿里通义千问 Qwen、MiniMax 等前沿 LLM，输入多期历史规律进行多模态概率推演。
        """)

        if faqs:
            st.markdown("#### ❓ 常见问题解答 (Frequently Asked Questions)")
            for q, a in faqs:
                st.markdown(f"**Q: {q}**")
                st.markdown(f"A: {a}")
                st.markdown("---")
        
        st.markdown("""
        > ⚠️ **理性购彩倡议与法律声明**：
        > 
        > 本系统所有数据均来源于官方公开开奖结果，统计分析与 AI 预测结果仅供数学建模、数据科学研究及技术交流参考，不构成任何投资或购买建议。彩票开奖具有完全随机性，任何模型均无法保证 100% 预测命中。请广大彩民保持良好心态，量力而行，理性对待。
        """)

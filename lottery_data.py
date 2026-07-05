"""
lottery_data.py — Lucky 数据访问层

封装对 Lottery 项目 CSV 和 config.py 的访问,提供:
- load(lottery) -> (df, config): 加载数据+配置
- freshness_summary() -> list[dict]: 8 彩种数据状态
- statistical_summary(lottery, n) -> dict: 近 n 期统计概览
"""

from __future__ import annotations

import sys
import importlib.util
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


class LotteryData:
    """Lucky 的数据访问门面。

    使用示例:
        ld = LotteryData(Path("/path/to/Lottery"))
        df, conf = ld.load("双色球")
        summary = ld.statistical_summary("双色球", 30)
    """

    def __init__(self, lottery_root: Path):
        self.root = Path(lottery_root)
        if not self.root.exists():
            raise FileNotFoundError(f"Lottery 项目根目录不存在: {self.root}")

        # 动态加载项目的 config.py
        self.config = self._load_config()
        self.data_dir = self.root / "data"

    # ---------- 配置加载 ----------

    def _load_config(self) -> dict:
        """动态 import 项目的 config.py 获取 LOTTERY_CONFIG。"""
        config_path = self.root / "config.py"
        if not config_path.exists():
            raise FileNotFoundError(f"找不到 config.py: {config_path}")

        spec = importlib.util.spec_from_file_location("lottery_config", config_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["lottery_config"] = module
        spec.loader.exec_module(module)
        return module.LOTTERY_CONFIG

    # ---------- 数据加载 ----------

    def load(self, lottery_name: str) -> tuple[pd.DataFrame | None, dict | None]:
        """加载指定彩种的历史数据 + 配置。

        Returns:
            (df, conf) — df 按日期降序, 包含期号、开奖日期、红蓝球等列
            任一失败时返回 (None, None)
        """
        conf = self.config.get(lottery_name)
        if not conf:
            print(f"❌ 未知彩种: {lottery_name}")
            print(f"   可用: {list(self.config.keys())}")
            return None, None

        csv_path = self.root / conf["data_file"]
        if not csv_path.exists():
            print(f"❌ 数据文件不存在: {csv_path}")
            return None, None

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"❌ 读取 CSV 失败: {e}")
            return None, None

        # 列名归一化 (项目里 issue/openTime 等英文列需映射)
        rename_map = {
            "issue": "期号",
            "openTime": "开奖日期",
            "period": "期号",
        }
        rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
        if rename_map:
            df = df.rename(columns=rename_map)

        # 期号清理 + 排序
        if "期号" in df.columns:
            df["期号"] = df["期号"].astype(str).str.replace(r"\.0$", "", regex=True)
            if "开奖日期" in df.columns:
                df["开奖日期"] = pd.to_datetime(df["开奖日期"], errors="coerce")
                df = df.sort_values(
                    ["开奖日期", "期号"], ascending=[False, False]
                ).reset_index(drop=True)
            else:
                df = df.sort_values("期号", ascending=False).reset_index(drop=True)

        return df, conf

    # ---------- 数据新鲜度 ----------

    def freshness_summary(self) -> list[dict[str, Any]]:
        """扫描 8 个彩种,返回数据新鲜度报告。"""
        rows = []
        for name, conf in self.config.items():
            csv_path = self.root / conf["data_file"]
            row = {
                "name": name,
                "code": conf["code"],
                "exists": csv_path.exists(),
            }
            if not csv_path.exists():
                row.update({"rows": 0, "latest_issue": "—", "mtime_human": "—",
                            "fresh": False})
                rows.append(row)
                continue

            try:
                # 仅读必要列,节省内存
                df = pd.read_csv(csv_path, usecols=lambda c: c in {"issue", "期号"})
                rows_count = len(df)
                if "issue" in df.columns:
                    latest = str(df["issue"].iloc[0]) if rows_count else "—"
                elif "期号" in df.columns:
                    latest = str(df["期号"].iloc[0]) if rows_count else "—"
                else:
                    latest = "—"
            except Exception:
                rows_count = 0
                latest = "—"

            mtime = datetime.fromtimestamp(csv_path.stat().st_mtime)
            age_hours = (datetime.now() - mtime).total_seconds() / 3600
            mtime_human = mtime.strftime("%Y-%m-%d %H:%M")

            row.update({
                "rows": rows_count,
                "latest_issue": latest,
                "mtime_human": mtime_human,
                "age_hours": round(age_hours, 1),
                "fresh": age_hours < 24,
            })
            rows.append(row)
        return rows

    # ---------- 统计分析 ----------

    def statistical_summary(self, lottery_name: str, n: int = 30) -> dict | None:
        """计算近 n 期的统计概览。

        Returns:
            dict with keys: latest_issue, latest_date, total_rows, window,
                            reds {hot_5, cold_5}, blues {hot_5, cold_5},
                            morphology {sum_avg, span_avg, ac_avg, ...}
            失败返回 None
        """
        df, conf = self.load(lottery_name)
        if df is None:
            return None

        # 取近 n 期
        recent = df.head(n).copy()
        if len(recent) < n:
            print(f"⚠️  {lottery_name} 数据只有 {len(recent)} 期,少于请求的 {n}")

        # 红球列
        red_count = conf["red_count"]
        red_cols = [f"红球{i}" for i in range(1, red_count + 1)]

        # 蓝球列
        blue_cols = []
        if conf.get("has_blue"):
            blue_count = conf.get("blue_count", 1)
            base = conf.get("blue_col_name", "蓝球")
            if blue_count == 1:
                blue_cols = [base] if base in recent.columns else []
                if not blue_cols and "蓝球" in recent.columns:
                    blue_cols = ["蓝球"]
            else:
                blue_cols = [f"{base}{i}" for i in range(1, blue_count + 1)]

        # 计算号码出现频次
        def freq_top_bottom(cols: list[str], top_k: int = 5) -> dict:
            if not cols or not all(c in recent.columns for c in cols):
                return {"hot_5": [], "cold_5": []}
            nums = recent[cols].values.flatten()
            nums = pd.Series(nums).dropna().astype(int)
            freq = nums.value_counts()
            if freq.empty:
                return {"hot_5": [], "cold_5": []}
            hot = freq.head(top_k).index.tolist()
            cold = freq.tail(top_k).sort_index().index.tolist()
            return {"hot_5": hot, "cold_5": cold}

        reds_stats = freq_top_bottom(red_cols)
        blues_stats = freq_top_bottom(blue_cols) if blue_cols else None

        # 形态指标 (使用项目预计算的列)
        morphology = {}
        for col in ["和值", "跨度", "AC", "奇数", "偶数", "重号", "邻号"]:
            if col in recent.columns:
                try:
                    morphology[f"{col}(均值)"] = round(float(recent[col].mean()), 2)
                except (TypeError, ValueError):
                    pass

        return {
            "latest_issue": str(df["期号"].iloc[0]) if "期号" in df.columns else "—",
            "latest_date": str(df["开奖日期"].iloc[0])[:10] if "开奖日期" in df.columns else "—",
            "total_rows": len(df),
            "window": len(recent),
            "reds": reds_stats,
            "blues": blues_stats,
            "morphology": morphology,
        }


# 让模块可独立测试
if __name__ == "__main__":
    import json

    LOTTERY_ROOT = (Path.home() /
                    "Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/Lottery")
    ld = LotteryData(LOTTERY_ROOT)

    print("=" * 60)
    print("数据新鲜度:")
    print("=" * 60)
    for row in ld.freshness_summary():
        print(f"  {row['name']:<8} {row['latest_issue']:<8} "
              f"{row['mtime_human']:<16} {'✅' if row['fresh'] else '⚠️'}")

    print("\n" + "=" * 60)
    print("双色球近 30 期统计:")
    print("=" * 60)
    summary = ld.statistical_summary("双色球", 30)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
# Shared Configuration for Lotteries

LOTTERY_CONFIG = {
    "双色球": {
        "code": "ssq",
        "data_file": "data/双色球_lottery_data.csv",
        "has_blue": True,
        "red_col_prefix": "红球",
        "blue_col_name": "蓝球",
        "red_count": 6,
        "blue_count": 1,
        "red_range": (1, 33),
        "blue_range": (1, 16),
        "eval_metrics": {"top_n_1": 6, "top_n_2": 10, "green_threshold": 3, "red_threshold": 4},
        "morphology_rules": {
            "sum_range": (75, 135),
            "span_range": (18, 31),
            "ideal_repeats": (0, 2),
            "ideal_consecutive": [0, 1],
            "ideal_odd_counts": [2, 3, 4],
            "ideal_same_tails": [0, 1, 2] # 增加 0 组同尾 (24%)
        },
        "supported_charts": [
            "red_freq", "blue_freq", "odd_even_ratio", "odd_even_trend",
            "consecutive_dist", "consecutive_trend", "jump_dist", "jump_trend",
            "tail_dist", "tail_trend", "size_ratio", "size_trend",
            "zone_dist", "zone_trend", "repeat_dist", "repeat_trend",
            "neighbor_dist", "neighbor_trend", "isolated_dist", "isolated_trend",
            "sum_trend", "span_trend", "ac_trend", "hot_pairs", "hot_triples"
        ]
    },
    "超级大乐透": {
        "code": "dlt",
        "data_file": "data/超级大乐透_lottery_data.csv",
        "has_blue": True,
        "red_col_prefix": "红球",
        "blue_col_name": "蓝球",
        "red_count": 5,
        "blue_count": 2,
        "red_range": (1, 35),
        "blue_range": (1, 12),
        "eval_metrics": {"top_n_1": 5, "top_n_2": 10, "green_threshold": 3, "red_threshold": 4},
        "morphology_rules": {
            "sum_range": (65, 115),
            "span_range": (18, 33),
            "ideal_repeats": (0, 1),
            "ideal_consecutive": [0, 1],
            "ideal_odd_counts": [1, 2, 3, 4], # 覆盖 95% 奇偶比
            "ideal_same_tails": [0, 1] # 覆盖 92% 同尾情况
        },
        "supported_charts": ["red_freq", "blue_freq", "odd_even_ratio", "odd_even_trend",
            "consecutive_dist", "consecutive_trend", "jump_dist", "jump_trend",
            "tail_dist", "tail_trend", "size_ratio", "size_trend",
            "zone_dist", "zone_trend", "repeat_dist", "repeat_trend",
            "neighbor_dist", "neighbor_trend", "isolated_dist", "isolated_trend",
            "sum_trend", "span_trend", "ac_trend", "hot_pairs", "hot_triples"]
    },
    "七乐彩": {
        "code": "qlc",
        "data_file": "data/七乐彩_lottery_data.csv",
        "has_blue": True,
        "red_col_prefix": "红球",
        "blue_col_name": "篮球",
        "red_count": 7,
        "blue_count": 1,
        "red_range": (1, 30),
        "blue_range": (1, 30),
        "eval_metrics": {"top_n_1": 7, "top_n_2": 12, "green_threshold": 4, "red_threshold": 5},
        "morphology_rules": {
            "sum_range": (80, 140),
            "span_range": (16, 29),
            "ideal_repeats": (1, 2),
            "ideal_consecutive": [0, 1, 2],
            "ideal_odd_counts": [3, 4],
            "ideal_same_tails": [1, 2]
        },
        "supported_charts": [ "red_freq", "blue_freq", "odd_even_ratio", "odd_even_trend",
            "consecutive_dist", "consecutive_trend", "jump_dist", "jump_trend",
            "tail_dist", "tail_trend", "size_ratio", "size_trend",
            "zone_dist", "zone_trend", "repeat_dist", "repeat_trend",
            "neighbor_dist", "neighbor_trend", "isolated_dist", "isolated_trend",
            "sum_trend", "span_trend", "ac_trend", "hot_pairs", "hot_triples"]
    },
    "快乐8": {
        "code": "kl8",
        "data_file": "data/快乐8_lottery_data.csv",
        "has_blue": False,
        "red_col_prefix": "红球",
        "blue_col_name": None,
        "red_count": 20,
        "blue_count": 0,
        "red_range": (1, 80),
        "blue_range": None,
        "eval_metrics": {"top_n_1": 10, "top_n_2": 20, "green_threshold": 4, "red_threshold": 6},
        "morphology_rules": {
            "sum_range": (680, 940),
            "span_range": (55, 78),
            "ideal_repeats": (4, 8),
            "ideal_consecutive": [2, 3, 4, 5],
            "ideal_odd_counts": [8, 9, 10, 11, 12],
            "ideal_same_tails": [5, 6, 7, 8]
        },
        "supported_charts": [ "red_freq", "blue_freq", "odd_even_ratio", "odd_even_trend",
            "consecutive_dist", "consecutive_trend", "jump_dist", "jump_trend",
            "tail_dist", "tail_trend", "size_ratio", "size_trend",
            "zone_dist", "zone_trend", "repeat_dist", "repeat_trend",
            "neighbor_dist", "neighbor_trend", "isolated_dist", "isolated_trend",
            "sum_trend", "span_trend", "ac_trend", "hot_pairs", "hot_triples"]
    }
}

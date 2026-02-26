# Shared Configuration for Lotteries

# Format for each lottery:
# "Name": {
#     ... metadata ...
#     "eval_metrics": {
#         "top_n_1": N1, 
#         "top_n_2": N2,
#         "green_threshold": M1, # hits needed for green in top_n_1
#         "red_threshold": M2    # hits needed for red in top_n_1
#     }
# }

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
        "supported_charts": [
            "red_freq", "blue_freq", "odd_even_ratio", "odd_even_trend",
            "consecutive_dist", "consecutive_trend", "jump_dist", "jump_trend",
            "tail_dist", "tail_trend", "size_ratio", "size_trend",
            "zone_dist", "zone_trend", "repeat_dist", "repeat_trend",
            "neighbor_dist", "neighbor_trend", "isolated_dist", "isolated_trend",
            "sum_trend", "span_trend", "ac_trend", "hot_pairs", "hot_triples"
        ]
    },
    "福彩3D": {
        "code": "d3",
        "data_file": "data/福彩3D_lottery_data.csv",
        "has_blue": False,
        "red_col_prefix": "红球",
        "blue_col_name": None,
        "red_count": 3,
        "blue_count": 0,
        "red_range": (0, 9),
        "blue_range": None,
        "eval_metrics": {"top_n_1": 3, "top_n_2": 5, "green_threshold": 2, "red_threshold": 3},
        "supported_charts": [ "red_freq", "blue_freq", "odd_even_ratio", "odd_even_trend",
            "consecutive_dist", "consecutive_trend", "jump_dist", "jump_trend",
            "tail_dist", "tail_trend", "size_ratio", "size_trend",
            "zone_dist", "zone_trend", "repeat_dist", "repeat_trend",
            "neighbor_dist", "neighbor_trend", "isolated_dist", "isolated_trend",
            "sum_trend", "span_trend", "ac_trend", "hot_pairs", "hot_triples"]
    },
    "排列三": {
        "code": "pl3",
        "data_file": "data/排列三_lottery_data.csv",
        "has_blue": False,
        "red_col_prefix": "红球",
        "blue_col_name": None,
        "red_count": 3,
        "blue_count": 0,
        "red_range": (0, 9),
        "blue_range": None,
        "eval_metrics": {"top_n_1": 3, "top_n_2": 5, "green_threshold": 2, "red_threshold": 3},
        "supported_charts": [ "red_freq", "blue_freq", "odd_even_ratio", "odd_even_trend",
            "consecutive_dist", "consecutive_trend", "jump_dist", "jump_trend",
            "tail_dist", "tail_trend", "size_ratio", "size_trend",
            "zone_dist", "zone_trend", "repeat_dist", "repeat_trend",
            "neighbor_dist", "neighbor_trend", "isolated_dist", "isolated_trend",
            "sum_trend", "span_trend", "ac_trend", "hot_pairs", "hot_triples"]
    },
    "排列五": {
        "code": "pl5",
        "data_file": "data/排列五_lottery_data.csv",
        "has_blue": False,
        "red_col_prefix": "红球",
        "blue_col_name": None,
        "red_count": 5,
        "blue_count": 0,
        "red_range": (0, 9),
        "blue_range": None,
        "eval_metrics": {"top_n_1": 5, "top_n_2": 10, "green_threshold": 3, "red_threshold": 4},
        "supported_charts": [ "red_freq", "blue_freq", "odd_even_ratio", "odd_even_trend",
            "consecutive_dist", "consecutive_trend", "jump_dist", "jump_trend",
            "tail_dist", "tail_trend", "size_ratio", "size_trend",
            "zone_dist", "zone_trend", "repeat_dist", "repeat_trend",
            "neighbor_dist", "neighbor_trend", "isolated_dist", "isolated_trend",
            "sum_trend", "span_trend", "ac_trend", "hot_pairs", "hot_triples"]
    },
    "超级大乐透": {
        "code": "dlt",
        "data_file": "data/超级大乐透_lottery_data.csv",
        "has_blue": True,
        "red_col_prefix": "红球",
        "blue_col_name": "蓝球", # Data uses 蓝球1, 蓝球2. logic handles this.
        "red_count": 5,
        "blue_count": 2,
        "red_range": (1, 35),
        "blue_range": (1, 12),
        "eval_metrics": {"top_n_1": 5, "top_n_2": 10, "green_threshold": 3, "red_threshold": 4},
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
        "blue_col_name": "篮球", # CSV uses 篮球
        "red_count": 7,
        "blue_count": 1,
        "red_range": (1, 30),
        "blue_range": (1, 30),
        "eval_metrics": {"top_n_1": 7, "top_n_2": 12, "green_threshold": 4, "red_threshold": 5},
        "supported_charts": [ "red_freq", "blue_freq", "odd_even_ratio", "odd_even_trend",
            "consecutive_dist", "consecutive_trend", "jump_dist", "jump_trend",
            "tail_dist", "tail_trend", "size_ratio", "size_trend",
            "zone_dist", "zone_trend", "repeat_dist", "repeat_trend",
            "neighbor_dist", "neighbor_trend", "isolated_dist", "isolated_trend",
            "sum_trend", "span_trend", "ac_trend", "hot_pairs", "hot_triples"]
    },
    "七星彩": {
        "code": "xqxc", 
        "data_file": "data/七星彩_lottery_data.csv",
        "has_blue": True,
        "red_col_prefix": "红球",
        "blue_col_name": "篮球", # CSV uses 篮球
        "red_count": 6,
        "blue_count": 1,
        "red_range": (0, 9),
        "blue_range": (0, 14),
        "eval_metrics": {"top_n_1": 6, "top_n_2": 10, "green_threshold": 3, "red_threshold": 4},
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
        "supported_charts": [ "red_freq", "blue_freq", "odd_even_ratio", "odd_even_trend",
            "consecutive_dist", "consecutive_trend", "jump_dist", "jump_trend",
            "tail_dist", "tail_trend", "size_ratio", "size_trend",
            "zone_dist", "zone_trend", "repeat_dist", "repeat_trend",
            "neighbor_dist", "neighbor_trend", "isolated_dist", "isolated_trend",
            "sum_trend", "span_trend", "ac_trend", "hot_pairs", "hot_triples"]
    }
}

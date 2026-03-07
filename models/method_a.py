import numpy as np
import logging

# --- Pattern Feature Helpers ---

def count_consecutive(numbers):
    """Count consecutive number pairs in a draw."""
    sorted_nums = sorted(numbers)
    count = 0
    for i in range(len(sorted_nums) - 1):
        if sorted_nums[i+1] - sorted_nums[i] == 1:
            count += 1
    return count

def count_neighbor(current_nums, prev_draw_nums):
    """Count numbers that are neighbors (+/- 1) of previous draw."""
    count = 0
    for num in current_nums:
        for prev_num in prev_draw_nums:
            if abs(num - prev_num) == 1:
                count += 1
                break
    return count

def count_repeat(current_nums, recent_history):
    """Count numbers that repeat from recent history (last 3 draws)."""
    recent_nums = set()
    for draw in recent_history:
        recent_nums.update(draw)
    return len(set(current_nums) & recent_nums)

def count_jumps(numbers):
    """Count multi-step jump patterns (2-jump to 6-jump)."""
    sorted_nums = sorted(numbers)
    jump_counts = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    for i in range(len(sorted_nums) - 1):
        diff = sorted_nums[i+1] - sorted_nums[i]
        if diff in jump_counts:
            jump_counts[diff] += 1
    return sum(jump_counts.values())

def predict_similarity(df, model_config, lottery_config):
    """
    1. Find similar fragments in history (Pattern Matching).
    2. Analyze 'Property Range' (Sum, AC, etc.) of the following draw.
    3. Use property distributions to weight frequencies.
    4. Enhanced with pattern features: consecutive, neighbor, repeat, jump.
    """
    logging.info("Method A: 执行模式匹配与属性范围分析（含模式特征）...")

    WINDOW_SIZE = lottery_config['window_size']
    RED_COLS = lottery_config['red_cols']
    NUM_LIST = lottery_config['num_list']
    TOTAL_NUMBERS = lottery_config['total_numbers']

    target_window = df.tail(WINDOW_SIZE)
    target_nums = set(target_window[RED_COLS].values.flatten())
    target_sum = target_window['和值'].mean()
    target_ac = target_window['AC'].mean()

    omission_cols = [f'Omission_{i}' for i in NUM_LIST]
    has_omission = all(c in df.columns for c in omission_cols)
    if has_omission:
        target_omission = target_window.iloc[-1][omission_cols].values.astype(float)

    target_last_draw = target_window.iloc[-1][RED_COLS].values
    target_consecutive = count_consecutive(target_last_draw)
    target_recent_history = [target_window.iloc[i][RED_COLS].values for i in range(max(0, len(target_window)-3), len(target_window))]

    history_matches = []
    conf = model_config
    search_limit = min(conf['search_limit'], len(df) - WINDOW_SIZE - 2)
    for i in range(max(0, len(df) - search_limit), len(df) - WINDOW_SIZE - 1):
        window = df.iloc[i : i + WINDOW_SIZE]
        next_row = df.iloc[i + WINDOW_SIZE]

        win_nums = set(window[RED_COLS].values.flatten())
        overlap = len(target_nums & win_nums)
        sum_err = abs(window['和值'].mean() - target_sum)
        ac_err = abs(window['AC'].mean() - target_ac)

        win_last_draw = window.iloc[-1][RED_COLS].values
        win_consecutive = count_consecutive(win_last_draw)
        consecutive_match = 1.0 / (1.0 + abs(target_consecutive - win_consecutive))

        neighbor_count = count_neighbor(next_row[RED_COLS].values, win_last_draw)

        win_recent_history = [window.iloc[j][RED_COLS].values for j in range(max(0, len(window)-3), len(window))]
        repeat_count = count_repeat(next_row[RED_COLS].values, win_recent_history)

        jump_count = count_jumps(next_row[RED_COLS].values)
        target_jump_count = count_jumps(target_last_draw)
        jump_match = 1.0 / (1.0 + abs(target_jump_count - jump_count))

        score = (overlap * conf['weights']['overlap'] -
                 sum_err * conf['weights']['sum'] -
                 ac_err * conf['weights']['ac'] +
                 consecutive_match * conf['weights']['consecutive'] +
                 neighbor_count * conf['weights']['neighbor'] +
                 repeat_count * conf['weights']['repeat'] +
                 jump_match * conf['weights']['jump'])

        if has_omission:
            win_omission = window.iloc[-1][omission_cols].values.astype(float)
            omiss_err = np.mean(np.abs(target_omission - win_omission))
            omiss_match = 1.0 / (1.0 + omiss_err)
            score += omiss_match * conf['weights']['omission']

        history_matches.append({
            'score': score,
            'draw': next_row[RED_COLS].values,
            'sum_val': next_row['和值'],
            'ac_val': next_row['AC'],
            'span': next_row['跨度']
        })

    history_matches.sort(key=lambda x: x['score'], reverse=True)
    top_matches = history_matches[:conf['top_matches']]

    avg_next_sum = np.mean([m['sum_val'] for m in top_matches])
    avg_next_ac = np.mean([m['ac_val'] for m in top_matches])
    logging.info(f"模式匹配预测下期属性: 预计和值(avg)={avg_next_sum:.1f}, AC(avg)={avg_next_ac:.1f}")

    pred_counts = np.zeros(TOTAL_NUMBERS)
    for m in top_matches:
        for n in m['draw']:
            try:
                idx = NUM_LIST.index(int(n))
                pred_counts[idx] += 1.0
            except (ValueError, IndexError):
                pass

    probs = pred_counts / pred_counts.sum() if pred_counts.sum() > 0 else np.zeros(TOTAL_NUMBERS)
    return probs
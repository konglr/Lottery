from funcs.ball_filter import get_morphology_report
last_nums = [3, 13, 25, 26, 30, 31]
p1 = [3, 16, 19, 20, 26, 30]
p4 = [7, 16, 19, 20, 26, 30]
print("--- Result P1 (3 repeats) ---")
print(get_morphology_report(p1, last_nums))
print("\n--- Result P4 (2 repeats) ---")
print(get_morphology_report(p4, last_nums))

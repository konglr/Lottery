
import numpy as np
import pandas as pd
from KL8_data_import import result_df

def calculate_hit_rate(row_index, result_df, ahead=5, select_number=10, method='Top'):
    # Calculate the ball numbers in the previous n rows
    previous_ball_numbers = result_df.iloc[row_index - ahead:row_index, -20:]

    # Count how many unique numbers appeared in the previous n rows
    unique_numbers = previous_ball_numbers.stack().unique()

    # Calculate occurrence counts and frequency for each number
    previous_ball_numbers_array = previous_ball_numbers.values.flatten()
    occurrence_df = pd.Series(previous_ball_numbers_array).value_counts().sort_index().reset_index()
    occurrence_df.columns = ['number', 'frequency']
    occurrence_df = occurrence_df.sort_values(by='frequency', ascending=False).reset_index()

    # Calculate the total number of unique numbers
    total_unique_numbers = len(occurrence_df)

    # Select the numbers based on the specified method
    if method == 'Top':
        selected_numbers = occurrence_df.head(select_number)['number']
    elif method == 'Mid':
        start_index = (total_unique_numbers - select_number) // 2
        selected_numbers = occurrence_df.iloc[start_index:start_index + select_number]['number']
    else:
        raise ValueError("Invalid method. Method should be 'Top' or 'Mid'.")

    # Get the ball numbers for the current index
    current_ball_numbers = result_df.iloc[row_index, -20:].astype(int)

    # Calculate the number of matches between the selected numbers and current ball numbers
    matches = current_ball_numbers.isin(selected_numbers).sum()
    hit_rate = matches / 20

    # Find the matched numbers between the selected numbers and current ball numbers
    matched_balls = current_ball_numbers[current_ball_numbers.isin(selected_numbers)].index.tolist()
    matched_ball_numbers = current_ball_numbers[current_ball_numbers.isin(selected_numbers)].tolist()

    return {
        "unique_numbers_count": len(unique_numbers),
        "occurrence_counts": occurrence_df,
        "selected_numbers": selected_numbers.tolist(),
        "hit_numbers": matches,
        "hit_rate": hit_rate,
        "matched_balls": matched_balls,
        "matched_ball_numbers": matched_ball_numbers
    }



# Example usage:
row_index = 10 # Modify this with the desired row index
result = calculate_hit_rate(row_index, result_df, 10, 10, method= "Top")
print("Unique Numbers Count in Previous 5 Rows:", result["unique_numbers_count"])
print("Occurrence Counts for Each Number:")
print(result["occurrence_counts"])
print("Selected Numbers based on Middle 20 Ranking:", result["selected_numbers"])
print("Hit Rate:", result["hit_rate"])
print("Matched Balls:", result["matched_balls"])
print("Matched Ball Numbers:", result["matched_ball_numbers"])

# Example usage: check 1000 records hit rate of the selectiong method

hit_rates = []
hit_numbers= []

for row_index in range(1000, 1157):
    result = calculate_hit_rate(row_index, result_df, ahead=5, select_number=20,method= "Top")
    hit_rates.append(result["hit_rate"])
    hit_numbers.append(result["hit_numbers"])
# Print or use hit_rates as needed
print("Hit Rates from index 10 to 1000:", hit_rates)
print("Hit numbers from index 10 to 1000:", hit_numbers)


# Create two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot hit_rate in the first subplot
ax1.plot(range(1000, 1157), hit_rates, label='Hit Rate', color='blue')
ax1.set_title('Hit Rate Over Time')
ax1.set_xlabel('Index')
ax1.set_ylabel('Hit Rate')
ax1.legend()
ax1.grid(True)

# Plot hit_numbers in the second subplot
ax2.plot(range(1000, 1157), hit_numbers, label='Hit Numbers', color='red')
ax2.set_title('Hit Numbers Over Time')
ax2.set_xlabel('Index')
ax2.set_ylabel('Hit Numbers')
ax2.legend()
ax2.grid(True)

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()
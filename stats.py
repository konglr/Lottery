import numpy as np
import pandas as pd
from KL8_data_import import data_np,result_df


# Extract the first 20 numbers for Group 1
group_1 = data_np[-20:]
num_groups = int(len(data_np) / 20 - 1)
# Initialize a list to store re-occurrences for each group
re_occurrences_list = []

# Iterate through the subsequent groups
for i in range(2, num_groups-1):  # Assuming there are 6 groups in total
    # Extract the current group of 20 numbers
    current_group = data_np[-(i * 20): -(i - 1) * 20]

    # Find the common numbers between the previous group and the current group
    common_numbers = np.intersect1d(group_1, current_group)

    # Add the common numbers to the re-occurrences list
    re_occurrences_list.append(common_numbers)

    # Update the previous group to include the current group for the next iteration
    group_1 = current_group

# Print the re-occurrences list for each group
for i, re_occurrences in enumerate(re_occurrences_list, start=1):

    print(f"Re-occurrences for Group {i}: {len(re_occurrences)}")

correlation_matrix = result_df[['ball_1', 'ball_2', 'ball_3', 'ball_4', 'ball_5', 'ball_6', 'ball_7', 'ball_8', 'ball_9', 'ball_10', 'ball_11', 'ball_12', 'ball_13', 'ball_14', 'ball_15', 'ball_16', 'ball_17', 'ball_18', 'ball_19', 'ball_20']].corr()
print(correlation_matrix)

pd.plotting.scatter_matrix(correlation_matrix, figsize=(12, 12))
plt.show()




# Create a list to store the lengths of re_occurrences lists
re_occurrences_lengths = [len(re_occurrences) for re_occurrences in re_occurrences_list]

# Plot the frequency chart
plt.figure(figsize=(10, 6))
plt.hist(re_occurrences_lengths, bins=range(0, max(re_occurrences_lengths) + 2), color='skyblue', edgecolor='black')
plt.xlabel('Length of Re-Occurrences List')
plt.ylabel('Frequency')
plt.title('Frequency Chart of Length of Re-Occurrences Lists')
plt.xticks(range(0, max(re_occurrences_lengths) + 1))
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add frequency times and percentage to each bar
for i, bin_count in enumerate(plt.hist(re_occurrences_lengths, bins=range(0, max(re_occurrences_lengths) + 2))[0]):
    if bin_count != 0:
        plt.text(i + 0.5, bin_count, f'{int(bin_count)}\n{bin_count / len(re_occurrences_lengths) * 100:.2f}%', ha='center', va='bottom')

plt.show()



# Generate x-axis values (group numbers)
group_numbers = range(1, 100)

# Plot the line chart
plt.figure(figsize=(10, 6))
plt.plot(group_numbers, re_occurrences_lengths[0:99], marker='o', color='skyblue', linestyle='-')
plt.xlabel('Group Number')
plt.ylabel('Length of Re-Occurrences List')
plt.title('Trends of Changing Length of Re-Occurrences Lists Over Time')
plt.grid(axis='y', linestyle='--')  # Remove vertical gridlines
plt.xticks(group_numbers)
plt.show()

# Extract the first 20 numbers for Group 1
group_1 = data_np[-20:]

# Initialize a dictionary to store common numbers in each subsequent group
common_numbers_dict = {"total_common_numbers": []}

# Iterate through the next groups
for i in range(1, 6):
    # Extract the current group of 20 numbers
    current_group = data_np[ -((i + 1) * 20):-(i * 20)]

    # Find the common numbers between Group 1 and the current group
    common_numbers = np.intersect1d(group_1, current_group)

    # First occurred numbers between Group 1 and the current group
    first_occurred_numbers = np.intersect1d(np.setdiff1d(group_1, common_numbers_dict["total_common_numbers"]), current_group)

    # Add common numbers to the total_common_numbers list
    common_numbers_dict["total_common_numbers"].extend(common_numbers)

    # Store common numbers and first occurred numbers for each group
    common_numbers_dict[f"Group {i + 1}"] = {
        "total_common_number": len(common_numbers),
        "common_numbers": common_numbers,
        "first_occurred_numbers": first_occurred_numbers
    }

# Display the total common numbers, individual common numbers, and first occurred numbers for each group
for group, common_info in common_numbers_dict.items():
    if group != "total_common_numbers":
        total_common_number = common_info["total_common_number"]
        common_numbers = common_info["common_numbers"]
        first_occurred_numbers = common_info["first_occurred_numbers"]
        print(f"{group}: Total Common Numbers {total_common_number}, Common Numbers {common_numbers}, First Occurred Numbers {first_occurred_numbers}")
    else:
        total_common_numbers = common_info
        print(f"Total Common Numbers across all groups: {total_common_numbers}")

        # Convert the dictionary to a Pandas DataFrame
        df = pd.DataFrame.from_dict(common_numbers_dict, orient='index')

        # Print the column names to check the actual names in your DataFrame
        print(df.columns)

        # Extract the values under the 'total_common_numbers' column
        total_common_numbers_array = df.loc['total_common_numbers'].values

        # Calculate the frequency of each unique value in the array
        unique_values, counts = np.unique(total_common_numbers_array, return_counts=True)

        # Display the unique values and their frequencies
        for value, count in zip(unique_values, counts):
            print(f"Total Common Numbers {value}: Frequency {count}")

    # Extract the first 100 numbers
    first_100_numbers = data_np[-120:-20]

    # Calculate the frequency of each unique value in the array
    unique_values, counts = np.unique(first_100_numbers, return_counts=True)

    # Create a DataFrame to store the counts
    counts_df = pd.DataFrame({'Number': unique_values, 'Frequency': counts})

    # Sort the DataFrame by frequency in descending order
    sorted_counts_df = counts_df.sort_values(by='Frequency', ascending=False)

    # Get the top 10 high-frequency numbers as an array
    top_10_high_frequency_numbers = sorted_counts_df.head(20)['Number'].values
    # Display the top 10 high-frequency numbers
    print("Top 10 High-Frequency Numbers:")
    print(top_10_high_frequency_numbers)

    # Calculate how many top numbers are present in Group 1
    common_numbers_in_group_1 = np.intersect1d(top_10_high_frequency_numbers, group_1)

    # Calculate the winning rate
    winning_rate = len(common_numbers_in_group_1) / len(top_10_high_frequency_numbers) * 100

    # Display the results
    print("Top 10 High-Frequency Numbers:")
    print(top_10_high_frequency_numbers)
    print("\nGroup 1 Numbers:")
    print(group_1)
    print("\nCommon Numbers in Top 10 and Group 1:")
    print(common_numbers_in_group_1)
    print("\nWinning Rate:")
    print(f"{winning_rate:.2f}%")

    # Display the first 100 numbers and their frequencies
    print("First 100 Numbers:")
    print(first_100_numbers)
    print("\nFrequency of Numbers:")
    for value, count in zip(unique_values, counts):
        print(f"Number {value}: Frequency {count}")



import pandas as pd
from KL8_data_import import result_df


def calculate_occurrence_intervals(start_index, end_index):
    # Create an empty DataFrame to store the occurrences and intervals
    intervals_df = pd.DataFrame(columns=['Number', 'Interval'])

    # Iterate through each number
    for number in range(1, 81):  # Adjusted the range to iterate from 1 to 20
        occurrence_indices = []

        # Iterate through rows to find occurrences of the number
        for i in range(start_index, end_index + 1):
            if number in result_df.iloc[i, -20:].values:
                occurrence_indices.append(i)
                #print(number)
                #print(occurrence_indices)

        # Calculate intervals
        intervals = []

        if len(occurrence_indices) == 0:
            intervals.append(end_index-start_index) # if the number dosn't occured the interval is more than the search scope
        elif len(occurrence_indices) == 1:
            interval = occurrence_indices[0] - start_index
            intervals.append(interval)
        else:
            for j in range(1, len(occurrence_indices)):
                interval = occurrence_indices[j] - occurrence_indices[j - 1]
                intervals.append(interval)



       # print(intervals)

        # Calculate average interval
        if intervals:
            average_interval = sum(intervals) / len(intervals)
        else:
            average_interval = None

        times = len(occurrence_indices)


        # Append the number and its average interval to intervals_df
        intervals_df = intervals_df._append({'Number': number,  'Times': times}, ignore_index=True)

    return intervals_df


# Example usage:

end_index = 1222
start_index = end_index - 30
intervals_df = calculate_occurrence_intervals(start_index, end_index)
print(intervals_df)
